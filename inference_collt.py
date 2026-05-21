"""
COLLT 流式推理(论文 §3.1.5 Algorithm 1)。

Token 级伪代码:
    1) 读取首 token,若 == <CLR> → 直接转发整段澄清问句、终止本轮,等用户补充。
    2) 否则期望 <DRT>。在工具预算 |τ|<2 且尚未输出 <ER> 的状态下:
         · 若出现 head tag <X>:实际调用 serve_tools.dispatch(X) 取回 g_t,
           把 g_t + </X> 写回 stream;**同时丢弃模型自己在 <X>…</X> 之间
           生成的内容**(它只是占位幻觉);τ 计数 +1。
         · 若出现 <ER>:跳出工具循环。
    3) 后续 token 直接转发,生成 enhanced response。

设计要点
--------
* `TextIteratorStreamer` 实时拿 token;
* 不做二次 generate——只在文本流层面把"幻觉工具结果"剥掉、注入真实结果;
* `TOOL_BUDGET = 2`(Proposition 1);
* `wrap_with_tool_ablation()`:R3#7 程序化消融,把指定工具的输出强制清空。
"""
from __future__ import annotations
import os, re, threading
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

from .common import TOOL_NAMES, TOOL_BUDGET, ER_TOKEN
from .tools.serve_tools import dispatch, order_by_priority


HEAD_RE = re.compile(r"<(SCR|LAS|LCP|LER|LED|NET)>")
CLR_TAG = "<CLR>"
DRT_TAG = "<DRT>"
ALL_TOOLS = set(TOOL_NAMES)
_LOOKAHEAD = 16          # 防止 head/tail tag 跨 token 截断


@dataclass
class COLLTGenConfig:
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.05


class COLLTRunner:
    """把 HF causal-LM + tokenizer 包装为符合 Algorithm 1 的流式生成器。"""

    def __init__(self, model, tokenizer, gen_cfg: Optional[COLLTGenConfig] = None):
        self.model = model
        self.tok   = tokenizer
        self.cfg   = gen_cfg or COLLTGenConfig()

    # ────────────────────── public API ───────────────────────────────
    def stream(self, messages: list[dict]) -> Iterator[str]:
        """逐段产出 assistant 输出(已注入真实工具结果)。"""
        state          = "pre_action"     # pre_action → drt → er
        budget_used    = 0
        in_tool: str | None = None         # 正在被丢弃的 <X> 段
        buf            = ""
        used_tools: list[str] = []
        context: dict[str, str] = {}

        chunks = self._raw_stream(messages)

        def _emit(piece: str):
            return piece

        for chunk in chunks:
            buf += chunk

            # 进入循环:能消多少消多少
            progressed = True
            while progressed:
                progressed = False

                # ── 1) 还没看到动作 token ─────────────────────────
                if state == "pre_action":
                    clr_i = buf.find(CLR_TAG)
                    drt_i = buf.find(DRT_TAG)
                    # 谁先到
                    if clr_i != -1 and (drt_i == -1 or clr_i < drt_i):
                        # 把 <CLR> 之前的(通常是空)和 <CLR> 一起吐出
                        yield _emit(buf[: clr_i + len(CLR_TAG)])
                        buf = buf[clr_i + len(CLR_TAG):]
                        state = "clr"
                        progressed = True
                        continue
                    if drt_i != -1:
                        yield _emit(buf[: drt_i + len(DRT_TAG)])
                        buf = buf[drt_i + len(DRT_TAG):]
                        state = "drt"
                        progressed = True
                        continue
                    # 都没等到,保留 lookahead
                    if len(buf) > _LOOKAHEAD:
                        # 但又不能直接吐出可能含半个 <CLR/DRT> 的内容
                        yield _emit(buf[:-_LOOKAHEAD])
                        buf = buf[-_LOOKAHEAD:]
                    break

                # ── 2) CLR 分支:全部转发,本轮结束(由调用方等用户) ─
                if state == "clr":
                    if buf:
                        yield _emit(buf)
                        buf = ""
                    break

                # ── 3) DRT 分支:正在丢弃 <X> 段中模型的幻觉文本 ──
                if in_tool is not None:
                    tail = f"</{in_tool}>"
                    idx = buf.find(tail)
                    if idx == -1:
                        # 保留 lookahead 等 tail
                        if len(buf) > len(tail):
                            buf = buf[-len(tail):]
                        break
                    # 越过 tail,继续主流
                    buf = buf[idx + len(tail):]
                    in_tool = None
                    progressed = True
                    continue

                # ── 4) 主流:寻找 head tag / <ER> / 普通文本 ─────
                head_m = HEAD_RE.search(buf) if budget_used < TOOL_BUDGET and state == "drt" else None
                er_i   = buf.find(ER_TOKEN) if state == "drt" else -1

                # 4a) head 比 <ER> 先到 → 调用工具
                if head_m and (er_i == -1 or head_m.start() < er_i):
                    name = head_m.group(1)
                    # 先吐出 head 之前 + 头标签本身
                    yield _emit(buf[: head_m.end()])
                    buf = buf[head_m.end():]
                    # 按 R3#6 优先级把已用工具+当前工具排序,确认是否需要重排
                    ordered = order_by_priority(used_tools + [name])
                    # 这里 ordered 仅用于尊重 priority(若优先工具未跑则可在此插入)
                    # 简化处理:严格按模型给出的顺序逐个调用即可
                    q = messages[-1]["content"] if messages else ""
                    g = dispatch(name, q, history=messages, context=context)
                    context[name] = g
                    yield _emit(g)
                    yield _emit(f"</{name}>")
                    used_tools.append(name)
                    budget_used += 1
                    in_tool = name  # 进入丢弃阶段,直到模型自己写出 </X>
                    progressed = True
                    continue

                # 4b) <ER> 先到 → 结束工具阶段
                if er_i != -1:
                    yield _emit(buf[: er_i + len(ER_TOKEN)])
                    buf = buf[er_i + len(ER_TOKEN):]
                    state = "er"
                    progressed = True
                    continue

                # 4c) 普通文本:保留 lookahead 防止 head/tail 半截
                if state == "er":
                    if buf:
                        yield _emit(buf)
                        buf = ""
                    break
                if len(buf) > _LOOKAHEAD:
                    yield _emit(buf[:-_LOOKAHEAD])
                    buf = buf[-_LOOKAHEAD:]
                break

        # ── 流结束:倾泻剩余 ───────────────────────────────────────
        if state == "clr":
            if buf:
                yield buf
        elif in_tool is not None:
            # 模型流断在工具段中间,直接合上 </X> 收尾,丢掉残段
            yield f"</{in_tool}>"
        else:
            if buf:
                yield buf

    def generate(self, messages: list[dict]) -> str:
        return "".join(self.stream(messages))

    # ────────────────────── token streaming ──────────────────────────
    def _raw_stream(self, messages: list[dict]) -> Iterator[str]:
        from transformers import TextIteratorStreamer
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tok, skip_prompt=True, skip_special_tokens=False
        )
        kw = dict(
            **inputs,
            streamer            = streamer,
            max_new_tokens      = self.cfg.max_new_tokens,
            temperature         = self.cfg.temperature,
            top_p               = self.cfg.top_p,
            do_sample           = self.cfg.do_sample,
            repetition_penalty  = self.cfg.repetition_penalty,
        )
        thread = threading.Thread(target=self.model.generate, kwargs=kw)
        thread.start()
        for piece in streamer:
            yield piece
        thread.join()


# ── R3#7:程序化工具消融 ──────────────────────────────────────────
def wrap_with_tool_ablation(stream_fn, ablated: set[str]):
    """把任意 `stream_fn(messages) -> Iterator[str]` 包装成:对 `ablated`
    集合内的工具,把 `<X>g</X>` 重写为 `<X></X>`(等价于 serve_tools
    路径上设置 COLLT_ABLATE=X)。"""
    ablated = {t.upper() for t in ablated}
    if not ablated:
        return stream_fn
    pat = re.compile(
        r"<(" + "|".join(re.escape(t) for t in ablated) + r")>(.*?)</\1>",
        re.DOTALL,
    )

    def wrapped(messages):
        buf = ""
        for piece in stream_fn(messages):
            buf += piece
        # 在尾部一次性重写;调用方拿到的是单段字符串(失去 token-by-token)
        yield pat.sub(lambda m: f"<{m.group(1)}></{m.group(1)}>", buf)

    return wrapped


# ── 加载训练后的 COLLT-* checkpoint ──────────────────────────────
def load_runner(ckpt_dir: str) -> COLLTRunner:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        ckpt_dir, trust_remote_code=True,
        torch_dtype="auto", device_map="auto",
    )
    return COLLTRunner(mdl, tok)


# ── 通用基线 runner(给裸大模型也注入工具能力,响应 R3#2) ──────
TOOLING_SYSTEM_PROMPT = (
    "你是一名严谨的中国法律咨询助手,并被授权调用以下 6 种法律工具:\n"
    "  <SCR></SCR> 相似案例检索 / <LAS></LAS> 法条检索 / <LCP></LCP> 罪名预测\n"
    "  <LER></LER> 法律要素识别 / <LED></LED> 法律事件检测 / <NET></NET> 互联网搜索\n"
    "调用约定:\n"
    "  1) 若用户问题信息不全,需要先澄清,请以 <CLR> 开头给出澄清问句,本轮在 <CLR> 段结束;\n"
    "  2) 信息充分时,以 <DRT> 开头,可在回答中插入最多 2 个工具调用,每个调用用其头/尾标签包裹;\n"
    "  3) 完成所有调用后输出 <ER>,在 <ER> 之后再写最终答复。\n"
    "请严格遵循上述特殊 token,避免遗漏。"
)


class BaselineToolRunner:
    """非 COLLT 训练的基线 LLM:通过 system prompt 也具备工具调用能力,
    使 Table 3 / 工具消融能与 COLLT-* 进行公平对照(响应审稿人 R3#2)。

    工具调用解析与 COLLTRunner 完全一致,只是 system prompt 在外部加。"""

    def __init__(self, ckpt_dir: str, gen_cfg: Optional[COLLTGenConfig] = None,
                 tool_prompt: str = TOOLING_SYSTEM_PROMPT,
                 with_clarify: bool = True):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            ckpt_dir, trust_remote_code=True,
            torch_dtype="auto", device_map="auto",
        )
        # 把特殊 token 加进 tokenizer,这样基线也能输出 <CLR>/<DRT>/<X>...
        from .common import SPECIAL_TOKENS
        added = tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
        if added:
            mdl.resize_token_embeddings(len(tok))
        self.runner = COLLTRunner(mdl, tok, gen_cfg)
        self.tool_prompt = tool_prompt
        self.with_clarify = with_clarify

    def _wrap_messages(self, messages: list[dict]) -> list[dict]:
        # 若调用方已经给了 system,就保留;否则注入
        if messages and messages[0].get("role") == "system":
            return messages
        sys_content = self.tool_prompt
        if not self.with_clarify:
            # 去掉澄清条款,仅给工具能力
            sys_content = sys_content.replace(
                "  1) 若用户问题信息不全,需要先澄清,请以 <CLR> 开头给出澄清问句,本轮在 <CLR> 段结束;\n",
                "",
            )
        return [{"role": "system", "content": sys_content}, *messages]

    def stream(self, messages: list[dict]) -> Iterator[str]:
        return self.runner.stream(self._wrap_messages(messages))

    def generate(self, messages: list[dict]) -> str:
        return "".join(self.stream(messages))


# ── CLI smoke test ───────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--query", default="公司拖欠我三个月工资,我准备辞职,能要求赔偿吗?")
    p.add_argument("--ablate", default="", help="逗号分隔的工具名,如 LAS,SCR;ALL=全关")
    p.add_argument("--baseline", action="store_true",
                   help="作为裸基线(注入工具 system prompt)运行")
    args = p.parse_args()
    if args.ablate:
        os.environ["COLLT_ABLATE"] = args.ablate

    if args.baseline:
        runner = BaselineToolRunner(args.ckpt)
    else:
        runner = load_runner(args.ckpt)
    for piece in runner.stream([{"role": "user", "content": args.query}]):
        sys.stdout.write(piece); sys.stdout.flush()
    print()
