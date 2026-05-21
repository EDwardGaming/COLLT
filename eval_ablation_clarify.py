"""
澄清能力消融实验(R3#8 + 审稿人 3 第 8 条)。

审稿人主张:让裸大模型在 system prompt 中说"如果用户问题模糊请澄清",
也许就足以匹敌 COLLT 的复杂 SFT。我们用同一组 AmbigLegalQA 样本,做三组
对照,在指标层面客观回答这个问题:

  (a) base_vanilla            裸基座,无任何澄清提示,无工具
  (b) base_with_clarify       裸基座 + "如有模糊请澄清"的 system prompt(可用工具)
  (c) collt_sft               COLLT-* SFT 指令微调模型(工具能力来自微调)

三组都跑同一份 AmbigLegalQA,统计:
  · trigger-F1   —— 模型是否在该澄清时确实输出 <CLR>
  · coverage     —— 模型的澄清问题覆盖了多少 gold 关键点
  · ROUGE-L      —— 经过用户补充信息后的最终答复 vs gold final response

为保证对照公平,(a)(b)(c) 都共享 inference_collt 的流式 + 工具调度,
唯一区别在 system prompt 是否注入。审稿人 R3#2 也得到响应:基线在(b)模式
下同样能调工具,工具能力不再是 COLLT 的"独占优势"。

用法:
    python -m train.eval_ablation_clarify \\
        --collt_ckpt checkpoints/collt-qwen \\
        --base_ckpt  Qwen/Qwen2.5-7B-Instruct \\
        --out outputs/ablation_clarify_qwen.json
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

from .common import AMBIG_LEGAL_QA, OUT_DIR, jsonl_iter
from .inference_collt import load_runner, BaselineToolRunner
from .eval_ambiglegalqa import trigger_f1, coverage, rouge_l_final


# 澄清优先的 system prompt (审稿人 R3#8 推荐的"低成本基线")
CLARIFY_FIRST_SYSTEM = (
    "你是一名严谨的中国法律咨询助手,并被授权调用以下 6 种法律工具:\n"
    "  <SCR></SCR> 相似案例检索 / <LAS></LAS> 法条检索 / <LCP></LCP> 罪名预测\n"
    "  <LER></LER> 法律要素识别 / <LED></LED> 法律事件检测 / <NET></NET> 互联网搜索\n"
    "**澄清准则**:若用户问题信息不全或存在歧义,请先输出 <CLR> 然后给出 1~5 条澄清问句,"
    "在 <CLR> 段结束本轮,等待用户补充。\n"
    "信息充足时,以 <DRT> 开头,可在回答中插入最多 2 个工具调用,每个调用用其头/尾标签包裹;"
    "完成所有调用后输出 <ER>,再写最终答复。"
)

CLR_RE = re.compile(r"<CLR>", re.I)


def _run(runner, records):
    """对每条样本走"首轮 → (若 CLR 则注入 gold 补充再走一次)"。"""
    res = []
    for r in records:
        msgs = [{"role": "user", "content": r["question"]}]
        t1 = runner.generate(msgs)
        msgs.append({"role": "assistant", "content": t1})
        if CLR_RE.search(t1) and r.get("gold_supplements"):
            msgs.append({"role": "user", "content": r["gold_supplements"][0]})
            t2 = runner.generate(msgs)
        else:
            t2 = t1
        res.append({**r, "model_first_turn": t1, "model_final": t2})
    return res


def _metrics(runs):
    return {
        "trigger_f1":          round(trigger_f1(runs),    4),
        "clarification_cover": round(coverage(runs),      4),
        "rouge_l_final":       round(rouge_l_final(runs), 4),
        "n":                   len(runs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collt_ckpt", required=True,
                    help="COLLT-* SFT checkpoint (c 组)")
    ap.add_argument("--base_ckpt",  required=True,
                    help="未微调的基座模型 HF id 或本地路径 (a / b 组共用)")
    ap.add_argument("--data", default=str(AMBIG_LEGAL_QA))
    ap.add_argument("--out",  default=str(OUT_DIR / "ablation_clarify.json"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    records = list(jsonl_iter(args.data))
    if args.limit:
        records = records[:args.limit]
    print(f"[data] {len(records)} samples")

    report = {}

    # (a) 真正的裸基线:无任何 system,无工具能力
    base_plain = BaselineToolRunner(args.base_ckpt, tool_prompt="", with_clarify=False)
    report["base_vanilla"]      = _metrics(_run(base_plain, records))

    # (b) 裸基线 + 澄清提示词(工具也开放,与 COLLT 工具能力对齐)
    base_clarify = BaselineToolRunner(args.base_ckpt,
                                      tool_prompt=CLARIFY_FIRST_SYSTEM,
                                      with_clarify=True)
    report["base_with_clarify"] = _metrics(_run(base_clarify, records))

    # (c) COLLT SFT
    collt_runner = load_runner(args.collt_ckpt)
    report["collt_sft"]         = _metrics(_run(collt_runner, records))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
