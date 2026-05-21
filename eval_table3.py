"""
复现论文 Table 3 —— 在 9 个传统法律 NLP 任务上做零样本推理对比。

任务 ↔ LawBench 数据集映射(对应论文 Table 2):

    LCP   Legal Charge Prediction        LawBench 3-3  F1
    LAP   Legal Article Prediction       LawBench 3-1  F1
    PTP   Prison Term Prediction         LawBench 3-4  log-distance
    AM    Argument Mining                LawBench 2-8  Accuracy
    DFI   Dispute Focus Identification   LawBench 2-2  F1
    ITI   Issue Topic Identification     LawBench 2-4  Accuracy
    LED   Legal Event Detection          LawBench 2-9  F1
    OS    Opinion Summarization          LawBench 2-7  ROUGE-L
    CA    Case Analysis                  LawBench 3-6  Accuracy

评估数据统一来自本地 local_data/LawBench(git clone open-compass/LawBench)。

两条评估路径:
  · `--backend local`(默认):直接读取 LawBench JSON 文件,用指定 checkpoint
    推理并计分,无需网络。
  · `--backend lawbench`:调用 LawBench 提供的 CLI,本机需先 `git clone
    open-compass/LawBench` 并设置 `LAWBENCH_ROOT`。

为响应审稿人 R3#2(基线 LLM 当年缺乏联网/工具能力,对照不公平):

  · `--mode collt`(默认):加载 COLLT-* SFT 模型,流式推理时由
    `inference_collt.COLLTRunner` 真实调用工具。
  · `--mode baseline_tools`:加载 *未微调* 的基座大模型,通过 system prompt
    赋予同样的工具调用能力(`inference_collt.BaselineToolRunner`),与 COLLT
    使用同一套 `serve_tools.dispatch`,以保证基线和 COLLT 工具能力对等。
  · `--mode baseline_plain`:纯零样本,无工具,作为原始基线。

工具消融:`--ablate LAS,SCR,...`(或 `--ablate ALL`)透传 `COLLT_ABLATE`。
"""
from __future__ import annotations
import argparse, json, math, os, re, subprocess, sys
from pathlib import Path

from .common import OUT_DIR, LAWBENCH_DIR
from .inference_collt import load_runner, BaselineToolRunner


TABLE3_TASKS = ["LCP", "LAP", "PTP", "AM", "DFI", "ITI", "LED", "OS", "CA"]

# task → LawBench zero_shot file id
TASK_TO_LAWBENCH: dict[str, str] = {
    "LCP": "3-3",   # 罪名预测  F1
    "LAP": "3-1",   # 法条预测  F1
    "PTP": "3-4",   # 刑期预测  log-distance
    "AM":  "2-8",   # 论点挖掘  Accuracy
    "DFI": "2-2",   # 纠纷焦点  F1
    "ITI": "2-4",   # 问题主题  Accuracy
    "LED": "2-9",   # 事件检测  F1
    "OS":  "2-7",   # 舆情摘要  ROUGE-L
    "CA":  "3-6",   # 案例分析  Accuracy
}


# ──────────────────────────── metrics ──────────────────────────────
def f1_micro(preds: list[str], golds: list[str]) -> float:
    tp = fp = fn = 0
    for p, g in zip(preds, golds):
        ps = {x.strip() for x in p.replace("、", ",").split(",") if x.strip()}
        gs = {x.strip() for x in g.replace("、", ",").split(",") if x.strip()}
        tp += len(ps & gs); fp += len(ps - gs); fn += len(gs - ps)
    if tp + fp == 0 or tp + fn == 0: return 0.0
    p = tp / (tp + fp); r = tp / (tp + fn)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def accuracy(preds: list[str], golds: list[str]) -> float:
    return sum(p.strip() == g.strip() for p, g in zip(preds, golds)) / max(1, len(preds))


def log_distance(preds: list[float], golds: list[float]) -> float:
    """CAIL 刑期预测使用的指标 d = mean(|log(p+1) - log(g+1)|),越小越好。"""
    if not preds: return 0.0
    diffs = [abs(math.log1p(max(0,p)) - math.log1p(max(0,g))) for p, g in zip(preds, golds)]
    return sum(diffs) / len(diffs)


def rouge_l(preds: list[str], golds: list[str]) -> float:
    def lcs(a: str, b: str) -> int:
        m, n = len(a), len(b)
        dp = [0] * (n + 1)
        for i in range(1, m + 1):
            prev = 0
            for j in range(1, n + 1):
                cur = dp[j]
                if a[i-1] == b[j-1]: dp[j] = prev + 1
                else:                 dp[j] = max(dp[j], dp[j-1])
                prev = cur
        return dp[n]
    f_sum = 0.0
    for p, g in zip(preds, golds):
        l = lcs(p, g)
        if l == 0: continue
        prec = l / len(p); rec = l / len(g)
        f_sum += 2 * prec * rec / (prec + rec)
    return f_sum / max(1, len(preds))


# ───────────────────────── task adapters ───────────────────────────
def _prompt_for(_task: str, item: dict) -> str:
    """Build prompt from LawBench item (instruction + question)."""
    inst = item.get("instruction", "").strip()
    q    = item.get("question", "").strip()
    return f"{inst}\n{q}" if inst else q


def _extract_answer(task: str, text: str) -> str:
    """Normalize both gold answers and model predictions to comparable form.

    LawBench gold answers use "KEY:value" format.
    Model predictions (following LawBench instruction) use "[KEY]value<eoa>".
    Both are reduced to the bare value string.
    """
    t = (text or "").strip()
    if task == "LCP":
        # "罪名:X;Y" or "[罪名]X;Y<eoa>"
        m = re.search(r"罪名[:：](.*?)(?:<eoa>|$)", t)
        if not m:
            m = re.search(r"\[罪名\](.*?)(?:<eoa>|$)", t)
        return m.group(1).strip() if m else t
    if task == "LAP":
        m = re.search(r"法条[:：](.*?)(?:<eoa>|$)", t)
        if not m:
            m = re.search(r"\[法条\](.*?)(?:<eoa>|$)", t)
        return m.group(1).strip() if m else t
    if task == "PTP":
        # "刑期:4个月" → "4"
        m = re.search(r"刑期[:：]\s*([0-9]+)", t)
        if not m:
            m = re.search(r"\[刑期\]\s*([0-9]+)", t)
        if not m:
            m = re.search(r"([0-9]+)\s*个月", t)
        return m.group(1).strip() if m else t
    if task in ("AM", "CA"):
        # "[正确答案]C<eoa>" or "正确答案:C。"
        m = re.search(r"正确答案[:：]\s*([A-Ea-e])", t)
        if not m:
            m = re.search(r"\[(?:正确答案|答案)\]\s*([A-Ea-e])", t)
        if not m:
            m = re.search(r"\b([A-Ea-e])\b", t)
        return m.group(1).upper() if m else t
    if task == "DFI":
        # "争议焦点：X" — drop the prefix, keep value
        m = re.search(r"争议焦点[:：](.*)", t)
        return m.group(1).strip() if m else t
    if task == "LED":
        # "同意/信任" — slash-separated; normalise to comma-sep for f1_micro
        return t.replace("/", "、").replace(";", "、")
    # ITI, OS: return as-is
    return t


def _gold_for(task: str, item: dict) -> str:
    raw = str(item.get("answer") or "")
    return _extract_answer(task, raw)


# ──────────────────────── LawBench backend ────────────────────────
def run_lawbench(ckpt: Path, tasks: list[str], out: Path, limit: int | None):
    lb_root = os.environ.get("LAWBENCH_ROOT")
    if not lb_root or not Path(lb_root).exists():
        raise RuntimeError(
            "请先 `git clone https://github.com/open-compass/LawBench` 并设置 "
            "LAWBENCH_ROOT 环境变量;或者改用 --backend local。"
        )
    cmd = [
        sys.executable, str(Path(lb_root) / "run.py"),
        "--model_name_or_path", str(ckpt),
        "--tasks", ",".join(t.lower() for t in tasks),
        "--output_dir", str(out.parent),
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    print("[lawbench]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=lb_root)


# ──────────────────────── local backend (LawBench) ────────────────
def _load_lawbench_task(task: str, split: str = "zero_shot") -> list[dict]:
    """Load a LawBench task JSON from the local clone."""
    lb_id = TASK_TO_LAWBENCH[task]
    path = LAWBENCH_DIR / "data" / split / f"{lb_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"LawBench file not found: {path}")
    return json.loads(path.read_text("utf-8"))


def _build_runner(ckpt: Path, mode: str):
    """根据模式决定加载哪种 runner:
       collt:           训练后的 COLLT-* SFT,工具真实调用
       baseline_tools:  裸基座 + 工具 system prompt(R3#2 对照)
       baseline_plain:  裸基座,纯零样本
    """
    if mode == "collt":
        return load_runner(str(ckpt))
    if mode == "baseline_tools":
        return BaselineToolRunner(str(ckpt), with_clarify=True)
    if mode == "baseline_plain":
        return BaselineToolRunner(str(ckpt), tool_prompt="", with_clarify=False)
    raise ValueError(f"unknown mode: {mode}")


def run_local(ckpt: Path, tasks: list[str], out: Path,
              limit: int | None, mode: str):
    runner = _build_runner(ckpt, mode)

    report: dict[str, float | None] = {}
    for task in tasks:
        try:
            rows = _load_lawbench_task(task)
        except Exception as e:
            print(f"[skip] {task}: {e}")
            report[task] = None
            continue
        if limit:
            rows = rows[:limit]

        preds, golds = [], []
        for item in rows:
            q = _prompt_for(task, item)
            raw_pred = runner.generate([{"role": "user", "content": q}])
            # strip COLLT action/ER tokens before answer extraction
            for tag in ("<DRT>", "<CLR>", "<ER>"):
                raw_pred = raw_pred.replace(tag, "")
            preds.append(_extract_answer(task, raw_pred))
            golds.append(_gold_for(task, item))

        if task in ("LCP", "LAP", "DFI"):
            report[task] = round(f1_micro(preds, golds), 4)
        elif task == "LED":
            report[task] = round(f1_micro(preds, golds), 4)
        elif task == "OS":
            report[task] = round(rouge_l(preds, golds), 4)
        elif task in ("AM", "ITI", "CA"):
            report[task] = round(accuracy(preds, golds), 4)
        elif task == "PTP":
            try:
                ps = [float(p) for p in preds]
                gs = [float(g) for g in golds]
                report[task] = round(log_distance(ps, gs), 4)
            except ValueError:
                report[task] = None
        print(f"[{task}] {report[task]}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[done] table-3 report → {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",    required=True,
                   help="COLLT-* 微调路径 或 裸基座路径(取决于 --mode)")
    p.add_argument("--backend", choices=["local", "lawbench"], default="local")
    p.add_argument("--mode",    choices=["collt", "baseline_tools", "baseline_plain"],
                   default="collt",
                   help="评估模式:collt = SFT 后 COLLT 模型; "
                        "baseline_tools = 裸基线 + 工具 system prompt (R3#2 公平对照); "
                        "baseline_plain = 裸基线,纯零样本")
    p.add_argument("--tasks",   default=",".join(TABLE3_TASKS))
    p.add_argument("--limit",   type=int, default=None)
    p.add_argument("--out",     default=str(OUT_DIR / "eval_table3.json"))
    p.add_argument("--ablate",  default="",
                   help="R3#7 工具消融:逗号分隔工具名 或 ALL")
    args = p.parse_args()
    if args.ablate:
        os.environ["COLLT_ABLATE"] = args.ablate

    tasks = [t.strip().upper() for t in args.tasks.split(",") if t.strip()]
    if args.backend == "lawbench":
        run_lawbench(Path(args.ckpt), tasks, Path(args.out), args.limit)
    else:
        run_local(Path(args.ckpt), tasks, Path(args.out), args.limit, args.mode)


if __name__ == "__main__":
    main()
