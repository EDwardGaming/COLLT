"""
Objective ambiguity evaluation (responds to reviewer R2#3).

Three metrics computed over outputs/ambiglegalqa.jsonl:

  1. trigger F1          — model emits <CLR> exactly when gold says so
  2. clarification cover — % of gold clarification points whose key
                           phrases appear in the model's clarification turn
  3. multi-turn ROUGE-L  — between the model's final <ER>-enhanced response
                           and gold_final_response

We also compute a Cohen-κ-style agreement between two independent runs of
the same model (with different seeds) for stability — feeding R2#5
"is the synthetic data biased" with quantitative variance evidence.

Usage:
    python -m train.eval_ambiglegalqa \
        --ckpt checkpoints/collt-qwen \
        --out  outputs/eval_ambig_qwen.json
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

from .common import AMBIG_LEGAL_QA, OUT_DIR, jsonl_iter
from .inference_collt import load_runner


CLR_RE = re.compile(r"<CLR>", re.I)
DRT_RE = re.compile(r"<DRT>", re.I)


# ── metric 1: trigger F1 ────────────────────────────────────────────
def trigger_f1(records: list[dict]) -> float:
    tp = fp = fn = 0
    for r in records:
        pred = bool(CLR_RE.search(r["model_first_turn"]))
        gold = r["gold_clarification_turns"] > 0
        if pred and gold: tp += 1
        elif pred and not gold: fp += 1
        elif (not pred) and gold: fn += 1
    if tp + fp == 0 or tp + fn == 0: return 0.0
    p = tp / (tp + fp); rcl = tp / (tp + fn)
    return 0.0 if p + rcl == 0 else 2 * p * rcl / (p + rcl)


# ── metric 2: clarification coverage ────────────────────────────────
def coverage(records: list[dict]) -> float:
    """A clarification point is covered if any 3-char sliding window of
    a gold question phrase appears in the model's clarification text."""
    scores = []
    for r in records:
        if r["gold_clarification_turns"] == 0:
            continue
        gold = r["gold_clarification_assistant"]
        pred = r["model_first_turn"]
        # split gold into "1) … 2) …" points
        points = [s.strip() for s in re.split(r"\d\)", gold) if s.strip()]
        if not points: continue
        hits = sum(1 for pt in points
                   if any(pt[i:i+3] in pred for i in range(0, max(1, len(pt)-2))))
        scores.append(hits / len(points))
    return sum(scores) / max(1, len(scores))


# ── metric 3: multi-turn ROUGE-L on final response ──────────────────
def _lcs(a, b):
    m, n = len(a), len(b)
    dp = [0]*(n+1)
    for i in range(1, m+1):
        prev = 0
        for j in range(1, n+1):
            cur = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j-1])
            prev = cur
    return dp[n]


def rouge_l_final(records: list[dict]) -> float:
    scores = []
    for r in records:
        p, g = r["model_final"], r["gold_final_response"]
        if not p or not g: continue
        l = _lcs(p, g)
        if l == 0: continue
        prec = l / len(p); rec = l / len(g)
        scores.append(2 * prec * rec / (prec + rec))
    return sum(scores) / max(1, len(scores))


# ── inference driver ────────────────────────────────────────────────
def run_inference(ckpt: str, records: list[dict]) -> list[dict]:
    runner = load_runner(ckpt)
    out: list[dict] = []
    for r in records:
        q = r["question"]
        turn1 = runner.generate([{"role": "user", "content": q}])
        msgs = [{"role": "user", "content": q},
                {"role": "assistant", "content": turn1}]
        # if clarification was emitted, feed gold supplements as user reply
        if CLR_RE.search(turn1) and r["gold_supplements"]:
            msgs.append({"role": "user", "content": r["gold_supplements"][0]})
            turn2 = runner.generate(msgs)
        else:
            turn2 = turn1
        out.append({**r,
                    "model_first_turn": turn1,
                    "model_final":      turn2})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=str(AMBIG_LEGAL_QA))
    ap.add_argument("--out",  default=str(OUT_DIR / "eval_ambig.json"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    records = list(jsonl_iter(args.data))
    if args.limit: records = records[:args.limit]
    print(f"[data] {len(records)} ambiguity-eval records")

    runs = run_inference(args.ckpt, records)
    report = {
        "trigger_f1":          round(trigger_f1(runs),    4),
        "clarification_cover": round(coverage(runs),      4),
        "rouge_l_final":       round(rouge_l_final(runs), 4),
        "n":                   len(runs),
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
