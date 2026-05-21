"""
One-shot orchestrator — trains every legal tool, then trains every
COLLT-* causal-LM, then runs the three Table-3 / ambiguity / ablation
evaluation pipelines.

Stages are idempotent: re-running re-uses checkpoints if they already exist.
Pass --only to limit to one stage during debugging.

Tool training (manuscript §3.2.2 / §4.1):
    T_LAS  fine-tuned Lawformer, data: outputs/tlas_train.jsonl
    T_LCP  fine-tuned Lawformer, data: outputs/tlcp_train.jsonl
    T_SCR  fine-tuned Lawformer, data: local_data/CAIL2019/相似案例匹配/
    T_LER  fine-tuned Lawformer (requires --data_jsonl; skips if absent)
    T_LED  fine-tuned Lawformer, data: local_data/LEVEN/
    T_NET  Bing Search API — no training

Stages:
    tools     T_LAS, T_LCP, T_SCR, T_LER, T_LED
    collt     COLLT-Qwen / -LLaMa / -GLM / -InternLM / -Baichuan
    eval      Table-3 + AmbigLegalQA + R3#7 + R3#8

Smoke test (everything on a tiny subset):
    python -m train.train_all --only tools --limit 200
    python -m train.train_all --only collt --models qwen --limit 200
    python -m train.train_all --only eval  --models qwen --limit 50
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

from .common import CKPT_DIR, BASE_MODELS, base_model_id


def _sh(cmd: list[str], **kw) -> None:
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)


def stage_tools(limit: int | None):
    # T_LER 训练前先把 CAIL2019 元素识别数据下载并平铺到 outputs/tler_*.jsonl。
    # 该步骤幂等(本地已存在则跳过下载),失败不阻塞其它工具训练。
    try:
        _sh([sys.executable, "-m", "train.tools.build_ler_data"])
    except subprocess.CalledProcessError as e:
        print(f"[warn] build_ler_data failed ({e}); T_LER will fall back to skipping")

    for tool, mod in [
        ("T_LAS", "train.tools.train_tlas"),
        ("T_LCP", "train.tools.train_tlcp"),
        ("T_SCR", "train.tools.train_tscr"),
        ("T_LER", "train.tools.train_tler"),
        ("T_LED", "train.tools.train_tled"),
    ]:
        cmd = [sys.executable, "-m", mod]
        if limit:
            cmd += ["--limit", str(limit)]
        try:
            _sh(cmd)
        except subprocess.CalledProcessError as e:
            print(f"[warn] {tool} training failed ({e}); continuing")


def stage_collt(models: list[str], limit: int | None, epochs: int):
    for m in models:
        cmd = [sys.executable, "-m", "train.train_collt_sft",
               "--model", m, "--epochs", str(epochs)]
        if limit:
            cmd += ["--limit", str(limit)]
        _sh(cmd)


def stage_eval(models: list[str], limit: int | None):
    for m in models:
        ckpt = CKPT_DIR / f"collt-{m}"
        if not ckpt.exists():
            print(f"[skip] {ckpt} missing")
            continue
        base_args = ["--ckpt", str(ckpt)]
        if limit: base_args += ["--limit", str(limit)]

        _sh([sys.executable, "-m", "train.eval_table3",
             *base_args, "--out", str(ckpt / "eval_table3.json")])
        _sh([sys.executable, "-m", "train.eval_ambiglegalqa",
             *base_args, "--out", str(ckpt / "eval_ambig.json")])
        _sh([sys.executable, "-m", "train.eval_ablation_tools",
             *base_args, "--out", str(ckpt / "eval_ablation_tools.json")])
        _sh([sys.executable, "-m", "train.eval_ablation_clarify",
             "--collt_ckpt", str(ckpt),
             "--base_ckpt",  base_model_id(m),  # 本地优先,缺失回退 HF
             "--out",        str(ckpt / "eval_ablation_clarify.json"),
             *(["--limit", str(limit)] if limit else [])])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["tools", "collt", "eval", "all"], default="all")
    p.add_argument("--models", default="qwen,llama,glm,internlm,baichuan",
                   help="Comma-separated subset of {glm,llama,internlm,qwen,baichuan}")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--limit", type=int, default=None,
                   help="Truncate every dataset / step for smoke testing")
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.only in ("tools", "all"):  stage_tools(args.limit)
    if args.only in ("collt", "all"):  stage_collt(models, args.limit, args.epochs)
    if args.only in ("eval", "all"):   stage_eval(models, args.limit)


if __name__ == "__main__":
    main()