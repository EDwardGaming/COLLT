"""
Shared constants and lightweight IO for COLLT training / inference.

This module is intentionally framework-free (no torch / transformers / unsloth
imports) so it can be imported by both training scripts and the streaming
inference runtime.

Special-token spec follows manuscript Table 1 exactly:

  Action tokens (open-only, no closing pair):
    <CLR>  clarification action
    <DRT>  direct-response action

  Legal tool tokens (paired head/tail):
    <SCR></SCR>   Similar Case Retrieval
    <LAS></LAS>   Legal Article Search
    <LCP></LCP>   Legal Charge Prediction
    <LER></LER>   Legal Element Recognition
    <LED></LED>   Legal Event Detection
    <NET></NET>   Internet Search

  Enhanced-response delimiter (open-only):
    <ER>
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Iterator

import orjson

# ── repo paths ───────────────────────────────────────────────────────────
# After the folder split, train/ lives as a sibling of bulidData/ under
# E:\...\buildData\.  All paths are anchored to TRAIN_DIR so the package
# is self-contained regardless of where it is cloned or run from.
TRAIN_DIR = Path(__file__).resolve().parent          # …/buildData/train
OUT_DIR   = TRAIN_DIR / "outputs"                    # …/buildData/train/outputs
CKPT_DIR  = TRAIN_DIR / "checkpoints"                # …/buildData/train/checkpoints
# Data-build pipeline root (bulidData/) — only needed if you want to re-run
# run_all.py from here; not required for training.
DATA_BUILD_DIR = TRAIN_DIR.parent / "bulidData"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# COLLT dataset directory (checked-in copies; preferred over built outputs/)
DATASET_DIR = TRAIN_DIR / "DATASET" / "COLLT_DATASET"


def _prefer_dataset(filename: str) -> Path:
    """Return DATASET_DIR/<filename> if it exists, else OUT_DIR/<filename>."""
    candidate = DATASET_DIR / filename
    return candidate if candidate.exists() else OUT_DIR / filename


# generated datasets (from run_all.py) — prefer checked-in DATASET/ copy
COLLT_SFT        = _prefer_dataset("collt_sft.jsonl")
TLAS_TRAIN       = _prefer_dataset("tlas_train.jsonl")
TLCP_TRAIN       = _prefer_dataset("tlcp_train.jsonl")
AMBIG_LEGAL_QA   = _prefer_dataset("ambiglegalqa.jsonl")
TLER_TRAIN       = _prefer_dataset("tler_train.jsonl")
TLER_VALID       = _prefer_dataset("tler_valid.jsonl")
TLER_LABEL_NAMES = _prefer_dataset("tler_label_names.json")

# ── special tokens ────────────────────────────────────────────────────────
ACTION_TOKENS = ["<CLR>", "<DRT>"]
TOOL_NAMES    = ["SCR", "LAS", "LCP", "LER", "LED", "NET"]
TOOL_HEAD_TAGS = [f"<{n}>"  for n in TOOL_NAMES]
TOOL_TAIL_TAGS = [f"</{n}>" for n in TOOL_NAMES]
ER_TOKEN      = "<ER>"

# whole set we add to the tokenizer so SFT loss never splits them
SPECIAL_TOKENS: list[str] = (
    ACTION_TOKENS + TOOL_HEAD_TAGS + TOOL_TAIL_TAGS + [ER_TOKEN]
)

# tool budget — Proposition 1
TOOL_BUDGET = 2

# ── small JSONL helpers (no orjson option mismatch with utils.py) ─────────
def jsonl_iter(path: str | Path) -> Iterator[dict]:
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                yield orjson.loads(line)


def jsonl_write(records: Iterable[dict], path: str | Path) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "wb") as f:
        for r in records:
            f.write(orjson.dumps(r, option=orjson.OPT_APPEND_NEWLINE))
            n += 1
    return n


# ── base-model registry for COLLT-* series (Table 3) ──────────────────────
# All entries are HuggingFace repo ids. The user may override via env.
BASE_MODELS: dict[str, str] = {
    "glm":      "THUDM/chatglm3-6b",
    "llama":    "meta-llama/Meta-Llama-3-8B-Instruct",
    "internlm": "internlm/internlm3-8b-instruct",
    "qwen":     "Qwen/Qwen2.5-7B-Instruct",
    "baichuan": "baichuan-inc/Baichuan2-7B-Chat",
}


def base_model_id(short: str) -> str:
    """解析基座模型路径的优先级:
       1) 环境变量 COLLT_BASE_<NAME>       (硬覆盖)
       2) $COLLT_LOCAL_MODELS_DIR/<repo_basename> 如果该目录存在
       3) HuggingFace repo id (BASE_MODELS[short])
    """
    import os
    env_key = f"COLLT_BASE_{short.upper()}"
    if os.environ.get(env_key):
        return os.environ[env_key]
    hf_id = BASE_MODELS[short]
    local_root = os.environ.get("COLLT_LOCAL_MODELS_DIR")
    if local_root:
        candidate = Path(local_root) / hf_id.split("/")[-1]
        if candidate.exists():
            return str(candidate)
    return hf_id


def local_model_path(hf_id: str) -> str:
    """通用本地权重解析:对任意 HF repo id,优先返回本地路径(若存在),
    否则原样返回 hf_id 让 transformers 走远程下载。"""
    import os
    local_root = os.environ.get("COLLT_LOCAL_MODELS_DIR")
    if local_root:
        candidate = Path(local_root) / hf_id.split("/")[-1]
        if candidate.exists():
            return str(candidate)
    return hf_id


def local_dataset_path(hf_id: str) -> str:
    """通用本地数据集解析:与 local_model_path 同语义,根目录为
    $COLLT_LOCAL_DATA_DIR。若本地不存在,返回原 hf_id 由 datasets
    走远程加载。"""
    import os
    local_root = os.environ.get("COLLT_LOCAL_DATA_DIR")
    if local_root:
        candidate = Path(local_root) / hf_id.split("/")[-1]
        if candidate.exists():
            return str(candidate)
    return hf_id


# ── LawBench evaluation data ──────────────────────────────────────────────────
# Resolved at import time; override with COLLT_LOCAL_DATA_DIR if needed.
def _lawbench_dir() -> Path:
    import os
    data_root = os.environ.get("COLLT_LOCAL_DATA_DIR")
    if data_root:
        candidate = Path(data_root) / "LawBench"
        if candidate.exists():
            return candidate
    # default: repo-root/local_data/LawBench (sibling of train/)
    candidate = TRAIN_DIR.parent / "local_data" / "LawBench"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        "LawBench not found. Clone https://github.com/open-compass/LawBench "
        "into local_data/LawBench or set COLLT_LOCAL_DATA_DIR."
    )


try:
    LAWBENCH_DIR: Path = _lawbench_dir()
except FileNotFoundError:
    LAWBENCH_DIR = TRAIN_DIR.parent / "local_data" / "LawBench"  # placeholder

# ── Tool-training dataset paths ───────────────────────────────────────────────
# CAIL2019 相似案例匹配 — T_SCR training (A/B/C pairwise format)
CAIL2019_SCM_DIR: Path = TRAIN_DIR.parent / "local_data" / "CAIL2019" / "相似案例匹配"

# LEVEN — T_LED training (document-level span annotation, native jsonl)
LEVEN_DIR: Path = TRAIN_DIR.parent / "local_data" / "LEVEN"
