"""
一站式下载脚本(Linux 优先)。

落地两个目录:
  · $COLLT_LOCAL_DATA_DIR(默认 ./local_data)
      DISC-Law-SFT 原始 jsonl (T_LAS / T_LCP / COLLT-SFT 训练数据来源)

      注意:
      · Table 3 评测数据统一使用本地 LawBench(local_data/LawBench)。
        请先 `git clone https://github.com/open-compass/LawBench local_data/LawBench`
      · T_SCR 训练数据已就绪: local_data/CAIL2019/相似案例匹配/
      · T_LED 训练数据已就绪: local_data/LEVEN/
      · T_LER 暂无 BIO 语料(训练时自动跳过)

  · $COLLT_LOCAL_MODELS_DIR(默认 ./local_models)
      Lawformer 编码器
      5 个 COLLT 基座模型权重(若无 token 或带 gated repo,会跳过并打印提示)

之后所有训练 / 推理脚本都会优先在这两个目录查文件,缺失时自动回退到
HuggingFace 远端,不需要手动改代码。

用法:
    bash download_all.sh                       # 推荐:走 shell 入口,带日志
    python -m train.download_all               # 等价于全量下载
    python -m train.download_all --skip models # 只下数据集
    python -m train.download_all --skip data   # 只下模型权重
    python -m train.download_all --only Lawformer Qwen2.5-7B-Instruct
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path


# ────────────────────────── 资源清单 ─────────────────────────────
DATASETS: list[tuple[str, str, str]] = [
    # (HF repo id, 子集名(可空), 备注)
    # 只剩 DISC-Law-SFT 需要从 HuggingFace 下载。
    # 其他数据集已通过 git clone 等方式存放在 local_data/ 下:
    #   T_SCR  → local_data/CAIL2019/相似案例匹配  ({"A","B","C","label"} jsonl)
    #   T_LER  → 暂无可用训练数据(CAIL-LeRec 不可达,训练自动跳过)
    #   T_LED  → 暂无可用训练数据(LEVEN 语料需手动下载,训练自动跳过)
    #   评测   → local_data/LawBench  (git clone open-compass/LawBench)
    ("ShengbinYue/DISC-Law-SFT", "", "DISC-Law-SFT 主数据(必备,T_LAS/T_LCP/COLLT-SFT)"),
]

MODELS: list[tuple[str, str]] = [
    ("thunlp/Lawformer",                     "工具主干(必备)"),
    ("THUDM/chatglm3-6b",                    "COLLT-GLM"),
    ("meta-llama/Meta-Llama-3-8B-Instruct",  "COLLT-LLaMa(gated,需要 HF token)"),
    ("internlm/internlm3-8b-instruct",       "COLLT-InternLM"),
    ("Qwen/Qwen2.5-7B-Instruct",             "COLLT-Qwen"),
    ("baichuan-inc/Baichuan2-7B-Chat",       "COLLT-Baichuan"),
]


def _ensure_dirs() -> tuple[Path, Path]:
    data_root  = Path(os.environ.get("COLLT_LOCAL_DATA_DIR",   "./local_data")).resolve()
    model_root = Path(os.environ.get("COLLT_LOCAL_MODELS_DIR", "./local_models")).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    return data_root, model_root


def _snapshot(repo_id: str, target: Path, repo_type: str) -> bool:
    """huggingface_hub.snapshot_download 包装。"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[error] 请先 `pip install huggingface_hub`", file=sys.stderr)
        sys.exit(2)
    try:
        snapshot_download(
            repo_id   = repo_id,
            repo_type = repo_type,
            local_dir = str(target),
            tqdm_class = None,
        )
        return True
    except Exception as e:
        print(f"[warn] 跳过 {repo_id}: {e}")
        return False


def _name(repo_id: str) -> str:
    return repo_id.split("/")[-1]


def download_datasets(data_root: Path, only: set[str] | None) -> None:
    print(f"\n[step] 数据集 → {data_root}")
    for repo, _sub, note in DATASETS:
        if only and _name(repo) not in only and repo not in only:
            continue
        target = data_root / _name(repo)
        if target.exists() and any(target.iterdir()):
            print(f"[skip] {repo} 已存在: {target}")
            continue
        print(f"[get ] {repo}  ({note})")
        _snapshot(repo, target, "dataset")


def download_models(model_root: Path, only: set[str] | None) -> None:
    print(f"\n[step] 模型权重 → {model_root}")
    for repo, note in MODELS:
        if only and _name(repo) not in only and repo not in only:
            continue
        target = model_root / _name(repo)
        if target.exists() and any(target.iterdir()):
            print(f"[skip] {repo} 已存在: {target}")
            continue
        print(f"[get ] {repo}  ({note})")
        _snapshot(repo, target, "model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", choices=["data", "models", "none"], default="none",
                    help="跳过某一类(data 只下模型;models 只下数据)")
    ap.add_argument("--only", nargs="*",
                    help="只下载指定的资源(短名或全名),例如 Lawformer Qwen2.5-7B-Instruct")
    args = ap.parse_args()

    only = set(args.only) if args.only else None
    data_root, model_root = _ensure_dirs()
    print(f"[env] COLLT_LOCAL_DATA_DIR   = {data_root}")
    print(f"[env] COLLT_LOCAL_MODELS_DIR = {model_root}")

    if args.skip != "data":
        download_datasets(data_root, only)
    if args.skip != "models":
        download_models(model_root, only)

    print("\n[done] 后续训练 / 推理脚本会自动优先使用这两个目录。")
    print("       建议把以下两行加入 ~/.bashrc:")
    print(f'         export COLLT_LOCAL_DATA_DIR="{data_root}"')
    print(f'         export COLLT_LOCAL_MODELS_DIR="{model_root}"')


if __name__ == "__main__":
    main()