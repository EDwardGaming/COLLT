#!/usr/bin/env bash
# 数据集 + 模型权重一站式下载脚本(Linux/WSL 推荐)
#
# 使用前:
#   1) 安装依赖
#        pip install -r requirements_train.txt
#        pip install -U huggingface_hub
#   2) 如果需要下载 gated 仓库(LLaMa-3、Qwen2.5 等):
#        huggingface-cli login
#
# 用法:
#   bash download_all.sh                  # 全量下载
#   bash download_all.sh --skip models    # 只下数据
#   bash download_all.sh --skip data      # 只下模型
#   bash download_all.sh --only Lawformer Qwen2.5-7B-Instruct
#
# 产物默认落在 ./local_data 和 ./local_models;可通过环境变量覆盖。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/train/outputs/_download_logs"
mkdir -p "$LOG_DIR"

export COLLT_LOCAL_DATA_DIR="${COLLT_LOCAL_DATA_DIR:-$ROOT/local_data}"
export COLLT_LOCAL_MODELS_DIR="${COLLT_LOCAL_MODELS_DIR:-$ROOT/local_models}"

# huggingface_hub 性能/容错调优
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DOWNLOAD_TIMEOUT=120

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/download_${STAMP}.log"

echo "[run] python -m train.download_all $* (log: $LOG_FILE)"
python -m train.download_all "$@" 2>&1 | tee "$LOG_FILE"

echo
echo "[hint] 把下面两行写入 ~/.bashrc / ~/.zshrc,后续训练脚本即自动用本地路径:"
echo "       export COLLT_LOCAL_DATA_DIR=\"$COLLT_LOCAL_DATA_DIR\""
echo "       export COLLT_LOCAL_MODELS_DIR=\"$COLLT_LOCAL_MODELS_DIR\""
