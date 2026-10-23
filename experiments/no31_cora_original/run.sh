#!/usr/bin/env bash
# Faithful original CoRA weight-generation and PLoRA integration over archived
# NO31 CTR=.05 / KD=.05. One epoch only, Valid only, GPU0 foreground.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
if [[ "${CONDA_DEFAULT_ENV:-}" != "minigpt4" ]]; then
  echo "[CORA-ORIGINAL-ERROR] conda activate minigpt4 first" >&2; exit 2
fi
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

python -m py_compile \
  experiments/no31_cora_original/qformer.py \
  experiments/no31_cora_original/weight_generator.py \
  experiments/no31_cora_original/adapter.py \
  experiments/no31_cora_original/entry.py \
  experiments/no31_cora_original/run.py \
  experiments/no31_cora_original/test_cpu.py

python -u experiments/no31_cora_original/test_cpu.py

ACTION="${1:-}"
case "$ACTION" in
  prepare|probe|run) ;;
  *) echo "Usage: bash experiments/no31_cora_original/run.sh [prepare|probe|run]" >&2; exit 2 ;;
esac

if [[ "$ACTION" != "prepare" ]]; then
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
       | grep -Eq '^[[:space:]]*[0-9]+'; then
    echo "[CORA-ORIGINAL-ERROR] GPU currently occupied; refusing overlapping work" >&2
    exit 3
  fi
fi
python -u experiments/no31_cora_original/run.py "$ACTION"
