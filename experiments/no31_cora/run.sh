#!/usr/bin/env bash
# Archival NO31+CTR/KD and CoRA. Foreground, serial on GPU0 only.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

EPOCHS="${NO31_EPOCHS:-1}"
if [[ "$EPOCHS" != "1" && "$EPOCHS" != "2" ]]; then
  echo "[NO31-ERROR] NO31_EPOCHS must be 1 or 2" >&2; exit 2
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "minigpt4" ]]; then
  echo "[NO31-ERROR] conda activate minigpt4 first" >&2; exit 2
fi
ACT="${1:-all}"
case "$ACT" in
  prepare|run|all) ;;
  *) echo "Usage: bash experiments/no31_cora/run.sh [prepare|run|all]" >&2; exit 2 ;;
esac

python -m py_compile \
  minigpt4/models/minigpt4rec_v2.py \
  minigpt4/runners/runner_base.py \
  experiments/no31_cora/adapter.py \
  experiments/no31_cora/entry.py \
  experiments/no31_cora/run_pair.py \
  experiments/no31_cora/test_probe_resume.py

python -u experiments/no31_cora/test_cpu.py
python -u experiments/no31_cora/test_probe_resume.py

if [[ "$ACT" == "prepare" || "$ACT" == "all" ]]; then
  python -u experiments/no31_cora/run_pair.py prepare --epochs "$EPOCHS"
  if [[ "$ACT" == "prepare" ]]; then
    echo "[NO31-PREPARED] No GPU training started."
    exit 0
  fi
fi

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | grep -Eq '^[[:space:]]*[0-9]+'; then
  echo "[NO31-ERROR] Another GPU compute process detected; don't overlap training" >&2
  exit 3
fi
python -u experiments/no31_cora/run_pair.py run --epochs "$EPOCHS"
