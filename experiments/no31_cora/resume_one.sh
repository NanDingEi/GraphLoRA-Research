#!/usr/bin/env bash
# Continue ONLY the finished epoch0 NO31 CTR/KD CoRA model for epoch1.
# Foreground on GPU0; no previous experiment outputs will be overwritten.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
if [[ "${CONDA_DEFAULT_ENV:-}" != "minigpt4" ]]; then
  echo "[NO31-RESUME-ERROR] conda activate minigpt4 first" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

python -m py_compile \
  minigpt4/models/minigpt4rec_v2.py \
  minigpt4/runners/runner_base.py \
  experiments/no31_cora/adapter.py \
  experiments/no31_cora/entry.py \
  experiments/no31_cora/resume_one.py

if [[ "${1:-}" != "--dry-run" ]]; then
  # The user has only GPU0 for this experiment. Never overlap with another task.
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
       | grep -Eq '^[[:space:]]*[0-9]+'; then
    echo "[NO31-RESUME-ERROR] GPU occupied; refusing overlapping training" >&2
    exit 3
  fi
fi
python -u experiments/no31_cora/resume_one.py "$@"
