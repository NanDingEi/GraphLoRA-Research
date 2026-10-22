#!/usr/bin/env bash
# Five DISTINCT models with seed=42: TWO complete epochs each; S2 cosine horizon remains 3.
# GPU0 only, fully serial, FOREGROUND. No nohup, tmux or background execution.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
EXP="experiments/s2_cora"
LOG="logs/S2_CORA_FIVE_20261022"
ACTION="${1:-all}"
case "$ACTION" in prepare|run|all|summary) ;; *) echo "Usage: $0 {prepare|run|all|summary}" >&2; exit 2;; esac

if [[ "$ACTION" == "prepare" || "$ACTION" == "all" ]]; then
  python -m py_compile \
    minigpt4/models/minigpt4rec_v2.py \
    minigpt4/runners/runner_base.py \
    "$EXP/s2_cora_adapter.py" \
    "$EXP/s2_entry.py" \
    "$EXP/night5.py"
  python -u "$EXP/test_s2_cora_cpu.py"
  if [[ ! -e "$LOG" ]]; then
    python -u "$EXP/night5.py" prepare
  else
    echo "[FIVE-INFO] Prepared run directory already exists; refusing to overwrite configs" >&2
  fi
  if [[ "$ACTION" == "prepare" ]]; then
    echo "[FIVE-PREPARE-ONLY] No GPU training started"
    exit 0
  fi
fi

if [[ "$ACTION" == "summary" ]]; then
  python "$EXP/night5.py" summary
  cat "$LOG/summary.csv"
  exit 0
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "minigpt4" ]]; then
  echo "[FIVE-STOP] Please activate conda minigpt4" >&2
  exit 2
fi

# Do not launch if existing server training is consuming GPU0.
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
     | grep -Eq '^[[:space:]]*[0-9]+'; then
  echo "[FIVE-STOP] GPU has active compute processes. Stop previous training before launching the queue." >&2
  exit 3
fi

echo "[FIVE-LAUNCH] 5 total runs (S2 + 4 models) × 2 full epochs, seed=42, single GPU0, Valid only; $(date -Is)"
python -u "$EXP/night5.py" run 2>&1 | tee -a "$LOG/master.log"
echo "[FIVE-RESULTS]"
cat "$LOG/summary.csv"
