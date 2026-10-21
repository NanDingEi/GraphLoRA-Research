#!/usr/bin/env bash
# R2 eight slots in strict order, GPU0 foreground only, safe continuation.
set -euo pipefail
cd /home/bbx/user_lx/GraphLoRA_collab_v1
export CUDA_VISIBLE_DEVICES=0
if [[ "${CONDA_DEFAULT_ENV:-}" != "minigpt4" ]]; then
  echo '[R2-STOP] Activate conda env minigpt4 first' >&2
  exit 2
fi
mkdir -p logs/collab_v1
echo '[R2-PREFLIGHT] syntax + CPU roles + training source'
python -m py_compile \
  experiments/collab_injection_v1/r2_collab_runtime.py \
  experiments/collab_injection_v1/r2_01_cpu_score_audit.py \
  experiments/collab_injection_v1/r2_probe.py \
  experiments/collab_injection_v1/r2_runtime_entry.py \
  experiments/collab_injection_v1/r2_train.py \
  experiments/collab_injection_v1/r2_verify_resume.py \
  experiments/collab_injection_v1/test_r2_cpu.py
python -u experiments/collab_injection_v1/test_r2_cpu.py
python -u experiments/collab_injection_v1/r2_train.py \
  --round 06 --check-source-only "${@}"

# Exit 0 means completed archive, exit 3 means safe to run,
# anything else is a hard stop. A failed/partial training run is
# never deleted or silently restarted.
resume_status() {
  local rc
  set +e
  python -u experiments/collab_injection_v1/r2_verify_resume.py --round "$1"
  rc=$?
  set -e
  if [[ "$rc" -eq 0 ]]; then
    return 0
  elif [[ "$rc" -eq 3 ]]; then
    return 1
  else
    echo "[R2-STOP] Unsafe or corrupted status for round $1" >&2
    exit 2
  fi
}

if resume_status 01; then
  :
else
  echo '[R2-01-START] CPU native Yes-logit audit'
  python -u experiments/collab_injection_v1/r2_01_cpu_score_audit.py \
    2>&1 | tee logs/collab_v1/R2_01_SCORE_AUDIT.log
fi

for n in 02 03 04 05; do
  if resume_status "$n"; then
    continue
  fi
  echo "[R2-${n}-START] untrained 512-sample Valid probe"
  python -u experiments/collab_injection_v1/r2_probe.py --round "$n" \
    2>&1 | tee "logs/collab_v1/R2_${n}_PROBE.log"
done

for n in 06 07 08; do
  if resume_status "$n"; then
    continue
  fi
  echo "[R2-${n}-START] isolated 24-step smoke then one full epoch"
  python -u experiments/collab_injection_v1/r2_train.py --round "$n" "${@}" \
    2>&1 | tee "logs/collab_v1/R2_${n}_TRAIN_DRIVER.log"
done
echo '[R2-EIGHT-COMPLETE] verified 8 slots; Test set untouched'
