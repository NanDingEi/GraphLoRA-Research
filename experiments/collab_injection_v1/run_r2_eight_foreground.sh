#!/usr/bin/env bash
# Eight R2 slots serial, foreground GPU0, abort on the first error.
set -euo pipefail
cd /home/bbx/user_lx/GraphLoRA_collab_v1
export CUDA_VISIBLE_DEVICES=0
if [[ "${CONDA_DEFAULT_ENV:-}" != "minigpt4" ]]; then
  echo '[R2-STOP] Activate conda env minigpt4 first' >&2
  exit 2
fi
mkdir -p logs/collab_v1
echo '[R2-PREFLIGHT] code syntax'
python -m py_compile \
  experiments/collab_injection_v1/r2_collab_runtime.py \
  experiments/collab_injection_v1/r2_01_cpu_score_audit.py \
  experiments/collab_injection_v1/r2_probe.py \
  experiments/collab_injection_v1/r2_runtime_entry.py \
  experiments/collab_injection_v1/r2_train.py \
  experiments/collab_injection_v1/test_r2_cpu.py
echo '[R2-PREFLIGHT] CPU tensor and role tests'
python -u experiments/collab_injection_v1/test_r2_cpu.py
echo '[R2-01-START] CPU native Yes-logit audit'
python -u experiments/collab_injection_v1/r2_01_cpu_score_audit.py \
  2>&1 | tee logs/collab_v1/R2_01_SCORE_AUDIT.log
for n in 02 03 04 05; do
  echo "[R2-${n}-START] 512-sample forward probe"
  python -u experiments/collab_injection_v1/r2_probe.py --round "${n}" \
    2>&1 | tee "logs/collab_v1/R2_${n}_PROBE.log"
done
echo '[R2-GATE] probes finished; starting original-S2 matched training'
for n in 06 07 08; do
  echo "[R2-${n}-START] smoke then full 1 epoch"
  python -u experiments/collab_injection_v1/r2_train.py --round "${n}" \
    2>&1 | tee "logs/collab_v1/R2_${n}_TRAIN_DRIVER.log"
done
echo '[R2-EIGHT-COMPLETE] completed sequentially; no Test run'
