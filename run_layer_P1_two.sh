#!/usr/bin/env bash
# Phase 1: GraphLoRA layer-location study. Exactly two SINGLE-layer experiments.
# One epoch each, only Valid AUC/UAUC. No Test or multi-layer budget split.
# Runs must be started from the GraphLoRA server repository:
#   nohup env GPU=0 bash run_layer_P1_two.sh > logs/LAYER_P1_launcher.log 2>&1 < /dev/null &
set -euo pipefail
cd /home/bbx/user_lx/GraphLoRA
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)"
python -m py_compile minigpt4/models/minigpt4rec_v2.py
echo "[LAYER-P1] start=${STAMP} | fresh shared seed42 | Q+V | gnn_scale=0.1 | trainable NGCF | L24 vs L31"
for L in 24 31; do
  CFG="tran_configs/LAYER_P1_L${L}.yaml"
  LOG="logs/LAYER_P1_${STAMP}_L${L}.log"
  OUT="output/LAYER_P1_${STAMP}_L${L}"
  test -f "$CFG"
  echo "[LAYER-P1] START L${L}, 1 epoch full train then Valid"
  python -u train_collm_mf_din.py --cfg-path "$CFG" --options \
    "run.evaluate=false" "run.max_epoch=1" "run.output_dir=${OUT}" \
    2>&1 | tee "$LOG"
  grep -Fq "[LAYER-EXACT] decoder layer ${L}: q_proj and v_proj" "$LOG" || {
    echo "[LAYER-P1] ERROR: layer matcher did not verify L${L}"; exit 1;
  }
  grep -Fq "***auc:" "$LOG" || {
    echo "[LAYER-P1] ERROR: Valid AUC missing for L${L}"; exit 1;
  }
  echo "[LAYER-P1] L${L} Valid:"
  grep -F "***auc:" "$LOG" | tail -n 1 | cut -c 1-145
  echo "[LAYER-P1] DONE L${L}; train_log=${LOG}"
done
echo "[LAYER-P1] Both layers Valid complete (no Test run)."
