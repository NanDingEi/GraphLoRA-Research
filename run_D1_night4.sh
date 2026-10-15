#!/usr/bin/env bash
# Night4: four S2-start LLM-side injection comparisons, 2 epochs each.
# Run as a CHILD shell: GPU=0 bash run_D1_night4.sh
# Never source this script into an interactive terminal.
set -euo pipefail

cd /home/bbx/user_lx/GraphLoRA
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

S2="output/S2_groups_acc3/20261010223/checkpoint_best.pth"
test -f "$S2"
test -f "minigpt4/models/d1_decision_injection.py"
test -f "complete_D1_checkpoint.py"

mkdir -p logs out
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="out/NIGHT4_${STAMP}_summary.tsv"
printf 'name\tpreflight_auc\tbest_checkpoint\ttrain_log\ttest_log\n' > "$SUMMARY"

python -m unittest tests.test_D1_decision_injection -v

# Run synchronously: one experiment finishes its Test before the next trains.
EXPERIMENTS=(
  "N0_S2_CONTINUE"
  "N1_ID_V_ONLY"
  "N2_DECISION_Q_ID_V"
  "N3_GATED_DECISION_Q_ID_V"
)

for TAG in "${EXPERIMENTS[@]}"; do
  CFG="tran_configs/NIGHT4_${TAG}.yaml"
  TRAIN_FOLDER="output/NIGHT4_${STAMP}_${TAG}_train"
  TRAIN_DISK="minigpt4/${TRAIN_FOLDER}"
  PRE_LOG="logs/NIGHT4_${STAMP}_${TAG}_preflight.log"
  TRAIN_LOG="logs/NIGHT4_${STAMP}_${TAG}_train.log"
  TEST_LOG="logs/NIGHT4_${STAMP}_${TAG}_test.log"

  test -f "$CFG"
  echo "=========================================================="
  echo "[NIGHT4] ${TAG}: S2 champion Valid preflight"
  echo "[NIGHT4] frozen-small | lr=2e-5 | epochs=2 | GPU=$CUDA_VISIBLE_DEVICES"

  python -u train_collm_mf_din.py --cfg-path "$CFG" --options \
    "run.evaluate=true" "run.test_splits=[valid]" \
    "run.output_dir=output/NIGHT4_${STAMP}_${TAG}_preflight" \
    2>&1 | tee "$PRE_LOG"

  PRE_AUC="$(python - "$PRE_LOG" <<'PY'
import re,sys
from pathlib import Path
txt=Path(sys.argv[1]).read_text(encoding='utf-8',errors='replace')
m=re.findall(r'rank_0 auc:\s*([0-9]+(?:\.[0-9]+)?)',txt)
if not m:
    raise RuntimeError("Night4 preflight emitted no valid AUC")
a=float(m[-1])
if abs(a-0.7328835632321707)>0.003:
    raise RuntimeError("Night4 baseline preflight not equivalent to S2: "+str(a))
print(f"{a:.9f}")
PY
)"
  echo "[NIGHT4] preflight auc: ${PRE_AUC}"

  echo "[NIGHT4] ${TAG}: 2-epoch train with per-epoch Valid"
  python -u train_collm_mf_din.py --cfg-path "$CFG" --options \
    "run.evaluate=false" "run.max_epoch=2" \
    "run.output_dir=${TRAIN_FOLDER}" \
    2>&1 | tee "$TRAIN_LOG"

  # Runner creates minigpt4/<output_dir>/<job_id>/checkpoint_best.pth.
  BEST="$(find "$TRAIN_DISK" -type f -name checkpoint_best.pth -print -quit)"
  test -n "$BEST"
  COMPLETE="${BEST%.pth}_complete.pth"
  python -u complete_D1_checkpoint.py \
    --s2 "$S2" --d1 "$BEST" --out "$COMPLETE"

  echo "[NIGHT4] ${TAG}: Test best-Valid checkpoint"
  python -u train_collm_mf_din.py --cfg-path "$CFG" --options \
    "model.ckpt=$(realpath "$COMPLETE")" \
    "run.evaluate=true" "run.test_splits=[test]" \
    "run.output_dir=output/NIGHT4_${STAMP}_${TAG}_test" \
    2>&1 | tee "$TEST_LOG"

  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$TAG" "$PRE_AUC" "$COMPLETE" "$TRAIN_LOG" "$TEST_LOG" >> "$SUMMARY"
  echo "[NIGHT4] ${TAG} finished. Results appended to ${SUMMARY}"
done

echo "=========================================================="
echo "[NIGHT4] Finished all 4 conditions"
echo "[NIGHT4] Summary: ${SUMMARY}"
python - "$SUMMARY" <<'PY'
import re,sys
from pathlib import Path
rows=Path(sys.argv[1]).read_text().splitlines()[1:]
for row in rows:
    tag,pre,ckpt,trainlog,testlog=row.split('\t')
    t=Path(trainlog).read_text(encoding='utf-8',errors='replace')
    v=re.findall(r'\*\*\*auc:\s*([0-9.]+).*?\*\*\*uauc:\(([^,]+)',t)
    q=Path(testlog).read_text(encoding='utf-8',errors='replace')
    test=re.findall(r'\*\*\*auc:\s*([0-9.]+).*?\*\*\*uauc:\(([^,]+)',q)
    print(tag, "VALID epochs=",v, "TEST=",test[-1] if test else None)
PY
