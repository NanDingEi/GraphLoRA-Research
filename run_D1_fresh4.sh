#!/usr/bin/env bash
# FRESH4: independently initialize LLM LoRA/projector; use only frozen S2 MF/NGCF.
# Runs 4 experiments serially with 2 train epochs each, best-Valid Test.
# Execute as: GPU=0 bash run_D1_fresh4.sh
set -euo pipefail
cd /home/bbx/user_lx/GraphLoRA
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
S2="output/S2_groups_acc3/20261010223/checkpoint_best.pth"
test -f "$S2"
mkdir -p logs out
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="out/FRESH4_${STAMP}_summary.tsv"
printf 'name\tinitial_valid_auc\tbest_checkpoint\ttrain_log\ttest_log\n' > "$SUMMARY"

python -m unittest tests.test_D1_decision_injection -v
EXPERIMENTS=(N0_BASELINE N1_ID_V_ONLY N2_DECISION_Q_ID_V N3_GATED_DECISION_Q_ID_V)
for TAG in "${EXPERIMENTS[@]}"; do
  CFG="tran_configs/FRESH4_${TAG}.yaml"
  TRAIN_FOLDER="output/FRESH4_${STAMP}_${TAG}_train"
  PRE_LOG="logs/FRESH4_${STAMP}_${TAG}_initial_valid.log"
  TRAIN_LOG="logs/FRESH4_${STAMP}_${TAG}_train.log"
  TEST_LOG="logs/FRESH4_${STAMP}_${TAG}_test.log"
  echo "[FRESH4] START $TAG from fresh LoRA/projector (S2 MF/NGCF frozen)"
  test -f "$CFG"
  # This is a genuine fresh-LLM initial validation, NOT an S2 equivalence check.
  python -u train_collm_mf_din.py --cfg-path "$CFG" --options \
    "run.evaluate=true" "run.test_splits=[valid]" \
    "run.output_dir=output/FRESH4_${STAMP}_${TAG}_initial" 2>&1 | tee "$PRE_LOG"

  INIT_AUC="$(python - "$PRE_LOG" <<'PY'
import sys,re
from pathlib import Path
log=Path(sys.argv[1]).read_text(errors="replace")
assert "[D1-FRESH] restored small-model state tensors" in log, \
    "Initialization was not fresh: missing D1-FRESH log"
v=re.findall(r"rank_0 auc:\s*([0-9]+(?:\.[0-9]+)?)",log)
assert v, "No initial Valid AUC recorded"
print(v[-1])
PY
)"
  echo "[FRESH4] $TAG initial Valid AUC: $INIT_AUC"

  python -u train_collm_mf_din.py --cfg-path "$CFG" --options \
    "run.evaluate=false" "run.max_epoch=2" \
    "run.output_dir=${TRAIN_FOLDER}" 2>&1 | tee "$TRAIN_LOG"

  # Runner's output is under minigpt4/output even if run.output_dir is relative.
  BEST="$(find "minigpt4/${TRAIN_FOLDER}" -type f -name checkpoint_best.pth -print -quit)"
  test -n "$BEST"
  COMPLETE="${BEST%.pth}_complete.pth"
  python -u complete_D1_checkpoint.py --s2 "$S2" --d1 "$BEST" --out "$COMPLETE"

  # No pointless 7331-sample Test if decision-Q projector remained all-zero:
  if [[ "$TAG" == N2_* || "$TAG" == N3_* ]]; then
    python - "$COMPLETE" <<'PY'
import sys,torch
ckpt=torch.load(sys.argv[1],map_location="cpu",weights_only=False)
p=ckpt["model"].get("d1_pair_proj.weight")
if p is None:
    raise RuntimeError("D1 pair projector MISSING in saved checkpoint")
norm=p.detach().float().norm().item()
print("[FRESH4] trained decision Q projection L2 = %.12g" % norm,flush=True)
if norm <= 1e-8:
    raise RuntimeError("D1 decision-Q projection still zero; stop before Test")
PY
  fi

  python -u train_collm_mf_din.py --cfg-path "$CFG" --options \
    "model.ckpt=$(realpath "$COMPLETE")" \
    "run.evaluate=true" "run.test_splits=[test]" \
    "run.output_dir=output/FRESH4_${STAMP}_${TAG}_test" 2>&1 | tee "$TEST_LOG"

  printf '%s\t%s\t%s\t%s\t%s\n' "$TAG" "$INIT_AUC" "$COMPLETE" \
    "$TRAIN_LOG" "$TEST_LOG" >> "$SUMMARY"
  echo "[FRESH4] FINISHED $TAG"
done
echo "[FRESH4] All four experiments complete. $SUMMARY"
