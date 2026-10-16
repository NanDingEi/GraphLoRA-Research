#!/usr/bin/env bash
# R2: 2-arm controlled, independent *fresh LoRA/projector* runs.
# User's pretrained S2 MF/NGCF only, frozen throughout.
# Launch: nohup env GPU=0 bash run_D1_R2_two.sh > logs/R2_launcher.log 2>&1 < /dev/null &
# This script is for a CHILD shell: do not source it.
set -euo pipefail
cd /home/bbx/user_lx/GraphLoRA
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
S2="output/S2_groups_acc3/20261010223/checkpoint_best.pth"
test -f "$S2"
mkdir -p logs out
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="out/R2_${STAMP}_summary.tsv"
printf 'arm\tcheckpoint\ttrain_log\ttest_log\n' > "$SUMMARY"
python -m py_compile minigpt4/models/minigpt4rec_v2.py
python -m unittest tests.test_D1_decision_injection -v

ARMS=(C0_ID_V_ONLY C1_DECISION_Q30)
PREV_BEST=""
for ARM in "${ARMS[@]}"; do
  CFG="tran_configs/R2_${ARM}.yaml"
  TRAIN_REL="output/R2_${STAMP}_${ARM}_train"
  TRAIN_LOG="logs/R2_${STAMP}_${ARM}_train.log"
  TEST_LOG="logs/R2_${STAMP}_${ARM}_test.log"
  test -f "$CFG"
  echo "[R2] START ${ARM} | fresh LoRA/projector | frozen MF+NGCF | 2 epochs | bs=2 accum=4"
  python -u train_collm_mf_din.py --cfg-path "$CFG" --options \
    "run.evaluate=false" "run.max_epoch=2" \
    "run.output_dir=${TRAIN_REL}" 2>&1 | tee "$TRAIN_LOG"

  TRAIN_DISK="minigpt4/${TRAIN_REL}"
  BEST="$(find "$TRAIN_DISK" -type f -name checkpoint_best.pth -print -quit)"
  test -n "$BEST"
  # Before final testing, ensure the new Q path changed more than a label.
  # The first arm's checkpoint is held for a fair, same-initialization comparison.
  if [[ "$ARM" == C1_DECISION_Q30 ]]; then
    python - "$PREV_BEST" "$BEST" <<'PY'
import sys,torch
a=torch.load(sys.argv[1],map_location="cpu",weights_only=False)["model"]
b=torch.load(sys.argv[2],map_location="cpu",weights_only=False)["model"]
q=b.get("d1_pair_proj.weight")
if q is None or not torch.isfinite(q).all():
    raise RuntimeError("[R2] trained decision-Q parameter MISSING or non-finite")
common=[k for k in a if k in b and ("lora_" in k or k.startswith("llama_proj."))]
diff=[k for k in common if not torch.equal(a[k],b[k])]
print("[R2-AUDIT] Q projection L2:",float(q.float().norm()),flush=True)
print("[R2-AUDIT] common LoRA/projector tensors different:",len(diff),"/",len(common),flush=True)
if not common:
    raise RuntimeError("[R2] No comparable trainable LLM tensors found in checkpoints")
if not diff:
    raise RuntimeError("[R2] LLM tensors unexpectedly identical across arms; inspect training before Test")
PY
  fi
  COMPLETE="${BEST%.pth}_complete.pth"
  python -u complete_D1_checkpoint.py \
    --s2 "$S2" --d1 "$BEST" --out "$COMPLETE"

  echo "[R2] Test one best-Valid checkpoint for ${ARM}"
  python -u train_collm_mf_din.py --cfg-path "$CFG" --options \
    "model.ckpt=$(realpath "$COMPLETE")" \
    "run.evaluate=true" "run.test_splits=[test]" \
    "run.output_dir=output/R2_${STAMP}_${ARM}_test" \
    2>&1 | tee "$TEST_LOG"

  printf '%s\t%s\t%s\t%s\n' \
    "$ARM" "$COMPLETE" "$TRAIN_LOG" "$TEST_LOG" >> "$SUMMARY"
  PREV_BEST="$BEST"
  echo "[R2] FINISHED ${ARM}"
done

echo "[R2] All arms finished. Summary: $SUMMARY"
python - "$SUMMARY" <<'PY'
import re,sys
from pathlib import Path
for line in Path(sys.argv[1]).read_text().splitlines()[1:]:
    arm,ckpt,tr,tst=line.split('\t')
    a=Path(tr).read_text(errors='replace')
    b=Path(tst).read_text(errors='replace')
    def pairs(t):
        return re.findall(r'\*\*\*auc:\s*([0-9.]+).*?\*\*\*uauc:\s*\(?([0-9.]+)',t)
    print("[R2-RESULT]",arm,"Valid epochs",pairs(a),"TEST",pairs(b)[-1:] or "NOT FOUND",flush=True)
PY
