#!/usr/bin/env bash
# P3 eval-only diagnostic: compare RAW-GNN (old) vs TRAIN-NORMALIZED-GNN
# on EXACTLY THE SAME trained checkpoint. No training and no Test usage.
# Example: P3_STAMP=20261017_011429 AUDIT_LAYERS="30 29" GPU=0 bash run_layer_P3_eval_norm_audit.sh
set -euo pipefail
cd /home/bbx/user_lx/GraphLoRA
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONUNBUFFERED=1
mkdir -p logs
P3_STAMP="${P3_STAMP:-20261017_011429}"
AUDIT_STAMP="$(date +%Y%m%d_%H%M%S)"
DIR="logs/LAYER_P3_NORM_AUDIT_${AUDIT_STAMP}"
mkdir -p "$DIR"
TAB="$DIR/summary.tsv"
printf 'layer\teval_match_train\tvalid_auc\tvalid_uauc\tvalid_loss\tcheckpoint\n' > "$TAB"
read -ra LAYERS <<< "${AUDIT_LAYERS:-30 29}"
python -m py_compile minigpt4/models/minigpt4rec_v2.py

for L in "${LAYERS[@]}"; do
  CFG="tran_configs/LAYER_P3_L${L}.yaml"
  test -f "$CFG"
  BASE="output/LAYER_P3_${P3_STAMP}_L${L}"
  test -d "$BASE" || { echo "[AUDIT-ERROR] missing $BASE"; exit 2; }
  mapfile -t CKPTS < <(find "$BASE" -maxdepth 3 -type f -name 'checkpoint_best.pth' | sort)
  [[ "${#CKPTS[@]}" -eq 1 ]] || {
    echo "[AUDIT-ERROR] expected exactly one checkpoint_best.pth under $BASE, got ${#CKPTS[@]}"; exit 2;
  }
  CKPT="${CKPTS[0]}"
  for MATCH in false true; do
    LOG="${DIR}/L${L}_match_${MATCH}.log"
    OUT="output/LAYER_P3_NORM_AUDIT_${AUDIT_STAMP}_L${L}_match_${MATCH}"
    echo "[AUDIT-START] L${L} match_train=${MATCH} ckpt=${CKPT} $(date -Is)"
    python -u train_collm_mf_din.py --cfg-path "$CFG" --options \
      "model.ckpt=${CKPT}" \
      "model.d1_gcn_eval_match_train=${MATCH}" \
      "run.evaluate=true" "run.test_splits=[valid]" \
      "run.output_dir=${OUT}" \
      2>&1 | tee "$LOG"
    python - "$LOG" "$TAB" "$L" "$MATCH" "$CKPT" <<'PY'
import re,sys
from pathlib import Path
log,table,layer,flag,ckpt=sys.argv[1:]
rows=[s for s in Path(log).read_text(errors="replace").splitlines()
      if "Averaged stats:" in s and "***auc:" in s and "***uauc:" in s]
if len(rows)!=1: raise SystemExit(f"[AUDIT-ERROR] L{layer} {flag}: expected one Valid evaluation, got {len(rows)}")
s=rows[0]
def num(pattern):
    m=re.search(pattern,s)
    if not m: raise SystemExit(f"[AUDIT-ERROR] cannot parse {pattern}")
    return float(m.group(1))
auc=num(r"\*\*\*auc:\s*([0-9.]+)")
uauc=num(r"\*\*\*uauc:\s*\(?([0-9.]+)")
loss=num(r"\bloss:\s*([0-9.]+)")
with open(table,"a") as f:
    f.write(f"{layer}\t{flag}\t{auc:.9f}\t{uauc:.9f}\t{loss:.9f}\t{ckpt}\n")
print(f"[AUDIT-METRIC] L{layer} norm_match_train={flag} Valid AUC={auc:.9f} UAUC={uauc:.9f} Loss={loss:.9f}",flush=True)
PY
  done
done
echo "[AUDIT-COMPLETE] Paired Valid, same checkpoint. No training or Test."
cat "$TAB"
