#!/usr/bin/env bash
# Valid-only causal ablation of S2's TWO learned q_proj/v_proj biases.
# SAME historical S2 checkpoint, code, dataset, layer, and seed.
# legacy_bias: MyLinear's additional learned Q/V biases loaded from checkpoint.
# no_bias:     MyLinear preserves Vicuna's original bias=None; S2 Q/V bias
#              checkpoint keys become unexpected (ignored by strict=False).
# Does NOT train or run Test, and does NOT modify S2 checkpoint.
# Usage: bash run_S2_qv_bias_valid_ablation.sh
set -euo pipefail
cd /home/bbx/user_lx/GraphLoRA
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONUNBUFFERED=1
S2="/home/bbx/user_lx/GraphLoRA/output/S2_groups_acc3/20261010223/checkpoint_best.pth"
CFG="tran_configs/S2_groups_acc3_recovered.yaml"
test -f "$S2" || { echo "[S2-BIAS-ERROR] missing historical S2: $S2" >&2; exit 2; }
test -f "$CFG" || { echo "[S2-BIAS-ERROR] missing cfg: $CFG" >&2; exit 2; }
mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)"
DIR="logs/S2_BIAS_VALID_ABLATION_${STAMP}"
mkdir -p "$DIR"
TSV="$DIR/summary.tsv"
printf 'setting\tpreserve_source_qv_bias\tvalid_auc\tvalid_uauc\tvalid_loss\tlog\n' > "$TSV"

# S2 expected to contain learned Q/V biases; check before expensive GPU work.
python - "$S2" <<'PY'
import sys, torch
p=sys.argv[1]
try: obj=torch.load(p, map_location="cpu", mmap=True, weights_only=False)
except TypeError: obj=torch.load(p, map_location="cpu")
keys=obj["model"]
for name in ("q_proj","v_proj"):
    key=f"llama_model.model.layers.31.self_attn.{name}.bias"
    if key not in keys: raise SystemExit("[S2-BIAS-ERROR] missing historical weight: "+key)
    tensor=keys[key]
    print(f"[S2-BIAS-CKPT] {key}: shape={tuple(tensor.shape)} "
          f"l2={float(tensor.float().norm()):.6f} "
          f"mean={float(tensor.float().mean()):.8f}", flush=True)
print("[S2-BIAS-CKPT] PASS: both original learned S2 biases exist",flush=True)
PY

for SETTING in legacy_bias no_bias; do
  if [[ "$SETTING" == legacy_bias ]]; then
    PRESERVE=false
  else
    PRESERVE=true
  fi
  LOG="$DIR/${SETTING}.log"
  OUT="output/S2_BIAS_VALID_ABLATION_${STAMP}_${SETTING}"
  echo "[S2-BIAS-START] ${SETTING} preserve_source_qv_bias=$PRESERVE $(date -Is)"
  python -u train_collm_mf_din.py --cfg-path "$CFG" --options \
    "model.ckpt=$S2" \
    "model.preserve_source_qv_bias=$PRESERVE" \
    "run.evaluate=true" \
    "run.test_splits=[valid]" \
    "run.batch_size_eval=4" \
    "run.output_dir=$OUT" \
    2>&1 | tee "$LOG"
  python - "$LOG" "$TSV" "$SETTING" "$PRESERVE" <<'PY'
import re,sys
from pathlib import Path
log,tsv,setting,preserve=sys.argv[1:]
s=Path(log).read_text(errors="replace")
auc=re.findall(r"\*\*\*auc:\s*([0-9.]+)",s)
uauc=re.findall(r"\*\*\*uauc:\s*\(?([0-9.]+)",s)
if len(auc)!=1 or len(uauc)!=1:
    raise SystemExit(f"[S2-BIAS-ERROR] {setting} requires exactly one Valid AUC/UAUC, got {len(auc)}, {len(uauc)}")
losses=re.findall(r"Averaged stats:.*?\bloss:\s*([0-9.]+)",s)
if not losses: raise SystemExit("[S2-BIAS-ERROR] missing Valid loss")
a,u,loss=float(auc[0]),float(uauc[0]),float(losses[-1])
with open(tsv,"a") as out:out.write(f"{setting}\t{preserve}\t{a:.9f}\t{u:.9f}\t{loss:.9f}\t{log}\n")
print(f"[S2-BIAS-METRIC] {setting} preserve={preserve} Valid AUC={a:.9f} UAUC={u:.9f} Loss={loss:.9f}",flush=True)
PY
done
echo "[S2-BIAS-COMPLETE] Same S2 checkpoint, two Valid evaluations, no training, no Test"
cat "$TSV"
python - "$TSV" <<'PY'
import csv,sys
with open(sys.argv[1]) as f:r={x["setting"]:x for x in csv.DictReader(f,delimiter="\t")}
a,b=r["legacy_bias"],r["no_bias"]
print("[S2-BIAS-DELTA] NoBias-Minus-Legacy AUC={:+.9f} UAUC={:+.9f} Loss={:+.9f}".format(
 float(b["valid_auc"])-float(a["valid_auc"]),
 float(b["valid_uauc"])-float(a["valid_uauc"]),
 float(b["valid_loss"])-float(a["valid_loss"])))
PY
