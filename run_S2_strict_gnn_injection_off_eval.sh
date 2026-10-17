#!/usr/bin/env bash
# S2 strict trained-checkpoint GNN-injection ablation.
# Valid ONLY; 4 sequential evals; NO training or Test.
# S2 epoch0 trained L31 or L30 checkpoints: evaluate GNN scale=0.1 vs 0.0.
set -euo pipefail
MAIN="/home/bbx/user_lx/GraphLoRA"
S2_SHA="d818fa2943ede4e9a810b032a3422e94d0c77ed0"
RUN_ID="${S2_RUN_ID:-20261017_143611}"
WORKTREE="${S2_WORKTREE:-/home/bbx/user_lx/GraphLoRA_S2_STRICT_${RUN_ID}}"
RUN_DIR="$MAIN/logs/S2_STRICT_${RUN_ID}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONUNBUFFERED=1
cd "$MAIN"
mkdir -p logs
exec 9>logs/S2_STRICT_ABLATION.lock
flock -n 9 || { echo "[S2-ABL-ERROR] another ablation process running"; exit 2; }
test -f "$WORKTREE/minigpt4/models/minigpt4rec_v2.py" || { echo "[S2-ABL-ERROR] S2 worktree missing: $WORKTREE"; exit 2; }
test -f "$RUN_DIR/L31.yaml" || { echo "[S2-ABL-ERROR] run config missing $RUN_DIR"; exit 2; }
test -f "$RUN_DIR/L30.yaml" || { echo "[S2-ABL-ERROR] run config missing $RUN_DIR"; exit 2; }
test "$(git -C "$WORKTREE" rev-parse HEAD)" = "$S2_SHA" || { echo "[S2-ABL-ERROR] wrong source commit"; exit 2; }
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$MAIN/logs/S2_STRICT_INJECTION_ABLATION_${STAMP}"
mkdir -p "$OUT"
TSV="$OUT/summary.tsv"
printf 'trained_layer\teval_gnn_scale\tvalid_auc\tvalid_uauc\tvalid_loss\tckpt\tlog\n' > "$TSV"

for L in 31 30; do
  CFG="$RUN_DIR/L${L}.yaml"
  mapfile -t CHECKPOINTS < <(find "$RUN_DIR/L${L}_output" -maxdepth 3 -type f -name checkpoint_best.pth | sort)
  [[ "${#CHECKPOINTS[@]}" -eq 1 ]] || {
    echo "[S2-ABL-ERROR] L$L expected one checkpoint, got ${#CHECKPOINTS[@]}"; exit 2;
  }
  CKPT="${CHECKPOINTS[0]}"
  python - "$CKPT" "$L" <<'PY'
import sys,torch
path,layer=sys.argv[1:]
try: c=torch.load(path,map_location="cpu",mmap=True,weights_only=False)
except TypeError: c=torch.load(path,map_location="cpu",weights_only=False)
assert int(c["epoch"])==0, f"not epoch0: {c.get('epoch')}"
assert int(c["config"]["model"]["gcn_layer"])==int(layer), "trained layer mismatch"
assert abs(float(c["config"]["model"]["gnn_scale"])-0.1)<1e-10
print(f"[S2-ABL-CKPT] layer={layer} epoch0 trained GNN scale=0.1 verified")
PY
  for SCALE in 0.1 0.0; do
    LOG="$OUT/L${L}_eval_gnn${SCALE}.log"
    EVAL_OUT="$OUT/L${L}_eval_gnn${SCALE}_output"
    echo "[S2-ABL-START] trained_layer=$L eval_gnn_scale=$SCALE $(date -Is)"
    (
      cd "$WORKTREE"
      python -u train_collm_mf_din.py --cfg-path "$CFG" --options \
        "model.ckpt=$CKPT" \
        "model.gnn_scale=$SCALE" \
        "run.evaluate=true" \
        "run.test_splits=[valid]" \
        "run.output_dir=$EVAL_OUT" \
        2>&1 | tee "$LOG"
    )
    grep -Fq "Setting LoRA, GCN, and MyLinear for layer $L Done. Replaced 2 modules." "$LOG" || {
      echo "[S2-ABL-ERROR] wrong number of MyLinear modules for L$L"; exit 3;
    }
    python - "$LOG" "$TSV" "$L" "$SCALE" "$CKPT" <<'PY'
import re,sys
from pathlib import Path
log,tsv,layer,scale,ckpt=sys.argv[1:]
lines=Path(log).read_text(errors="replace").splitlines()
rows=[x for x in lines if "Averaged stats:" in x and "***auc:" in x and "***uauc:" in x]
if len(rows)!=1:raise SystemExit(f"[S2-ABL-ERROR] L{layer} scale{scale}: expected one Valid, got {len(rows)}")
r=rows[0]
def get(pattern):
    m=re.search(pattern,r)
    if m is None:raise SystemExit(f"[S2-ABL-ERROR] missing {pattern}")
    return float(m.group(1))
auc=get(r"\*\*\*auc:\s*([0-9.]+)")
uauc=get(r"\*\*\*uauc:\s*\(?([0-9.]+)")
loss=get(r"\bloss:\s*([0-9.]+)")
with open(tsv,"a") as f:
    f.write(f"{layer}\t{scale}\t{auc:.9f}\t{uauc:.9f}\t{loss:.9f}\t{ckpt}\t{log}\n")
print(f"[S2-ABL-METRIC] trained_layer={layer} eval_scale={scale} AUC={auc:.9f} UAUC={uauc:.9f} ValidLoss={loss:.9f}",flush=True)
PY
  done
done
python - "$TSV" <<'PY'
import csv,sys
with open(sys.argv[1]) as f:rs=list(csv.DictReader(f,delimiter="\t"))
assert len(rs)==4,len(rs)
for l in ("31","30"):
    a=next(x for x in rs if x["trained_layer"]==l and x["eval_gnn_scale"]=="0.1")
    b=next(x for x in rs if x["trained_layer"]==l and x["eval_gnn_scale"]=="0.0")
    print(f"[S2-ABL-DELTA] trained_L{l} GNN-OFF minus GNN-ON AUC="
          f"{float(b['valid_auc'])-float(a['valid_auc']):+.9f} "
          f"UAUC={float(b['valid_uauc'])-float(a['valid_uauc']):+.9f} "
          f"ValidLoss={float(b['valid_loss'])-float(a['valid_loss']):+.9f}")
print("[S2-ABL-COMPLETE] Valid ONLY; same checkpoint in each matched pair; no training or Test")
PY
cat "$TSV"
