#!/usr/bin/env bash
# S2 historical-source inference normalization audit (NO TRAINING / NO TEST).
# Compare prior raw-eval metrics against an isolated source worktree with ONLY
# GNN output normalized identically to TRAIN before LoRA bottleneck injection.
# Original S2 trained epoch0 L31/L30 checkpoints and run configs remain untouched.
set -euo pipefail
MAIN="/home/bbx/user_lx/GraphLoRA"
SHA="d818fa2943ede4e9a810b032a3422e94d0c77ed0"
ID="${S2_RUN_ID:-20261017_143611}"
RUN_DIR="$MAIN/logs/S2_STRICT_${ID}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONUNBUFFERED=1
mkdir -p "$MAIN/logs"
cd "$MAIN"
exec 9>logs/S2_STRICT_NORM_AUDIT.lock
flock -n 9 || { echo "[S2-NORM-ERROR] audit running"; exit 2; }
test -f "$RUN_DIR/L31.yaml" && test -f "$RUN_DIR/L30.yaml"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$MAIN/logs/S2_STRICT_NORM_AUDIT_${STAMP}"
WT="$MAIN/../GraphLoRA_S2_NORM_AUDIT_${STAMP}"
mkdir -p "$OUT"
git cat-file -e "$SHA^{commit}"
git worktree add --detach "$WT" "$SHA"
test "$(git -C "$WT" rev-parse HEAD)" = "$SHA"
# Replicate the dataset symlinks of the prior S2 strict worktree.
if [[ ! -e "$WT/dataset" ]]; then
  ln -s "$MAIN/dataset" "$WT/dataset"
else
  mkdir -p "$WT/dataset/ml-1m"
  for NAME in movie_graph_train.npz movie_graph_valid.npz movie_graph_test.npz; do
    if [[ ! -e "$WT/dataset/ml-1m/$NAME" ]]; then
      ln -s "$MAIN/dataset/ml-1m/$NAME" "$WT/dataset/ml-1m/$NAME"
    fi
  done
fi
for NAME in movie_graph_train.npz movie_graph_valid.npz movie_graph_test.npz; do
  test -f "$WT/dataset/ml-1m/$NAME"
done

# Patch exactly one line in EVAL. Train forward remains fully unmodified.
python - "$WT/minigpt4/models/minigpt4rec_v2.py" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]);s=p.read_text()
needle="gcn_output = fb['gcn_output'].to(dtype=torch.float16)"
replace="gcn_output = (F.normalize(fb['gcn_output'], dim=-1) * 0.1).to(dtype=torch.float16)"
assert s.count(needle)==1, "exact S2 eval normalization anchor missing or duplicated"
p.write_text(s.replace(needle,replace))
print("[S2-NORM-PATCH] ONE eval expression: raw GNN -> normalize(dim=-1)*0.1",flush=True)
PY
git -C "$WT" diff -- minigpt4/models/minigpt4rec_v2.py > "$OUT/eval_norm_only.diff"
python -m py_compile "$WT/minigpt4/models/minigpt4rec_v2.py"
printf 'trained_layer\teval_gnn_scale\tvalid_auc\tvalid_uauc\tvalid_loss\tcheckpoint\tlog\n' > "$OUT/summary.tsv"
for L in 31 30; do
  mapfile -t CKPTS < <(find "$RUN_DIR/L${L}_output" -maxdepth 3 -type f -name checkpoint_best.pth | sort)
  [[ "${#CKPTS[@]}" -eq 1 ]] || { echo "[S2-NORM-ERROR] wrong checkpoint count for L$L"; exit 3; }
  CKPT="${CKPTS[0]}"
  # ON uses training-matched GNN norm; OFF is independent comparison anchor.
  for SCALE in 0.1 0.0; do
    LOG="$OUT/L${L}_normmatched_gnn${SCALE}.log"
    ODIR="$OUT/L${L}_normmatched_gnn${SCALE}_output"
    echo "[S2-NORM-START] trained_layer=$L scale=$SCALE"
    (
      cd "$WT"
      python -u train_collm_mf_din.py --cfg-path "$RUN_DIR/L${L}.yaml" --options \
        "model.ckpt=$CKPT" \
        "model.gnn_scale=$SCALE" \
        "run.evaluate=true" \
        "run.test_splits=[valid]" \
        "run.output_dir=$ODIR" 2>&1 | tee "$LOG"
    )
    grep -Fq "Setting LoRA, GCN, and MyLinear for layer $L Done. Replaced 2 modules." "$LOG" || {
      echo "[S2-NORM-ERROR] wrong injected layer or modules";exit 3;
    }
    python - "$LOG" "$OUT/summary.tsv" "$L" "$SCALE" "$CKPT" <<'PY'
import re,sys
from pathlib import Path
log,tsv,layer,scale,ckpt=sys.argv[1:]
a=[x for x in Path(log).read_text(errors="replace").splitlines()
   if "Averaged stats:" in x and "***auc:" in x and "***uauc:" in x]
assert len(a)==1,f"expected exactly one Valid, got {len(a)}"
def extract(pattern):
    m=re.search(pattern,a[0])
    assert m is not None,pattern
    return float(m.group(1))
auc=extract(r"\*\*\*auc:\s*([0-9.]+)")
uauc=extract(r"\*\*\*uauc:\s*\(?([0-9.]+)")
loss=extract(r"\bloss:\s*([0-9.]+)")
with open(tsv,"a") as f:f.write(f"{layer}\t{scale}\t{auc:.9f}\t{uauc:.9f}\t{loss:.9f}\t{ckpt}\t{log}\n")
print(f"[S2-NORM-METRIC] L{layer} scale={scale} Valid AUC={auc:.9f} UAUC={uauc:.9f} loss={loss:.9f}",flush=True)
PY
  done
done
python - "$OUT/summary.tsv" <<'PY'
import csv,sys
with open(sys.argv[1]) as f:rows=list(csv.DictReader(f,delimiter="\t"))
assert len(rows)==4
for l in ("31","30"):
    on=next(x for x in rows if x["trained_layer"]==l and x["eval_gnn_scale"]=="0.1")
    off=next(x for x in rows if x["trained_layer"]==l and x["eval_gnn_scale"]=="0.0")
    print(f"[S2-NORM-DELTA] L{l} normmatched GNN-ON minus GNN-OFF "
          f"AUC={float(on['valid_auc'])-float(off['valid_auc']):+.9f} "
          f"UAUC={float(on['valid_uauc'])-float(off['valid_uauc']):+.9f}")
print("[S2-NORM-COMPLETE] Four Valid-only evaluations. Checkpoints, train code, old S2 run unchanged.")
PY
cat "$OUT/summary.tsv"
