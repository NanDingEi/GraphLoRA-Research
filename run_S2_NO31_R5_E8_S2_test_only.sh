#!/usr/bin/env bash
# Read-only held-out TEST evaluation of four preselected checkpoints:
# R5 epoch1 (NO31 CTR+KD), E8/R1 epoch0 (NO31 CTR+KD), S2 L31 epoch0, NO-GNN epoch0.
# No training, no model selection on Test, no checkpoints written/overwritten.
# S2 source: d818fa2943ede4e9a810b032a3422e94d0c77ed0
# NO31 source: the exact Night8 training patch on the same historical source.
set -euo pipefail

MAIN="/home/bbx/user_lx/GraphLoRA"
SHA="d818fa2943ede4e9a810b032a3422e94d0c77ed0"
NIGHT="${NIGHT8_RUN_DIR:-$MAIN/logs/S2_NO31_NIGHT8_20261018_012408}"
DIAG="${S2_DIAG_RUN_DIR:-$MAIN/logs/S2_NO31_NIGHT8_DIAG_20261019_003549}"
S2="$MAIN/logs/S2_STRICT_20261017_143611"
NOGNN="$MAIN/logs/S2_STRICT_NOGNN_20261017_172652"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONUNBUFFERED=1
cd "$MAIN"
mkdir -p logs
exec 9>logs/S2_NO31_MECH4.lock
flock -n 9 || { echo "[S2-TEST-ERROR] NO31 experiments running; refusing overlap"; exit 2; }
exec 8>logs/S2_STRICT_ABLATION.lock
flock -n 8 || { echo "[S2-TEST-ERROR] strict S2 evaluation running; refusing overlap"; exit 2; }
test -f "$NIGHT/source_patch.diff" || { echo "[S2-TEST-ERROR] missing Night8 patch"; exit 2; }

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$MAIN/logs/S2_NO31_TEST_${STAMP}"
WT_NO31="/home/bbx/user_lx/GraphLoRA_NO31_TEST_${STAMP}"
WT_S2="/home/bbx/user_lx/GraphLoRA_S2_TEST_${STAMP}"
mkdir -p "$OUT"
echo "[S2-TEST-ROOT] $OUT"
echo "[S2-TEST-PLAN] 4 preselected checkpoints; test only; AUC UAUC loss; no training"

# Fresh, independent source trees. The patched NO31 source *asserts* scale=0,
# therefore S2 L31 runs in unpatched historical source, not the NO31 worktree.
git cat-file -e "$SHA^{commit}"
git worktree add --detach "$WT_NO31" "$SHA"
git worktree add --detach "$WT_S2" "$SHA"
for WT in "$WT_NO31" "$WT_S2"; do
  test "$(git -C "$WT" rev-parse HEAD)" = "$SHA"
  if [[ ! -e "$WT/dataset" ]]; then
    ln -s "$MAIN/dataset" "$WT/dataset"
  else
    mkdir -p "$WT/dataset/ml-1m"
    for N in movie_graph_train.npz movie_graph_valid.npz movie_graph_test.npz; do
      if [[ ! -e "$WT/dataset/ml-1m/$N" ]]; then
        ln -s "$MAIN/dataset/ml-1m/$N" "$WT/dataset/ml-1m/$N"
      fi
    done
  fi
  test -f "$WT/dataset/ml-1m/movie_graph_test.npz"
done
git -C "$WT_NO31" apply --check "$NIGHT/source_patch.diff"
git -C "$WT_NO31" apply "$NIGHT/source_patch.diff"
python -m py_compile "$WT_NO31/minigpt4/models/minigpt4rec_v2.py" "$WT_S2/minigpt4/models/minigpt4rec_v2.py"
git -C "$WT_NO31" diff -- minigpt4/models/minigpt4rec_v2.py minigpt4/runners/runner_base.py > "$OUT/no31_source_patch.diff"

R5_CFG="$DIAG/R5_E8_CTR_KD_2EPOCH/recipe.yaml"
R1_CFG="$DIAG/R1_E8_REPRO_SEED42/recipe.yaml"
S2_CFG="$S2/L31.yaml"
NOGNN_CFG="$NOGNN/NO_GNN.yaml"

pick_one() {
  local dir="$1"
  mapfile -t FOUND < <(find "$dir" -maxdepth 4 -type f -name checkpoint_best.pth | sort)
  if [[ "${#FOUND[@]}" -ne 1 ]]; then
    echo "[S2-TEST-ERROR] expected exactly one checkpoint_best under $dir; got ${#FOUND[@]}" >&2
    exit 2
  fi
  printf '%s' "${FOUND[0]}"
}
R5_CKPT="$(pick_one "$DIAG/R5_E8_CTR_KD_2EPOCH/output")"
R1_CKPT="$(pick_one "$DIAG/R1_E8_REPRO_SEED42/output")"
S2_CKPT="$(pick_one "$S2/L31_output")"
NOGNN_CKPT="$(pick_one "$NOGNN/NO_GNN_output")"

# The 4 arms are preselected *before reading any Test metrics*.
# TYPE label cfg ckpt expected_epoch expected_scale kd input_alpha source_tree
# Store arms in a NUL-separated safe TSV (no whitespace within these names/paths).
printf 'model\tselected_epoch\tgnn_scale\tinput_alpha\ttest_auc\ttest_uauc\ttest_loss\tcheckpoint\n' > "$OUT/summary.tsv"

precheck() {
  python - "$1" "$2" "$3" "$4" "$5" "$6" <<'PY'
import sys,yaml,torch
name,recipe,ckpt,expected_epoch,expected_scale,mode=sys.argv[1:]
c=yaml.safe_load(open(recipe))
assert c["run"]["seed"]==42, (name,"unexpected seed")
assert c["run"]["max_epoch"]==3, (name,"unexpected scheduler horizon")
assert abs(float(c["model"]["gnn_scale"])-float(expected_scale)) < 1e-10, (name,"recipe scale")
assert int(c["model"]["gcn_layer"])==31, (name,"not L31")
try:
    x=torch.load(ckpt,map_location="cpu",mmap=True,weights_only=False)
except TypeError:
    x=torch.load(ckpt,map_location="cpu",weights_only=False)
assert int(x["epoch"])==int(expected_epoch), (name,"wrong selected epoch",x.get("epoch"))
assert abs(float(x["config"]["model"]["gnn_scale"])-float(expected_scale)) < 1e-10, (name,"checkpoint scale")
assert int(x["config"]["model"]["gcn_layer"])==31, (name,"checkpoint layer")
if mode=="NO31":
    assert float(c["model"]["gnn_conf_loss_weight"])==0.05, (name,"wrong CTR")
    assert "gnn_conf_kappa" in x["model"], (name,"trained kappa not stored")
elif mode=="S2":
    assert float(c["model"].get("gnn_conf_loss_weight",0))==0, (name,"unexpected CTR")
elif mode=="NOGNN":
    assert float(c["model"].get("gnn_conf_loss_weight",0))==0, (name,"unexpected CTR")
else:
    raise AssertionError("unknown experiment mode")
print(f"[S2-TEST-CKPT] {name} epoch={expected_epoch} gnn_scale={expected_scale} mode={mode} keys={len(x['model'])} file={ckpt}",flush=True)
PY
}

run_arm() {
  local NAME="$1" CFG="$2" CKPT="$3" EP="$4" SCALE="$5" MODE="$6" WT="$7"
  test -f "$CFG" && test -f "$CKPT" || { echo "[S2-TEST-ERROR] missing cfg/ckpt for $NAME"; exit 2; }
  precheck "$NAME" "$CFG" "$CKPT" "$EP" "$SCALE" "$MODE"
  local LOG="$OUT/$NAME.test.log"
  local EVAL_DIR="$OUT/$NAME.eval_output"
  local KD=0.00 ALPHA=0.00
  if [[ "$MODE" == "NO31" ]]; then KD=0.05; fi
  echo "[S2-TEST-START] $NAME selected_epoch=$EP scale=$SCALE KD=$KD alpha=$ALPHA $(date -Is)"
  (
    cd "$WT"
    S2_INPUT_ALPHA="$ALPHA" S2_SMALL_KD="$KD" S2_NIGHT_EPOCHS=1 \
      python -u train_collm_mf_din.py --cfg-path "$CFG" --options \
       "model.ckpt=$CKPT" "run.evaluate=true" "run.test_splits=[test]" \
       "run.output_dir=$EVAL_DIR" > "$LOG" 2>&1
  ) || { echo "[S2-TEST-ERROR] $NAME eval failed"; tail -n 70 "$LOG"; exit 3; }

  grep -Fq "Load MiniGPT4Rec Checkpoint:" "$LOG" || {
    echo "[S2-TEST-ERROR] $NAME checkpoint loader not confirmed"; exit 3;
  }
  grep -Fq "Evaluating on test" "$LOG" || {
    echo "[S2-TEST-ERROR] $NAME did not evaluate Test split"; exit 3;
  }
  if [[ "$MODE" == "NO31" ]]; then
    grep -Fq "[NIGHT8-MODEL] gnn_scale=0 alpha=0.0" "$LOG" || {
      echo "[S2-TEST-ERROR] $NAME has unexpected injection"; exit 3;
    }
  fi
  python - "$LOG" "$OUT/summary.tsv" "$NAME" "$EP" "$SCALE" "$ALPHA" "$CKPT" <<'PY'
from pathlib import Path
import sys,re
log,tsv,name,epoch,scale,alpha,ckpt=sys.argv[1:]
lines=Path(log).read_text(errors="replace").splitlines()
rows=[s for s in lines if "Averaged stats:" in s and "***auc:" in s and "***uauc:" in s]
assert len(rows)==1,(name,"expected one test evaluation summary",len(rows))
r=rows[0]
def v(regex):
    m=re.search(regex,r);assert m,(name,regex,r)
    return float(m.group(1))
auc=v(r"\*\*\*auc:\s*([0-9.]+)")
uauc=v(r"\*\*\*uauc:\s*\(?([0-9.]+)")
loss=v(r"\bloss:\s*([0-9.]+)")
with open(tsv,"a") as f:
    f.write(f"{name}\t{epoch}\t{scale}\t{alpha}\t{auc:.9f}\t{uauc:.9f}\t{loss:.9f}\t{ckpt}\n")
print(f"[S2-TEST-METRIC] {name} epoch={epoch} TestAUC={auc:.9f} TestUAUC={uauc:.9f} TestLoss={loss:.9f}",flush=True)
PY
}

# Primary model first, followed by preselected comparisons.
run_arm "R5_CTR_KD_NO31" "$R5_CFG" "$R5_CKPT" 1 0.0 NO31 "$WT_NO31"
run_arm "E8_CTR_KD_NO31" "$R1_CFG" "$R1_CKPT" 0 0.0 NO31 "$WT_NO31"
run_arm "S2_L31_ORIGINAL" "$S2_CFG" "$S2_CKPT" 0 0.1 S2 "$WT_S2"
run_arm "NO_GNN_BASELINE" "$NOGNN_CFG" "$NOGNN_CKPT" 0 0.0 NOGNN "$WT_S2"

python - "$OUT/summary.tsv" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter="\t"))
assert len(rows)==4, ("Expected four test results",len(rows))
r={row["model"]:row for row in rows}
a=r["R5_CTR_KD_NO31"]
for name in ("S2_L31_ORIGINAL","E8_CTR_KD_NO31","NO_GNN_BASELINE"):
    b=r[name]
    print(f"[S2-TEST-DELTA] R5 minus {name} TestAUC={float(a['test_auc'])-float(b['test_auc']):+.9f} "
          f"TestUAUC={float(a['test_uauc'])-float(b['test_uauc']):+.9f}",flush=True)
print("[S2-TEST-COMPLETE] Four preselected checkpoints; Test only; no model training",flush=True)
PY
cat "$OUT/summary.tsv"
