#!/usr/bin/env bash
# Six-hour NO-L31 tuning sweep around E8. Five independent epoch-0 runs:
# CTR weight sweep, KD weight sweep, and a half-learning-rate stability check.
# All runs use gnn_scale=0, S2_INPUT_ALPHA=0, same seed=42, max_epoch=3,
# batch_size=3, accum_grad_iters=3, and the Night8 patch. Valid only; no Test.
set -euo pipefail
MAIN="/home/bbx/user_lx/GraphLoRA"
SHA="d818fa2943ede4e9a810b032a3422e94d0c77ed0"
NIGHT="${NIGHT8_RUN_DIR:-$MAIN/logs/S2_NO31_NIGHT8_20261018_012408}"
SOURCE="$NIGHT/E6_INPUT_CTR_KD/recipe.yaml"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONUNBUFFERED=1
BUDGET_SEC="${BUDGET_SEC:-21600}"
MIN_REMAINING_SEC="${MIN_REMAINING_SEC:-4200}"
cd "$MAIN"
mkdir -p logs
# Share the same lock as the mechanism sweep to prevent accidental overlap.
exec 9>logs/S2_NO31_MECH4.lock
flock -n 9 || { echo "[NO31-TUNE-ERROR] another NO31 serial run holds the GPU lock"; exit 2; }
test -f "$NIGHT/source_patch.diff" || { echo "[NO31-TUNE-ERROR] missing Night8 patch: $NIGHT/source_patch.diff"; exit 2; }
test -f "$SOURCE" || { echo "[NO31-TUNE-ERROR] missing E8 recipe: $SOURCE"; exit 2; }
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$MAIN/logs/S2_NO31_TUNE5_${STAMP}"
WT="/home/bbx/user_lx/GraphLoRA_NO31_TUNE5_${STAMP}"
mkdir -p "$OUT"
START_SEC="$(date +%s)"
echo "[NO31-TUNE-ROOT] $OUT"
echo "[NO31-TUNE-BASE] E8 CTR=0.05 KD=0.05 alpha=0, historical Valid AUC=0.719760366 UAUC=0.685106415"
git cat-file -e "$SHA^{commit}"
git worktree add --detach "$WT" "$SHA"
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
test -f "$WT/dataset/ml-1m/movie_graph_train.npz"
test -f "$WT/dataset/ml-1m/movie_graph_valid.npz"
git -C "$WT" apply --check "$NIGHT/source_patch.diff"
git -C "$WT" apply "$NIGHT/source_patch.diff"
python -m py_compile "$WT/minigpt4/models/minigpt4rec_v2.py" "$WT/minigpt4/runners/runner_base.py"
git -C "$WT" diff -- minigpt4/models/minigpt4rec_v2.py minigpt4/runners/runner_base.py > "$OUT/source_patch.diff"

# NAME CTR KD LR_SCALE. Each is a fresh run from the same pretrained starting point.
ARMS=(
  "T1_CTR_LOW_KD_BASE 0.02 0.05 1.0"
  "T2_CTR_HIGH_KD_BASE 0.10 0.05 1.0"
  "T3_CTR_BASE_KD_LOW 0.05 0.02 1.0"
  "T4_CTR_BASE_KD_HIGH 0.05 0.10 1.0"
  "T5_E8_HALF_LR 0.05 0.05 0.5"
)
printf 'experiment\tctr_weight\tkd_weight\tlr_scale\tseed\talpha\tauc\tuauc\tvalid_loss\tcheckpoint\n' > "$OUT/summary.tsv"
echo "[NO31-TUNE-PLAN] 5 independent epoch0 runs, serial; six-hour wall-clock guard; no layer31 injection or input fusion"
for ITEM in "${ARMS[@]}"; do
  NOW="$(date +%s)"
  ELAPSED=$((NOW-START_SEC))
  REMAINING=$((BUDGET_SEC-ELAPSED))
  if (( REMAINING < MIN_REMAINING_SEC )); then
    echo "[NO31-TUNE-TIME-GUARD] remaining_sec=$REMAINING required_sec=$MIN_REMAINING_SEC; stopping before next epoch"
    break
  fi
  read -r NAME CTR KD LR_SCALE <<< "$ITEM"
  AD="$OUT/$NAME"
  mkdir -p "$AD"
  CFG="$AD/recipe.yaml"
  LOG="$AD/train.log"
  python - "$SOURCE" "$CFG" "$AD" "$CTR" "$LR_SCALE" "$NAME" <<'PY'
from pathlib import Path
import copy,sys,yaml
source,dest,armdir,ctr,lr_scale,name=sys.argv[1:]
orig=yaml.safe_load(Path(source).read_text())
assert float(orig["model"]["gnn_scale"])==0.0, "base recipe must not inject GNN at L31"
assert float(orig["model"]["gnn_conf_loss_weight"])==0.05, "unexpected E8 CTR baseline"
assert orig["model"]["gnn_semantic_align"] is True, "keep E8 semantic alignment fixed"
assert orig["model"]["freeze_rec"] is False, "keep shared MF trainable"
assert int(orig["run"]["max_epoch"])==3
assert int(orig["run"]["batch_size_train"])==3
assert int(orig["run"]["accum_grad_iters"])==3
assert int(orig["run"]["seed"])==42
assert orig["run"]["amp"] is True
assert not orig["model"].get("ckpt"), "must train from original initialization, not resume E8"
c=copy.deepcopy(orig)
c["model"]["gnn_conf_loss_weight"]=float(ctr)
c["run"]["output_dir"]=str(Path(armdir)/"output")
scale=float(lr_scale)
if scale != 1.0:
    for k in ("init_lr","min_lr","warmup_lr"):
        assert k in c["run"] and float(c["run"][k])>0, f"missing/invalid run.{k}"
        c["run"][k]=float(c["run"][k])*scale
assert {k for k in c["model"] if c["model"][k]!=orig["model"][k]} <= {"gnn_conf_loss_weight"}
allowed={"output_dir"} | ({"init_lr","min_lr","warmup_lr"} if scale!=1.0 else set())
assert {k for k in c["run"] if c["run"][k]!=orig["run"][k]} <= allowed
Path(dest).write_text(yaml.safe_dump(c,sort_keys=False))
kd="0.02" if name.startswith("T3") else ("0.10" if name.startswith("T4") else "0.05")
print(f"[NO31-TUNE-CONFIG] {name} CTR={ctr} LR_scale={scale} KD={kd} "
      "seed=42 alpha=0 gnn_scale=0 batch=3",flush=True)
PY
  echo "[NO31-TUNE-START] $NAME ctr=$CTR kd=$KD lr_scale=$LR_SCALE $(date -Is)"
  (
    cd "$WT"
    S2_INPUT_ALPHA=0.00 S2_SMALL_KD="$KD" S2_NIGHT_EPOCHS=1 \
      python -u train_collm_mf_din.py --cfg-path "$CFG" > "$LOG" 2>&1
  ) || { echo "[NO31-TUNE-ERROR] $NAME crashed"; tail -n 80 "$LOG"; exit 3; }
  grep -Fq "[NIGHT8-MODEL] gnn_scale=0 alpha=0.0" "$LOG" || {
    echo "[NO31-TUNE-ERROR] $NAME did not confirm gnn_scale=0 alpha=0"; exit 3;
  }
  grep -Fq "[NIGHT8-STOP] trained_epochs=1" "$LOG" || {
    echo "[NO31-TUNE-ERROR] $NAME did not complete exactly one epoch"; exit 3;
  }
  grep -Fq "[NIGHT8-LOSS]" "$LOG" || {
    echo "[NO31-TUNE-ERROR] $NAME expected auxiliary-loss instrumentation missing"; exit 3;
  }
  python - "$LOG" "$AD" "$OUT/summary.tsv" "$NAME" "$CTR" "$KD" "$LR_SCALE" <<'PY'
import re,sys
from pathlib import Path
log,ad,summary,name,ctr,kd,lr=sys.argv[1:]
rows=[s for s in Path(log).read_text(errors="replace").splitlines()
      if "Averaged stats:" in s and "***auc:" in s and "***uauc:" in s]
assert len(rows)==1,(name,len(rows))
def val(pattern):
    m=re.search(pattern,rows[0]); assert m,(name,pattern,rows[0])
    return float(m.group(1))
auc=val(r"\*\*\*auc:\s*([0-9.]+)")
uauc=val(r"\*\*\*uauc:\s*\(?([0-9.]+)")
loss=val(r"\bloss:\s*([0-9.]+)")
ckpts=list((Path(ad)/"output").rglob("checkpoint_best.pth"))
assert len(ckpts)==1,(name,ckpts)
with Path(summary).open("a") as f:
    f.write(f"{name}\t{ctr}\t{kd}\t{lr}\t42\t0.00\t{auc:.9f}\t{uauc:.9f}\t{loss:.9f}\t{ckpts[0]}\n")
print(f"[NO31-TUNE-METRIC] {name} epoch0 ValidAUC={auc:.9f} "
      f"ValidUAUC={uauc:.9f} ValidLoss={loss:.9f}",flush=True)
PY
done
NOW="$(date +%s)"
echo "[NO31-TUNE-COMPLETE] elapsed_sec=$((NOW-START_SEC)); completed=$(($(wc -l < "$OUT/summary.tsv")-1)); valid only, no Test"
cat "$OUT/summary.tsv"
