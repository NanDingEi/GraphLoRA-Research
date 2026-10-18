#!/usr/bin/env bash
# Eight-epoch NO-L31 experiment plan:
# R1 E8 exact seed42 replay (1 epoch); R2 repeat (1); R3 seed43 (1);
# R4 KD-only seed42 (1); R5 E8 CTR+KD trajectory (2); R6 KD-only trajectory (2).
# Total = 8 epochs. Serial, Valid only, no Test.
# After R1/R2, stop if E8 does not reproduce; don't waste remaining budget.
set -euo pipefail

MAIN="/home/bbx/user_lx/GraphLoRA"
SHA="d818fa2943ede4e9a810b032a3422e94d0c77ed0"
NIGHT="${NIGHT8_RUN_DIR:-$MAIN/logs/S2_NO31_NIGHT8_20261018_012408}"
SOURCE="$NIGHT/E6_INPUT_CTR_KD/recipe.yaml"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONUNBUFFERED=1
cd "$MAIN"
mkdir -p logs

# Shared lock prevents overlap with MECH4/TUNE5 or another NO31 run.
exec 9>logs/S2_NO31_MECH4.lock
flock -n 9 || { echo "[NO31-8EP-ERROR] another NO31 serial run holds the GPU lock"; exit 2; }
test -f "$NIGHT/source_patch.diff" || { echo "[NO31-8EP-ERROR] missing patch: $NIGHT/source_patch.diff"; exit 2; }
test -f "$SOURCE" || { echo "[NO31-8EP-ERROR] missing recipe: $SOURCE"; exit 2; }

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$MAIN/logs/S2_NO31_NIGHT8_DIAG_${STAMP}"
WT="/home/bbx/user_lx/GraphLoRA_NO31_NIGHT8_DIAG_${STAMP}"
mkdir -p "$OUT"
START_SEC="$(date +%s)"
echo "[NO31-8EP-ROOT] $OUT"
echo "[NO31-8EP-BASE] E8 seed42 CTR=.05 KD=.05 alpha=0 gnn_scale=0 historical AUC=.719760366 UAUC=.685106415"
echo "[NO31-8EP-PLAN] 1+1+1+1+2+2=8 epochs; serial; no L31 injection; Valid only"

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

# NAME CTR KD SEED EPOCHS. Every arm starts from fresh model initialization.
ARMS=(
  "R1_E8_REPRO_SEED42 0.05 0.05 42 1"
  "R2_E8_REPEAT_SEED42 0.05 0.05 42 1"
  "R3_E8_SEED43 0.05 0.05 43 1"
  "R4_E9_KD_ONLY_SEED42 0.00 0.05 42 1"
  "R5_E8_CTR_KD_2EPOCH 0.05 0.05 42 2"
  "R6_E9_KD_ONLY_2EPOCH 0.00 0.05 42 2"
)
printf 'experiment\tctr_weight\tkd_weight\tseed\tplanned_epochs\tcompleted_epochs\tbest_epoch\tbest_auc\tbest_uauc\tbest_valid_loss\tcheckpoint\n' > "$OUT/summary.tsv"
printf 'experiment\tepoch\tauc\tuauc\tvalid_loss\n' > "$OUT/epochs.tsv"

for i in "${!ARMS[@]}"; do
  ITEM="${ARMS[$i]}"
  read -r NAME CTR KD SEED EPOCHS <<< "$ITEM"
  AD="$OUT/$NAME"
  mkdir -p "$AD"
  CFG="$AD/recipe.yaml"
  LOG="$AD/train.log"

  python - "$SOURCE" "$CFG" "$AD" "$CTR" "$SEED" "$NAME" <<'PY'
from pathlib import Path
import copy,sys,yaml
source,dest,armdir,ctr,seed,name=sys.argv[1:]
orig=yaml.safe_load(Path(source).read_text())
assert float(orig["model"]["gnn_scale"]) == 0.0, "base recipe must have gnn_scale=0"
assert float(orig["model"]["gnn_conf_loss_weight"]) == 0.05, "unexpected source CTR baseline"
assert orig["model"]["gnn_semantic_align"] is True, "keep semantic alignment fixed"
assert orig["model"]["freeze_rec"] is False, "keep MF trainable"
assert int(orig["run"]["max_epoch"]) == 3, "preserve original scheduler horizon"
assert int(orig["run"]["batch_size_train"]) == 3
assert int(orig["run"]["accum_grad_iters"]) == 3
assert orig["run"]["amp"] is True
assert not orig["model"].get("ckpt"), "must start from original initialization, not resume a trained checkpoint"
c=copy.deepcopy(orig)
c["model"]["gnn_conf_loss_weight"]=float(ctr)
c["run"]["seed"]=int(seed)
c["run"]["output_dir"]=str(Path(armdir)/"output")
changed_model={k for k in c["model"] if c["model"][k]!=orig["model"][k]}
changed_run={k for k in c["run"] if c["run"][k]!=orig["run"][k]}
assert changed_model <= {"gnn_conf_loss_weight"}, (name,changed_model)
assert changed_run <= {"seed","output_dir"}, (name,changed_run)
Path(dest).write_text(yaml.safe_dump(c,sort_keys=False))
print(f"[NO31-8EP-CONFIG] {name} ctr={ctr} kd=0.05 seed={seed} epochs=controlled-by-runner gnn_scale=0 alpha=0 batch=3",flush=True)
PY

  echo "[NO31-8EP-START] $NAME ctr=$CTR kd=$KD seed=$SEED epochs=$EPOCHS $(date -Is)"
  (
    cd "$WT"
    S2_INPUT_ALPHA=0.00 S2_SMALL_KD="$KD" S2_NIGHT_EPOCHS="$EPOCHS" \
      python -u train_collm_mf_din.py --cfg-path "$CFG" > "$LOG" 2>&1
  ) || { echo "[NO31-8EP-ERROR] $NAME crashed"; tail -n 100 "$LOG"; exit 3; }

  grep -Fq "[NIGHT8-MODEL] gnn_scale=0 alpha=0.0" "$LOG" || {
    echo "[NO31-8EP-ERROR] $NAME did not confirm no L31 injection / alpha=0"; exit 3;
  }
  grep -Fq "[NIGHT8-LOSS]" "$LOG" || {
    echo "[NO31-8EP-ERROR] $NAME missing auxiliary-loss instrumentation"; exit 3;
  }
  grep -Fq "[NIGHT8-STOP] trained_epochs=$EPOCHS" "$LOG" || {
    echo "[NO31-8EP-ERROR] $NAME did not complete exactly $EPOCHS epoch(s)"; tail -n 80 "$LOG"; exit 3;
  }

  python - "$LOG" "$AD" "$OUT/summary.tsv" "$OUT/epochs.tsv" "$NAME" "$CTR" "$KD" "$SEED" "$EPOCHS" <<'PY'
import re,sys
from pathlib import Path
log,ad,summary,epochs_file,name,ctr,kd,seed,planned=sys.argv[1:]
lines=Path(log).read_text(errors="replace").splitlines()
rows=[s for s in lines if "Averaged stats:" in s and "***auc:" in s and "***uauc:" in s]
assert len(rows)==int(planned),(name,"expected epoch metrics",planned,"found",len(rows))
def getval(row,pattern):
    m=re.search(pattern,row)
    assert m,(name,pattern,row)
    return float(m.group(1))
metrics=[]
for ep,row in enumerate(rows):
    auc=getval(row,r"\*\*\*auc:\s*([0-9.]+)")
    uauc=getval(row,r"\*\*\*uauc:\s*\(?([0-9.]+)")
    loss=getval(row,r"\bloss:\s*([0-9.]+)")
    metrics.append((ep,auc,uauc,loss))
    with Path(epochs_file).open("a") as f:
        f.write(f"{name}\t{ep}\t{auc:.9f}\t{uauc:.9f}\t{loss:.9f}\n")
best=max(metrics,key=lambda x:x[1])
ckpts=list((Path(ad)/"output").rglob("checkpoint_best.pth"))
assert len(ckpts)==1,(name,"checkpoint_best count",ckpts)
with Path(summary).open("a") as f:
    f.write(f"{name}\t{ctr}\t{kd}\t{seed}\t{planned}\t{len(metrics)}\t{best[0]}\t{best[1]:.9f}\t{best[2]:.9f}\t{best[3]:.9f}\t{ckpts[0]}\n")
print(f"[NO31-8EP-METRIC] {name} best_epoch={best[0]} AUC={best[1]:.9f} UAUC={best[2]:.9f} ValidLoss={best[3]:.9f}",flush=True)
PY

  # After both exact replays, gate the remaining six epochs on reproducibility.
  if [[ "$i" == "1" ]]; then
    python - "$OUT/summary.tsv" <<'PY'
import csv,sys
from pathlib import Path
rows=list(csv.DictReader(Path(sys.argv[1]).open(),delimiter="\t"))
assert len(rows)>=2, "both E8 replay runs must have summary rows"
a,b=rows[0],rows[1]
target_auc,target_uauc=0.719760366,0.685106415
checks={
 "R1_anchor_auc": abs(float(a["best_auc"])-target_auc) <= 0.015,
 "R1_anchor_uauc": abs(float(a["best_uauc"])-target_uauc) <= 0.020,
 "R2_anchor_auc": abs(float(b["best_auc"])-target_auc) <= 0.015,
 "R2_anchor_uauc": abs(float(b["best_uauc"])-target_uauc) <= 0.020,
 "repeat_auc_consistent": abs(float(a["best_auc"])-float(b["best_auc"])) <= 0.010,
 "repeat_uauc_consistent": abs(float(a["best_uauc"])-float(b["best_uauc"])) <= 0.015,
}
print("[NO31-8EP-REPRO-GATE] "+" ".join(f"{k}={str(v).lower()}" for k,v in checks.items()),flush=True)
if not all(checks.values()):
    print("[NO31-8EP-GATE-FAIL] E8 replay is not reliable; stop before spending the remaining epoch budget. Inspect config/init/losses first.",flush=True)
    sys.exit(20)
print("[NO31-8EP-GATE-PASS] E8 reproduces; continue with seed and trajectory comparisons.",flush=True)
PY
  fi
done

NOW="$(date +%s)"
echo "[NO31-8EP-COMPLETE] elapsed_sec=$((NOW-START_SEC)); valid only, no Test"
echo "[NO31-8EP-SUMMARY]"
cat "$OUT/summary.tsv"
echo "[NO31-8EP-EPOCHS]"
cat "$OUT/epochs.tsv"
