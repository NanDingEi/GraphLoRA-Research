#!/usr/bin/env bash
# S2 NO-L31 causal controls AFTER E8: 4 independent epoch0-only training arms.
# E9: KD only / no input fusion, E10: CTR+wrong-paired KD / no input fusion,
# E11/E12: E8 vs no-GNN, seed43 matched pair.
# ALL gnn_scale=0, no GNN input fusion. No Test.
# Historical d818fa2 source + exactly night8 source_patch.diff, plus small
# kappa-for-KD and soft-teacher negative-control patches.
set -euo pipefail
MAIN="/home/bbx/user_lx/GraphLoRA"
SHA="d818fa2943ede4e9a810b032a3422e94d0c77ed0"
NIGHT="${NIGHT8_RUN_DIR:-$MAIN/logs/S2_NO31_NIGHT8_20261018_012408}"
SOURCE="$NIGHT/E6_INPUT_CTR_KD/recipe.yaml"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONUNBUFFERED=1
cd "$MAIN"
mkdir -p logs
exec 9>logs/S2_NO31_MECH4.lock
flock -n 9 || { echo "[NO31-MECH-ERROR] another mechanism series active";exit 2; }
test -f "$NIGHT/source_patch.diff"
test -f "$SOURCE"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$MAIN/logs/S2_NO31_MECH4_${STAMP}"
WT="/home/bbx/user_lx/GraphLoRA_NO31_MECH4_${STAMP}"
mkdir -p "$OUT"
echo "[NO31-MECH-ROOT] $OUT"
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
python - "$WT" <<'PY'
from pathlib import Path
import sys
wt=Path(sys.argv[1])
p=wt/"minigpt4/models/minigpt4rec_v2.py"
s=p.read_text()
old="""        if self.pref_distill or self.gnn_conf_loss_weight > 0:
            self.gnn_conf_kappa = nn.Parameter(torch.tensor(float(pref_kappa_init)))"""
new="""        if self.pref_distill or self.gnn_conf_loss_weight > 0 or self.s2_small_kd > 0:
            self.gnn_conf_kappa = nn.Parameter(torch.tensor(float(pref_kappa_init)))"""
assert s.count(old)==1,"kappa anchor did not match"
s=s.replace(old,new)
old="""                    target = torch.sigmoid(logits.float().detach())
                loss_kd = F.binary_cross_entropy_with_logits(g_scores,target)"""
new="""                    target = torch.sigmoid(logits.float().detach())
                    # Negative control: keep exact same probability distribution,
                    # but break the user-item-to-LLM-teacher correspondence.
                    _control = os.environ.get("S2_KD_CONTROL","teacher")
                    assert _control in ("teacher","roll")
                    if _control == "roll":
                        assert target.ndim == 1 and target.numel() > 1
                        target = target.roll(1,dims=0)
                loss_kd = F.binary_cross_entropy_with_logits(g_scores,target)"""
assert s.count(old)==1,"teacher target anchor did not match"
s=s.replace(old,new)
p.write_text(s)
print("[NO31-MECH-PATCH] historical S2 + night8 input/CTR/KD; "
      "retain kappa in KD-only; deterministic wrong-paired teacher via roll(1)",flush=True)
PY
python -m py_compile "$WT/minigpt4/models/minigpt4rec_v2.py" "$WT/minigpt4/runners/runner_base.py"
git -C "$WT" diff -- minigpt4/models/minigpt4rec_v2.py minigpt4/runners/runner_base.py > "$OUT/source_patch.diff"
# NAME SEED ALPHA CTR KD TEACHER
ARMS=(
  "E9_KD_ONLY_NO_INPUT 42 0.00 0.00 0.05 teacher"
  "E10_CTR_SHUFFLED_KD 42 0.00 0.05 0.05 roll"
  "E11_E8_SEED43 43 0.00 0.05 0.05 teacher"
  "E12_NO31_SEED43 43 0.00 0.00 0.00 teacher"
)
printf 'arm\tseed\talpha\tctr\tkd\tteacher\tauc\tuauc\tvalid_loss\tcheckpoint\n' > "$OUT/summary.tsv"
echo "[NO31-MECH-PLAN] E9/E10 + seed43 matched E11/E12, 4 independent full epochs (one per arm)"
echo "[NO31-MECH-HIST] NoGNN seed42 auc=.633358; CTR only seed42=.617145; "
echo "                 KD+CTR seed42=.719760; S2-L31 seed42=.729675"
for ITEM in "${ARMS[@]}"; do
  read -r NAME SEED ALPHA CTR KD TEACHER <<< "$ITEM"
  AD="$OUT/$NAME"
  mkdir -p "$AD"
  CFG="$AD/recipe.yaml"
  LOG="$AD/train.log"
  python - "$SOURCE" "$CFG" "$AD" "$SEED" "$CTR" "$NAME" <<'PY'
from pathlib import Path
import sys,copy,yaml
source,dest,armdir,seed,ctr,name=sys.argv[1:]
orig=yaml.safe_load(Path(source).read_text())
assert float(orig["model"]["gnn_scale"])==0.
assert float(orig["model"]["gnn_conf_loss_weight"])==0.05
assert orig["model"]["gnn_semantic_align"] is True
assert orig["model"]["freeze_rec"] is False
assert orig["run"]["max_epoch"]==3
assert orig["run"]["batch_size_train"]==3
assert orig["run"]["accum_grad_iters"]==3
assert orig["run"]["seed"]==42
assert orig["run"]["amp"] is True
assert not orig["model"].get("ckpt")
c=copy.deepcopy(orig)
c["model"]["gnn_conf_loss_weight"]=float(ctr)
c["run"]["seed"]=int(seed)
c["run"]["output_dir"]=str(Path(armdir)/"output")
assert {k for k in c["model"] if c["model"][k]!=orig["model"][k]}<= {"gnn_conf_loss_weight"}
assert {k for k in c["run"] if c["run"][k]!=orig["run"][k]}<= {"seed","output_dir"}
Path(dest).write_text(yaml.safe_dump(c,sort_keys=False))
print(f"[NO31-MECH-CONFIG] {name} seed={seed} CTR={ctr} max_epoch=3, "
      "input_fusion=0, layer31_GNN=0",flush=True)
PY
  echo "[NO31-MECH-START] $NAME seed=$SEED ctr=$CTR kd=$KD teacher=$TEACHER $(date -Is)"
  (
    cd "$WT"
    S2_INPUT_ALPHA="$ALPHA" S2_SMALL_KD="$KD" S2_KD_CONTROL="$TEACHER" S2_NIGHT_EPOCHS=1 \
      python -u train_collm_mf_din.py --cfg-path "$CFG" > "$LOG" 2>&1
  ) || { echo "[NO31-MECH-ERROR] $NAME crashed";tail -n 80 "$LOG";exit 3; }
  grep -Fq "[NIGHT8-MODEL] gnn_scale=0 alpha=0.0" "$LOG" || {
    echo "[NO31-MECH-ERROR] $NAME nonzero fusion or layer31 GNN";exit 3;
  }
  grep -Fq "[NIGHT8-STOP] trained_epochs=1" "$LOG" || {
    echo "[NO31-MECH-ERROR] $NAME incomplete epoch0";exit 3;
  }
  if [[ "$KD" != "0.00" ]]; then
    grep -Fq "[NIGHT8-LOSS]" "$LOG" || {
      echo "[NO31-MECH-ERROR] $NAME KD did not activate";exit 3;
    }
  fi
  python - "$LOG" "$AD" "$OUT/summary.tsv" "$NAME" "$SEED" "$ALPHA" "$CTR" "$KD" "$TEACHER" <<'PY'
import re,sys
from pathlib import Path
log,ad,summary,name,seed,alpha,ctr,kd,teacher=sys.argv[1:]
rows=[s for s in Path(log).read_text(errors="replace").splitlines()
      if "Averaged stats:" in s and "***auc:" in s and "***uauc:" in s]
assert len(rows)==1,(name,len(rows))
def v(p):
    m=re.search(p,rows[0]);assert m,(name,p)
    return float(m.group(1))
auc=v(r"\*\*\*auc:\s*([0-9.]+)")
uauc=v(r"\*\*\*uauc:\s*\(?([0-9.]+)")
loss=v(r"\bloss:\s*([0-9.]+)")
files=list((Path(ad)/"output").rglob("checkpoint_best.pth"))
assert len(files)==1,(name,files)
with Path(summary).open("a") as f:
    f.write(f"{name}\t{seed}\t{alpha}\t{ctr}\t{kd}\t{teacher}\t"
            f"{auc:.9f}\t{uauc:.9f}\t{loss:.9f}\t{files[0]}\n")
print(f"[NO31-MECH-METRIC] {name} epoch0 ValidAUC={auc:.9f} "
      f"ValidUAUC={uauc:.9f} ValidLoss={loss:.9f}",flush=True)
PY
done
echo "[NO31-MECH-COMPLETE] 4 serial full epochs, no GNN injection or input fusion, Valid only"
cat "$OUT/summary.tsv"
