#!/usr/bin/env bash
# S2 NO-L31 followup: P3 same E6 checkpoint inference alpha=0.10/0,
# then 2 independent 1-epoch train ablations P1 KD-only and P2 no-input fusion.
# Original night8 source patch reused from disk. No original worktree edits.
# All train arms: gnn_scale=0; max_epoch remains 3; one full epoch+Valid.
set -euo pipefail
MAIN="/home/bbx/user_lx/GraphLoRA"
SHA="d818fa2943ede4e9a810b032a3422e94d0c77ed0"
NIGHT="${NIGHT8_RUN_DIR:-$MAIN/logs/S2_NO31_NIGHT8_20261018_012408}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONUNBUFFERED=1
cd "$MAIN"
mkdir -p logs
exec 9>logs/S2_NO31_followup.lock
flock -n 9 || { echo "[NO31-FOLLOWUP-ERROR] another followup run active"; exit 2; }
test -f "$NIGHT/source_patch.diff" || { echo "[NO31-FOLLOWUP-ERROR] night8 source patch not found: $NIGHT"; exit 2; }
test -f "$NIGHT/E6_INPUT_CTR_KD/recipe.yaml" || { echo "[NO31-FOLLOWUP-ERROR] E6 training recipe not found"; exit 2; }
mapfile -t CKPTS < <(find "$NIGHT/E6_INPUT_CTR_KD/output" -type f -name checkpoint_best.pth | sort)
[[ "${#CKPTS[@]}" -eq 1 ]] || { echo "[NO31-FOLLOWUP-ERROR] expected one E6 checkpoint, found ${#CKPTS[@]}"; exit 2; }
E6_CKPT="${CKPTS[0]}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$MAIN/logs/S2_NO31_FOLLOWUP_${STAMP}"
WT="/home/bbx/user_lx/GraphLoRA_NO31_FOLLOWUP_${STAMP}"
mkdir -p "$OUT"
echo "[NO31-FOLLOWUP-ROOT] $OUT"
echo "[NO31-FOLLOWUP-E6] $E6_CKPT"
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
test -f "$WT/dataset/ml-1m/movie_graph_valid.npz"
test -f "$WT/dataset/ml-1m/movie_graph_train.npz"
git -C "$WT" apply --check "$NIGHT/source_patch.diff"
git -C "$WT" apply "$NIGHT/source_patch.diff"

# Independent sanity checks before ANY GPU work. Additional code change:
# trainable kappa must remain in KD-only (CTR=0) as in E6 (CTR>0).
# Eval score dump is read-only; logs same user/item/label/logit for paired scores.
python - "$WT" "$E6_CKPT" "$NIGHT" <<'PY'
from pathlib import Path
import torch,yaml,sys
wt=Path(sys.argv[1]);ck=Path(sys.argv[2]);night=Path(sys.argv[3])
m=wt/"minigpt4/models/minigpt4rec_v2.py"
s=m.read_text()
old="""        if self.pref_distill or self.gnn_conf_loss_weight > 0:
            self.gnn_conf_kappa = nn.Parameter(torch.tensor(float(pref_kappa_init)))"""
new="""        if self.pref_distill or self.gnn_conf_loss_weight > 0 or self.s2_small_kd > 0:
            self.gnn_conf_kappa = nn.Parameter(torch.tensor(float(pref_kappa_init)))"""
assert s.count(old)==1, "kappa source anchor mismatch"
m.write_text(s.replace(old,new))
t=wt/"minigpt4/tasks/rec_base_task.py"
v=t.read_text()
old="""                    users.extend(samples['UserID'].detach().cpu().numpy())
                    results_logits.extend(eval_output['logits'].detach().cpu().numpy())
                    labels.extend(samples['label'].detach().cpu().numpy())"""
new="""                    users.extend(samples['UserID'].detach().cpu().numpy())
                    results_logits.extend(eval_output['logits'].detach().cpu().numpy())
                    labels.extend(samples['label'].detach().cpu().numpy())
                    # NO31 P3: dump logits exactly before in-place thresholding.
                    import os as _no31_os
                    if _no31_os.environ.get("S2_SCORE_DUMP_PATH"):
                        _u=samples['UserID'].detach().cpu().reshape(-1).tolist()
                        _i=samples['TargetItemID'].detach().cpu().reshape(-1).tolist()
                        _y=samples['label'].detach().cpu().reshape(-1).tolist()
                        _z=eval_output['logits'].detach().cpu().reshape(-1).tolist()
                        assert len(_u)==len(_i)==len(_y)==len(_z)
                        with open(_no31_os.environ['S2_SCORE_DUMP_PATH'],'a') as _f:
                            for _a,_b,_c,_d in zip(_u,_i,_y,_z):
                                _f.write(f'{int(_a)},{int(_b)},{int(_c)},{float(_d):.12g}\\n')
"""
assert v.count(old)==1, "logit dump anchor mismatch"
t.write_text(v.replace(old,new))
try: obj=torch.load(ck,map_location="cpu",mmap=True,weights_only=False)
except (TypeError,RuntimeError):obj=torch.load(ck,map_location="cpu",weights_only=False)
assert obj["epoch"]==1,"Only E6 epoch1 best checkpoint expected"
cfg=obj["config"]["model"]
assert float(cfg["gnn_scale"])==0.0, "E6 unexpectedly injected GNN at layer31"
assert int(cfg["gcn_layer"])==31
assert float(cfg["gnn_conf_loss_weight"])==0.05
assert "gnn_conf_kappa" in obj["model"], "E6 kappa missing"
recipe=yaml.safe_load((night/"E6_INPUT_CTR_KD/recipe.yaml").read_text())
assert float(recipe["model"]["gnn_scale"])==0.0
assert float(recipe["model"]["gnn_conf_loss_weight"])==0.05
print("[NO31-FOLLOWUP-PATCH] Night8 source applied exactly; KD-only keeps trainable kappa; read-only eval logit dump",flush=True)
print("[NO31-FOLLOWUP-E6-CHECK] trained_epoch=1, gnn_scale=0, GNN CTR=0.05, learned_kappa=yes",flush=True)
PY
python -m py_compile "$WT/minigpt4/models/minigpt4rec_v2.py" "$WT/minigpt4/runners/runner_base.py" "$WT/minigpt4/tasks/rec_base_task.py"
git -C "$WT" diff -- minigpt4/models/minigpt4rec_v2.py minigpt4/runners/runner_base.py minigpt4/tasks/rec_base_task.py > "$OUT/full_patch.diff"

# P3: SAME trained E6 checkpoint, SAME seed/data, 0.10 vs 0.00 input fusion.
for ALPHA in 0.10 0.00; do
  CFG="$NIGHT/E6_INPUT_CTR_KD/recipe.yaml"
  SCORES="$OUT/E6_eval_alpha_${ALPHA}.csv"
  LOG="$OUT/E6_eval_alpha_${ALPHA}.log"
  : > "$SCORES"
  echo "[NO31-EVAL-START] E6 best ckpt, alpha=$ALPHA, train=false, Valid only $(date -Is)"
  (
    cd "$WT"
    S2_INPUT_ALPHA="$ALPHA" S2_SMALL_KD=0.05 S2_NIGHT_EPOCHS=1 \
    S2_SCORE_DUMP_PATH="$SCORES" \
    python -u train_collm_mf_din.py --cfg-path "$CFG" --options \
      "model.ckpt=$E6_CKPT" "run.evaluate=true" "run.test_splits=[valid]" \
      "run.output_dir=$OUT/E6_eval_alpha_${ALPHA}_output" > "$LOG" 2>&1
  ) || { echo "[NO31-FOLLOWUP-ERROR] E6 eval alpha=$ALPHA";tail -n 75 "$LOG";exit 3; }
  grep -Fq "[NIGHT8-MODEL] gnn_scale=0 alpha=${ALPHA%0}" "$LOG" || {
    echo "[NO31-FOLLOWUP-ERROR] eval did not select alpha=$ALPHA"; exit 3;
  }
  python - "$LOG" "$ALPHA" <<'PY'
from pathlib import Path
import re,sys
log,alpha=sys.argv[1:]
rows=[s for s in Path(log).read_text(errors="replace").splitlines()
      if "Averaged stats:" in s and "***auc:" in s and "***uauc:" in s]
assert len(rows)==1,rows
def val(p):
    m=re.search(p,rows[0]);assert m,p
    return float(m.group(1))
auc=val(r"\*\*\*auc:\s*([0-9.]+)")
uauc=val(r"\*\*\*uauc:\s*\(?([0-9.]+)")
loss=val(r"\bloss:\s*([0-9.]+)")
print(f"[NO31-EVAL-METRIC] alpha={alpha} AUC={auc:.9f} "
      f"UAUC={uauc:.9f} ValidLoss={loss:.9f}",flush=True)
PY
done
python - "$OUT" <<'PY'
import csv,sys,numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
out=Path(sys.argv[1])
def load(alpha):
    p=out/f"E6_eval_alpha_{alpha}.csv"
    with p.open() as f:
        rows=[[float(z) for z in row] for row in csv.reader(f) if row]
    x=np.asarray(rows,dtype=np.float64)
    assert x.ndim==2 and x.shape[1]==4 and len(x)>0,x.shape
    assert np.isfinite(x).all()
    return x
on=load("0.10");off=load("0.00")
assert on.shape==off.shape,(on.shape,off.shape)
assert np.array_equal(on[:,:3],off[:,:3]),"Paired Valid sample order mismatched"
a=roc_auc_score(on[:,2],on[:,3])
b=roc_auc_score(off[:,2],off[:,3])
d=np.abs(off[:,3]-on[:,3])
print(f"[NO31-EVAL-PAIRED] rows={len(on)} alpha_on_auc={a:.9f} "
      f"alpha_off_auc={b:.9f} off_minus_on_auc={b-a:+.9f}",flush=True)
print(f"[NO31-EVAL-LOGITS] mean_abs={np.mean(d):.9g} median_abs={np.median(d):.9g} "
      f"p95_abs={np.quantile(d,.95):.9g} max_abs={np.max(d):.9g} "
      f"exactly_equal={np.count_nonzero(d==0)}/{len(d)}",flush=True)
print("[NO31-EVAL-COMPLETE] E6 same checkpoint, Valid only, no training")
PY

# P1 / P2: isolated, separately initialized runs, one full epoch each.
# E7: same E6 input alpha but GNN hard-label CTR=0, retain kappa via patch.
# E8: same E6 CTR+KD but input alpha=0. Verify GNN soft supervision through
# shared MF alone can improve LLM (GNN never enters final prediction).
ARMS=(
  "E7_INPUT_KD_ONLY 0.10 0.00 0.05"
  "E8_CTR_KD_NO_INPUT 0.00 0.05 0.05"
)
printf 'experiment\tauc\tuauc\tvalid_loss\tcheckpoint\n' > "$OUT/train_ablation_summary.tsv"
for ARM in "${ARMS[@]}"; do
  read -r NAME ALPHA CTR KD <<< "$ARM"
  ARMOUT="$OUT/$NAME"
  mkdir -p "$ARMOUT"
  CFG="$ARMOUT/recipe.yaml"
  LOG="$ARMOUT/train.log"
  python - "$NIGHT/E6_INPUT_CTR_KD/recipe.yaml" "$CFG" "$ARMOUT" "$CTR" "$ALPHA" "$KD" "$NAME" <<'PY'
import copy,yaml,sys
from pathlib import Path
original,dest,armout,ctr,alpha,kd,name=sys.argv[1:]
cfg=yaml.safe_load(Path(original).read_text())
assert float(cfg["model"]["gnn_scale"])==0
assert float(cfg["model"]["gnn_conf_loss_weight"])==0.05
assert int(cfg["run"]["max_epoch"])==3
assert int(cfg["run"]["seed"])==42
assert cfg["run"]["accum_grad_iters"]==3
assert cfg["model"]["train_gnn_with_rec_loss"] is True
assert cfg["model"]["freeze_rec"] is False
c=copy.deepcopy(cfg)
c["model"]["gnn_conf_loss_weight"]=float(ctr)
c["run"]["output_dir"]=str(Path(armout)/"output")
assert {k for k in c["model"] if c["model"][k]!=cfg["model"][k]}<= {"gnn_conf_loss_weight"}
assert {k for k in c["run"] if c["run"][k]!=cfg["run"][k]}== {"output_dir"}
Path(dest).write_text(yaml.safe_dump(c,sort_keys=False))
print(f"[NO31-TRAIN-CFG] {name} alpha={alpha} CTR={ctr} KD={kd} "
      "gnn_scale=0 max_epoch=3 seed=42 planned_epochs=1",flush=True)
PY
  echo "[NO31-TRAIN-START] $NAME alpha=$ALPHA ctr=$CTR kd=$KD $(date -Is)"
  (
    cd "$WT"
    S2_INPUT_ALPHA="$ALPHA" S2_SMALL_KD="$KD" S2_NIGHT_EPOCHS=1 \
    python -u train_collm_mf_din.py --cfg-path "$CFG" > "$LOG" 2>&1
  ) || { echo "[NO31-FOLLOWUP-ERROR] train $NAME failed";tail -n 75 "$LOG";exit 3; }
  grep -Fq "[NIGHT8-MODEL] gnn_scale=0 alpha=${ALPHA%0}" "$LOG" || {
    echo "[NO31-FOLLOWUP-ERROR] $NAME alpha/gnn mismatch";exit 3;
  }
  grep -Fq "[NIGHT8-STOP] trained_epochs=1" "$LOG" || {
    echo "[NO31-FOLLOWUP-ERROR] $NAME missing epoch 0 checkpoint";exit 3;
  }
  grep -Fq "[NIGHT8-LOSS]" "$LOG" || {
    echo "[NO31-FOLLOWUP-ERROR] $NAME CTR/KD not activated";exit 3;
  }
  python - "$LOG" "$ARMOUT" "$OUT/train_ablation_summary.tsv" "$NAME" <<'PY'
import re,sys
from pathlib import Path
log,armout,summary,name=sys.argv[1:]
rows=[x for x in Path(log).read_text(errors="replace").splitlines()
      if "Averaged stats:" in x and "***auc:" in x and "***uauc:" in x]
assert len(rows)==1,(name,len(rows))
def val(p):
    m=re.search(p,rows[0]);assert m,p
    return float(m.group(1))
auc=val(r"\*\*\*auc:\s*([0-9.]+)")
uauc=val(r"\*\*\*uauc:\s*\(?([0-9.]+)")
loss=val(r"\bloss:\s*([0-9.]+)")
ckpt=list((Path(armout)/"output").rglob("checkpoint_best.pth"))
assert len(ckpt)==1,ckpt
with open(summary,"a") as f:
    f.write(f"{name}\t{auc:.9f}\t{uauc:.9f}\t{loss:.9f}\t{ckpt[0]}\n")
print(f"[NO31-TRAIN-METRIC] {name} epoch=0 ValidAUC={auc:.9f} "
      f"ValidUAUC={uauc:.9f} ValidLoss={loss:.9f}",flush=True)
PY
done
echo "[NO31-FOLLOWUP-COMPLETE] 2 paired E6 evals + 2 one-epoch independent runs. No Test."
cat "$OUT/train_ablation_summary.tsv"
