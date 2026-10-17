#!/usr/bin/env bash
# Historical S2 SOURCE, 1 epoch. Exact L31 forward GNN injection as champion.
# ONLY stop L_rec -> GNN/MF via the injection path (semantic L_sem still ON).
# Controlled change: model.train_gnn_with_rec_loss=true -> false.
# All other hyperparameters including gnn_scale=0.1 and optimizer schedule match.
# Does not overwrite historical S2 checkpoint, performs Valid only.
set -euo pipefail
MAIN="/home/bbx/user_lx/GraphLoRA"
SHA="d818fa2943ede4e9a810b032a3422e94d0c77ed0"
ID="${S2_RUN_ID:-20261017_143611}"
WT="${S2_WORKTREE:-/home/bbx/user_lx/GraphLoRA_S2_STRICT_${ID}}"
SRC="$MAIN/logs/S2_STRICT_${ID}/L31.yaml"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONUNBUFFERED=1
cd "$MAIN"
mkdir -p logs
exec 9>logs/S2_STRICT_DETACH.lock
flock -n 9 || { echo "[S2-DETACH-ERROR] job already running"; exit 2; }
test "$(git -C "$WT" rev-parse HEAD)" = "$SHA"
test -f "$SRC"
grep -Fq "[S2-STRICT-STOP]" "$WT/minigpt4/runners/runner_base.py"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$MAIN/logs/S2_STRICT_DETACH_${STAMP}"
mkdir -p "$OUT"
CFG="$OUT/DETACH_GNN.yaml"
LOG="$OUT/DETACH_GNN.log"
python - "$SRC" "$CFG" "$OUT" <<'PY'
import sys,copy,yaml
from pathlib import Path
source,dest,out=sys.argv[1:]
orig=yaml.safe_load(Path(source).read_text())
c=copy.deepcopy(orig)
assert int(c["model"]["gcn_layer"])==31
assert float(c["model"]["gnn_scale"])==0.1
assert c["model"]["train_gnn_with_rec_loss"] is True
assert c["run"]["batch_size_train"]==3 and c["run"]["accum_grad_iters"]==3
assert c["run"]["max_epoch"]==3 and c["run"]["seed"]==42
assert not c["model"].get("ckpt")
c["model"]["train_gnn_with_rec_loss"]=False
c["run"]["output_dir"]=str(Path(out)/"DETACH_GNN_output")
assert {k for k in c["model"] if orig["model"][k]!=c["model"][k]}=={"train_gnn_with_rec_loss"}
assert {k for k in c["run"] if orig["run"][k]!=c["run"][k]}=={"output_dir"}
Path(dest).write_text(yaml.safe_dump(c,sort_keys=False))
print("[S2-DETACH-STATIC] matched S2 L31, only model.train_gnn_with_rec_loss changed true->false",flush=True)
PY
echo "[S2-DETACH-START] $(date -Is) 1 full epoch, GNN forward preserved (scale=0.1), no L_rec gradient through injected GNN"
(
 cd "$WT"
 python -u train_collm_mf_din.py --cfg-path "$CFG" 2>&1 | tee "$LOG"
)
grep -Fq "[S2-STRICT-STOP] finished epoch 0" "$LOG"
grep -Fq "Preparing to inject GCN into: model.layers.31 with gnn_scale: 0.1" "$LOG"
grep -Fq "GNN semantic correction ON: teacher_layer=8, weight=0.0001, rec_loss_updates_gnn=False" "$LOG"
python - "$LOG" <<'PY'
import re,sys
from pathlib import Path
data=Path(sys.argv[1]).read_text(errors="replace").splitlines()
a=[x for x in data if "Averaged stats:" in x and "***auc:" in x and "***uauc:" in x]
assert len(a)==1,len(a)
def f(p):return float(re.search(p,a[0]).group(1))
print("[S2-DETACH-METRIC] epoch0 Valid AUC={:.9f} UAUC={:.9f} ValidLoss={:.9f}".format(
  f(r"\*\*\*auc:\s*([0-9.]+)"),
  f(r"\*\*\*uauc:\s*\(?([0-9.]+)"),
  f(r"\bloss:\s*([0-9.]+)")))
print("[S2-DETACH-ANCHOR] S2 strict L31 rec_loss_updates_gnn=True epoch0 AUC=0.729675260 UAUC=0.684167021")
print("[S2-DETACH-COMPLETE] 1 epoch, Valid only, no Test")
PY
