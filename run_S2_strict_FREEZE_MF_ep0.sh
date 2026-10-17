#!/usr/bin/env bash
# Historical S2 source, one full epoch; same L31 GNN forward and gradient path.
# Freeze pretrained MF embeddings; GNN/LoRA/Projector and semantic alignment
# remain trainable. This tests whether MF updates are needed for high AUC.
# Controlled change: model.freeze_rec false -> true; output_dir differs.
# CAUTION: blocks BOTH direct LLM->MF and injected GNN->MF gradients.
# No other model hyperparameters changed. Valid only, no Test.
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
exec 9>logs/S2_STRICT_MFFREEZE.lock
flock -n 9 || { echo "[S2-MFFREEZE-ERROR] job already running"; exit 2; }
test "$(git -C "$WT" rev-parse HEAD)" = "$SHA"
test -f "$SRC"
grep -Fq "[S2-STRICT-STOP]" "$WT/minigpt4/runners/runner_base.py"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$MAIN/logs/S2_STRICT_MFFREEZE_${STAMP}"
mkdir -p "$OUT"
CFG="$OUT/MF_FROZEN.yaml"
LOG="$OUT/MF_FROZEN.log"
python - "$SRC" "$CFG" "$OUT" <<'PY'
import sys,copy,yaml
from pathlib import Path
source,dest,out=sys.argv[1:]
orig=yaml.safe_load(Path(source).read_text())
c=copy.deepcopy(orig)
assert int(c["model"]["gcn_layer"])==31
assert float(c["model"]["gnn_scale"])==0.1
assert c["model"]["train_gnn_with_rec_loss"] is True
assert c["model"]["freeze_rec"] is False
assert c["run"]["batch_size_train"]==3 and c["run"]["accum_grad_iters"]==3
assert c["run"]["max_epoch"]==3 and c["run"]["seed"]==42
assert not c["model"].get("ckpt")
c["model"]["freeze_rec"]=True
c["run"]["output_dir"]=str(Path(out)/"MF_FROZEN_output")
assert {k for k in c["model"] if orig["model"][k]!=c["model"][k]}=={"freeze_rec"}
assert {k for k in c["run"] if orig["run"][k]!=c["run"][k]}=={"output_dir"}
Path(dest).write_text(yaml.safe_dump(c,sort_keys=False))
print("[S2-MFFREEZE-STATIC] matched S2 L31, only model.freeze_rec changed false->true",flush=True)
PY
echo "[S2-MFFREEZE-START] $(date -Is) 1 full epoch, GNN forward/backward ON (scale=0.1), MF embeddings frozen"
(
 cd "$WT"
 python -u train_collm_mf_din.py --cfg-path "$CFG" 2>&1 | tee "$LOG"
)
grep -Fq "[S2-STRICT-STOP] finished epoch 0" "$LOG"
grep -Fq "Preparing to inject GCN into: model.layers.31 with gnn_scale: 0.1" "$LOG"
grep -Fq "GNN semantic correction ON: teacher_layer=8, weight=0.0001, rec_loss_updates_gnn=True" "$LOG"
grep -Fq "freeze rec encoder" "$LOG"
python - "$LOG" <<'PY'
import re,sys
from pathlib import Path
data=Path(sys.argv[1]).read_text(errors="replace").splitlines()
a=[x for x in data if "Averaged stats:" in x and "***auc:" in x and "***uauc:" in x]
assert len(a)==1,len(a)
def f(p):return float(re.search(p,a[0]).group(1))
print("[S2-MFFREEZE-METRIC] epoch0 Valid AUC={:.9f} UAUC={:.9f} ValidLoss={:.9f}".format(
  f(r"\*\*\*auc:\s*([0-9.]+)"),
  f(r"\*\*\*uauc:\s*\(?([0-9.]+)"),
  f(r"\bloss:\s*([0-9.]+)")))
print("[S2-MFFREEZE-ANCHOR] S2 strict L31 freeze_rec=False epoch0 AUC=0.729675260 UAUC=0.684167021")
print("[S2-MFFREEZE-COMPLETE] 1 epoch, Valid only, no Test")
PY
