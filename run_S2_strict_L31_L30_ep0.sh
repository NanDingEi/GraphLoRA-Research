#!/usr/bin/env bash
# S2 SOURCE-CONTROLLED one-epoch layer comparison.
# ORIGINAL S2 source at d818fa2943 (isolated git worktree), actual S2 checkpoint
# config (not an inferred YAML), and same seed/data/optimizer.
# Both arms train exactly one full epoch, retaining max_epoch=3 in the scheduler.
# The worktree runner gets only a control-flow break AFTER epoch-0 Valid/checkpoint.
# The ONLY model/training hyperparameter changed between arms: model.gcn_layer.
# Layers: 31 (reproduce S2 in same epoch window) then 30 (one-layer ablation).
# NO Test. Never overwrite historical S2 checkpoint.
# Usage: nohup env GPU=0 bash run_S2_strict_L31_L30_ep0.sh > logs/S2_STRICT_launcher.log 2>&1 < /dev/null &
set -euo pipefail

MAIN="/home/bbx/user_lx/GraphLoRA"
S2_SHA="d818fa2943ede4e9a810b032a3422e94d0c77ed0"
S2_CKPT="$MAIN/output/S2_groups_acc3/20261010223/checkpoint_best.pth"
cd "$MAIN"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONUNBUFFERED=1
mkdir -p logs output
exec 9>logs/S2_STRICT_serial.lock
flock -n 9 || { echo "[S2-STRICT-ERROR] Another S2 strict comparison is running"; exit 2; }
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$MAIN/logs/S2_STRICT_${STAMP}"
mkdir -p "$LOGDIR"
WORKTREE="${S2_WORKTREE:-/home/bbx/user_lx/GraphLoRA_S2_STRICT_${STAMP}}"
TSV="$LOGDIR/summary.tsv"
printf 'layer\tvalid_auc\tvalid_uauc\tvalid_loss\tcheckpoint\tlog\n' > "$TSV"

test -f "$S2_CKPT" || { echo "[S2-STRICT-ERROR] Missing original S2 checkpoint $S2_CKPT"; exit 2; }
test -d "$MAIN/dataset/ml-1m" || { echo "[S2-STRICT-ERROR] Missing ML-1M dataset"; exit 2; }
if ! git cat-file -e "${S2_SHA}^{commit}" 2>/dev/null; then
  git fetch origin "$S2_SHA"
fi
test ! -e "$WORKTREE" || { echo "[S2-STRICT-ERROR] worktree destination already exists: $WORKTREE"; exit 2; }
git worktree add --detach "$WORKTREE" "$S2_SHA"
cd "$WORKTREE"
test "$(git rev-parse HEAD)" = "$S2_SHA"
echo "[S2-STRICT-SOURCE] $(git rev-parse HEAD)"
# The source uses ./dataset/ml-1m/movie_graph_*.npz for GNN adjacency.
# All original dataset bytes remain in the main repository; link, do not copy.
if [[ ! -e dataset ]]; then
  ln -s "$MAIN/dataset" dataset
else
  # Worktree may contain tracked metadata but lack untracked graph data.
  mkdir -p dataset/ml-1m
  for NAME in movie_graph_train.npz movie_graph_valid.npz movie_graph_test.npz; do
    if [[ ! -e "dataset/ml-1m/$NAME" ]]; then
      ln -s "$MAIN/dataset/ml-1m/$NAME" "dataset/ml-1m/$NAME"
    fi
  done
fi
test -f "prompts/collm_movie_idlist.txt"
test -f "dataset/ml-1m/movie_graph_train.npz"
test -f "dataset/ml-1m/movie_graph_valid.npz"
test -f "dataset/ml-1m/movie_graph_test.npz"

# The S2 runner from 2026-10-11 has NO stop_after_epoch; setting max_epoch=1
# would CHANGE the cosine LR schedule. Instead insert one stop after finishing
# the epoch-0 validation/checkpoint. This is the only modification to source,
# identical in both arms, and happens after all optimization/evaluation work.
python - <<'PY'
from pathlib import Path
p=Path("minigpt4/runners/runner_base.py")
s=p.read_text()
old="""                if self.evaluate_only:
                    break

                if self.config.run_cfg.distributed:
                    dist.barrier()
"""
new="""                if self.evaluate_only:
                    break

                # S2-STRICT: stop after the first COMPLETE epoch and Valid;
                # max_epoch=3 remains unchanged for the cosine LR scheduler.
                if cur_epoch == 0:
                    logging.info("[S2-STRICT-STOP] finished epoch 0; max_epoch=%d remains unchanged",
                                 self.max_epoch)
                    break

                if self.config.run_cfg.distributed:
                    dist.barrier()
"""
assert s.count(old)==1, "Historical S2 runner does not match expected patch anchor"
p.write_text(s.replace(old,new))
print("[S2-STRICT-PATCH] one post-validation early exit; optimization path untouched",flush=True)
PY
git diff -- minigpt4/runners/runner_base.py > "$LOGDIR/runner_stop_only.diff"
python -m py_compile minigpt4/models/minigpt4rec_v2.py minigpt4/runners/runner_base.py

# Generate BOTH configs from the effective config SAVED IN THE ORIGINAL S2 CKPT.
# This removes the ambiguity of the earlier reconstructed S2 YAML.
# We alter output_dir only to protect old experiments; model.gcn_layer is the
# ONLY other difference. Both arms start fresh with model.ckpt=None.
python - "$S2_CKPT" "$LOGDIR" <<'PY'
import sys,copy,json
from pathlib import Path
import torch,yaml
src,output=sys.argv[1:]
try: ckpt=torch.load(src,map_location="cpu",mmap=True,weights_only=False)
except (TypeError,RuntimeError): ckpt=torch.load(src,map_location="cpu",weights_only=False)
cfg=ckpt["config"]
assert cfg["model"]["gcn_layer"]==31,cfg["model"]["gcn_layer"]
assert cfg["run"]["batch_size_train"]==3
assert cfg["run"]["accum_grad_iters"]==3
assert cfg["run"]["max_epoch"]==3
assert cfg["run"]["warmup_steps"]==200
assert cfg["run"]["seed"]==42
assert cfg["model"]["gnn_semantic_align"] is True
assert cfg["model"]["train_gnn_with_rec_loss"] is True
assert not cfg["model"].get("preserve_source_qv_bias",False)
assert not cfg["model"].get("fix_autocast_cache",False)
assert not cfg["model"].get("inj_diag",False)
assert not cfg["model"].get("ckpt"), "Original S2 training must be fresh, not checkpoint continuation"
out=Path(output)
a=copy.deepcopy(cfg)
b=copy.deepcopy(cfg)
a["model"]["gcn_layer"]=31
b["model"]["gcn_layer"]=30
a["run"]["output_dir"]=str(out/"L31_output")
b["run"]["output_dir"]=str(out/"L30_output")
assert a["model"]!=b["model"] and set(k for k in a["model"] if a["model"][k]!=b["model"][k])=={"gcn_layer"}
assert {k for k in a["run"] if a["run"][k]!=b["run"][k]}=={"output_dir"}
for tag,config in (("L31",a),("L30",b)):
    with (out/f"{tag}.yaml").open("w") as f:
        yaml.safe_dump(config,f,allow_unicode=True,sort_keys=False)
    print(f"[S2-STRICT-CONFIG] {tag} source_checkpoint={src} "
          f"layer={config['model']['gcn_layer']} "
          f"batch={config['run']['batch_size_train']} "
          f"accum={config['run']['accum_grad_iters']} "
          f"max_epoch={config['run']['max_epoch']} "
          f"warmup={config['run']['warmup_steps']}",flush=True)
print("[S2-STRICT-STATIC] configs identical except model.gcn_layer/run.output_dir",flush=True)
PY

# Run sequentially; no performance gate between arms.
for L in 31 30; do
  CFG="$LOGDIR/L${L}.yaml"
  LOG="$LOGDIR/L${L}.log"
  echo "[S2-STRICT-START] L$L, historical S2 source, 1 epoch $(date -Is)"
  python -u train_collm_mf_din.py --cfg-path "$CFG" 2>&1 | tee "$LOG"
  grep -Fq "[S2-STRICT-STOP] finished epoch 0" "$LOG" || {
    echo "[S2-STRICT-ERROR] missing one-epoch stop confirmation for L$L"; exit 3;
  }
  grep -Fq "Setting LoRA, GCN, and MyLinear for layer $L Done. Replaced 2 modules." "$LOG" || {
    echo "[S2-STRICT-ERROR] incorrect injection q/v replacement count L$L"; exit 3;
  }
  python - "$LOG" "$TSV" "$L" "$LOGDIR" <<'PY'
import re,sys
from pathlib import Path
log,tsv,layer,out=sys.argv[1:]
t=Path(log).read_text(errors="replace")
auc=re.findall(r"\*\*\*auc:\s*([0-9.]+)",t)
uauc=re.findall(r"\*\*\*uauc:\s*\(?([0-9.]+)",t)
val=[x for x in t.splitlines() if "Averaged stats:" in x and "***auc:" in x]
if len(auc)!=1 or len(uauc)!=1 or len(val)!=1:
    raise SystemExit(f"[S2-STRICT-ERROR] L{layer}: expected exactly one completed Valid")
m=re.search(r"\bloss:\s*([0-9.]+)",val[0])
if not m:raise SystemExit("[S2-STRICT-ERROR] missing Valid loss")
checkpoints=list(Path(out,f"L{layer}_output").rglob("checkpoint_best.pth"))
if len(checkpoints)!=1:
    raise SystemExit(f"[S2-STRICT-ERROR] L{layer}: checkpoint count {len(checkpoints)}")
with open(tsv,"a") as f:
    f.write(f"{layer}\t{float(auc[0]):.9f}\t{float(uauc[0]):.9f}\t{float(m.group(1)):.9f}\t{checkpoints[0]}\t{log}\n")
print(f"[S2-STRICT-METRIC] L{layer} epoch0 Valid AUC={float(auc[0]):.9f} "
      f"UAUC={float(uauc[0]):.9f} Loss={float(m.group(1)):.9f}",flush=True)
PY
done

echo "[S2-STRICT-COMPLETE] BOTH original-S2-source arms finished one epoch each; no Test"
echo "[S2-STRICT-RESULTS] $TSV"
cat "$TSV"
