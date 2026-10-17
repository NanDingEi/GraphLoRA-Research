#!/usr/bin/env bash
# Historical S2 L31, one epoch: freeze NGCF parameters but keep MF trainable.
# Forward GNN injection and gradient THROUGH GNN to MF remain enabled.
# Original worktree, original checkpoints, other experiments untouched.
set -euo pipefail
MAIN="/home/bbx/user_lx/GraphLoRA"
SHA="d818fa2943ede4e9a810b032a3422e94d0c77ed0"
RUN_ID="${S2_RUN_ID:-20261017_143611}"
BASE="$MAIN/logs/S2_STRICT_${RUN_ID}/L31.yaml"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONUNBUFFERED=1
cd "$MAIN"
mkdir -p logs
exec 9>logs/S2_STRICT_FREEZE_GNN.lock
flock -n 9 || { echo "[S2-FGNN-ERROR] another run is active"; exit 2; }
test -f "$BASE"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$MAIN/logs/S2_STRICT_FREEZE_GNN_${STAMP}"
WT="$MAIN/../GraphLoRA_S2_FGNN_${STAMP}"
CFG="$OUT/FREEZE_GNN.yaml"
LOG="$OUT/FREEZE_GNN.log"
mkdir -p "$OUT"
git worktree add --detach "$WT" "$SHA"
test "$(git -C "$WT" rev-parse HEAD)" = "$SHA"
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
test -f "$WT/dataset/ml-1m/movie_graph_train.npz"
test -f "$WT/dataset/ml-1m/movie_graph_valid.npz"
python - "$BASE" "$CFG" "$OUT" "$WT" <<'PY'
from pathlib import Path
import sys,copy,yaml
src,dst,out,wt=sys.argv[1:]
base=yaml.safe_load(Path(src).read_text())
m,r=base["model"],base["run"]
assert int(m["gcn_layer"])==31 and float(m["gnn_scale"])==0.1
assert m["train_gnn_with_rec_loss"] is True and m["freeze_rec"] is False
assert m["gnn_semantic_align"] is True and m["lora_gnn"]=="ngcf"
assert r["batch_size_train"]==3 and r["accum_grad_iters"]==3
assert r["max_epoch"]==3 and r["seed"]==42 and r["amp"] is True
assert not m.get("ckpt")
cfg=copy.deepcopy(base)
cfg["run"]["output_dir"]=str(Path(out)/"FREEZE_GNN_output")
assert cfg["model"]==base["model"]
assert {k for k in cfg["run"] if cfg["run"][k]!=base["run"][k]}=={"output_dir"}
Path(dst).write_text(yaml.safe_dump(cfg,sort_keys=False))

# Freeze GNN weights without no_grad or detaching its inputs.
p=Path(wt)/"minigpt4/models/minigpt4rec_v2.py"
s=p.read_text()
old="""            self.num_users = rec_config.user_num
            self.num_items = rec_config.item_num

            if self.gnn_semantic_align"""
new="""            self.num_users = rec_config.user_num
            self.num_items = rec_config.item_num

            # S2-FGNN: NGCF weights frozen but differentiable through inputs.
            assert self.gcn is not None
            _gnn_total = sum(p.numel() for p in self.gcn.parameters())
            for _p in self.gcn.parameters():
                _p.requires_grad_(False)
            _gnn_train = sum(p.numel() for p in self.gcn.parameters() if p.requires_grad)
            _mf_train = sum(p.numel() for p in self.rec_encoder.parameters() if p.requires_grad)
            assert _gnn_total>0 and _gnn_train==0 and _mf_train>0
            print(f"[S2-FGNN-FREEZE] GNN trainable=0 total={_gnn_total}, MF trainable={_mf_train}, "
                  f"GNN-to-MF autograd retained",flush=True)

            if self.gnn_semantic_align"""
assert s.count(old)==1,"model patch anchor not unique"
p.write_text(s.replace(old,new))

# Store frozen GNN params in checkpoint so it can be evaluated independently.
rp=Path(wt)/"minigpt4/runners/runner_base.py"
s=rp.read_text()
old="""                if not param_grad_dic[k]:
                    # delete parameters that do not require gradient
                    del state_dict[k]"""
new="""                if not param_grad_dic[k] and not k.startswith("gcn."):
                    # S2-FGNN: persist frozen GNN weights for correct replay.
                    del state_dict[k]"""
assert s.count(old)==1,"save patch anchor not unique"
s=s.replace(old,new)
old="""                if self.evaluate_only:
                    break

                if self.config.run_cfg.distributed:
                    dist.barrier()"""
new="""                if self.evaluate_only:
                    break

                if cur_epoch == 0:
                    logging.info("[S2-FGNN-STOP] completed epoch0 Valid and checkpoint; max_epoch=3 unchanged")
                    break

                if self.config.run_cfg.distributed:
                    dist.barrier()"""
assert s.count(old)==1,"runner stop anchor not unique"
rp.write_text(s.replace(old,new))
print("[S2-FGNN-STATIC] historical S2-L31 identical config except output_dir; "
      "freeze GNN parameters, not computation; MF remains trainable",flush=True)
PY
git -C "$WT" diff -- minigpt4/models/minigpt4rec_v2.py minigpt4/runners/runner_base.py > "$OUT/only_freeze_and_stop.diff"
python -m py_compile "$WT/minigpt4/models/minigpt4rec_v2.py" "$WT/minigpt4/runners/runner_base.py"
echo "[S2-FGNN-START] $(date -Is) 1 epoch L31, MF trainable, GNN params frozen"
(
  cd "$WT"
  python -u train_collm_mf_din.py --cfg-path "$CFG" 2>&1 | tee "$LOG"
)
grep -Fq "[S2-FGNN-FREEZE] GNN trainable=0" "$LOG" || { echo "[S2-FGNN-ERROR] freeze not active"; exit 3; }
grep -Fq "[S2-FGNN-STOP] completed epoch0 Valid" "$LOG" || { echo "[S2-FGNN-ERROR] epoch0 incomplete"; exit 3; }
python - "$LOG" "$OUT" <<'PY'
from pathlib import Path
import re,sys,torch
log,out=sys.argv[1:]
rows=[s for s in Path(log).read_text(errors="replace").splitlines()
      if "Averaged stats:" in s and "***auc:" in s and "***uauc:" in s]
assert len(rows)==1,f"Expected one Valid summary, got {len(rows)}"
def v(p):
    m=re.search(p,rows[0]);assert m,p
    return float(m.group(1))
auc=v(r"\*\*\*auc:\s*([0-9.]+)")
uauc=v(r"\*\*\*uauc:\s*\(?([0-9.]+)")
loss=v(r"\bloss:\s*([0-9.]+)")
ckpts=list((Path(out)/"FREEZE_GNN_output").rglob("checkpoint_best.pth"))
assert len(ckpts)==1,ckpts
try:ck=torch.load(ckpts[0],map_location="cpu",mmap=True,weights_only=False)
except TypeError:ck=torch.load(ckpts[0],map_location="cpu",weights_only=False)
assert int(ck["epoch"])==0
gkeys=[x for x in ck["model"] if x.startswith("gcn.")]
mkeys=[x for x in ck["model"] if x.startswith("rec_encoder.")]
assert len(gkeys)==12,(len(gkeys),gkeys)
assert len(mkeys)==2,(len(mkeys),mkeys)
print(f"[S2-FGNN-METRIC] epoch0 Valid AUC={auc:.9f} UAUC={uauc:.9f} ValidLoss={loss:.9f}",flush=True)
print(f"[S2-FGNN-CKPT] {ckpts[0]} frozen_gnn_keys={len(gkeys)} trainable_MF_keys={len(mkeys)}",flush=True)
print("[S2-FGNN-ANCHOR] S2-L31=0.729675260; Freeze-MF=0.635272859; DETACH=0.622691597",flush=True)
print("[S2-FGNN-COMPLETE] 1 epoch Valid only; real GNN signal, MF trainable, GNN fixed",flush=True)
PY
