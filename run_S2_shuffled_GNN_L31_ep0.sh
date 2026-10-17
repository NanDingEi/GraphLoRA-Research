#!/usr/bin/env bash
# Diagnostic 2: historical S2 L31, one epoch, true GNN-vectors vs MISALIGNED GNN-vectors.
# Only injection lookup changes: local_node_ids -> (local_node_ids+1) mod subgraph_size.
# No random permutation and no extra RNG consumption. Graph adjacency, GNN model,
# semantic alignment, MF, LoRA, schedule, optimizer and batch order remain unchanged.
# Existing strict S2-L31 epoch0 valid AUC ~0.729675 serves as real GNN anchor.
# NOTE: deterministic rotation can mix user/item types; this is a strong
# identity-destroying negative control, not matched-node-type shuffling.
set -euo pipefail
MAIN="/home/bbx/user_lx/GraphLoRA"
SHA="d818fa2943ede4e9a810b032a3422e94d0c77ed0"
ID="${S2_RUN_ID:-20261017_143611}"
RUN="$MAIN/logs/S2_STRICT_${ID}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONUNBUFFERED=1
cd "$MAIN"
mkdir -p logs
exec 9>logs/S2_SHUFFLED_GNN.lock
flock -n 9 || { echo "[S2-SHUFFLE-ERROR] shuffled train already running"; exit 2; }
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$MAIN/logs/S2_SHUFFLED_GNN_${STAMP}"
WT="$MAIN/../GraphLoRA_S2_SHUFFLE_${STAMP}"
mkdir -p "$OUT"
CFG="$OUT/L31_shuffled.yaml"
LOG="$OUT/L31_shuffled.log"
test -f "$RUN/L31.yaml"
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
python - "$RUN/L31.yaml" "$CFG" "$OUT" "$WT" <<'PY'
from pathlib import Path
import sys,copy,yaml
src,dst,out,wt=sys.argv[1:]
orig=yaml.safe_load(Path(src).read_text())
assert int(orig['model']['gcn_layer'])==31
assert float(orig['model']['gnn_scale'])==0.1
assert not orig['model'].get('ckpt')
assert orig['run']['max_epoch']==3
assert orig['run']['batch_size_train']==3
assert orig['run']['accum_grad_iters']==3
assert orig['run']['warmup_steps']==200
assert orig['run']['seed']==42
cfg=copy.deepcopy(orig)
cfg['run']['output_dir']=str(Path(out)/"shuffled_training_output")
assert cfg['model']==orig['model']
assert {k for k in cfg['run'] if cfg['run'][k]!=orig['run'][k]}=={'output_dir'}
Path(dst).write_text(yaml.safe_dump(cfg,sort_keys=False))

# Patch only MyLinear's 3 occurrences (both train/eval real injection branches
# plus passive diagnostic) to use wrong GNN node identity with same tensor norms.
p=Path(wt)/"minigpt4/models/minigpt4rec_v2.py"
s=p.read_text()
before,sep,after=s.partition('@registry.register_model("mini_gpt4rec_v2")')
assert sep
needle='self.gcn_output[local_node_ids]'
assert before.count(needle)==3,before.count(needle)
wrong='self.gcn_output[(local_node_ids + 1) % self.gcn_output.shape[0]]'
before=before.replace(needle,wrong)
# Add one-time runtime proof of the wrong-node mapping.
a="""                    local_node_ids = self.local_node_ids[valid_mask]

                    if self.training and self.inj_diag:"""
b="""                    local_node_ids = self.local_node_ids[valid_mask]
                    if not getattr(self, '_s2_shuffle_printed', False):
                        self._s2_shuffle_printed = True
                        print(f'[S2-SHUFFLE-APPLIED] layer={self.layer_idx} '
                              f'local_nodes={self.gcn_output.shape[0]} '
                              f'active_tokens={len(local_node_ids)} '
                              f'first_3_original={local_node_ids[:3].detach().cpu().tolist()} '
                              f'first_3_wrong={((local_node_ids[:3]+1)%self.gcn_output.shape[0]).detach().cpu().tolist()}',
                              flush=True)

                    if self.training and self.inj_diag:"""
assert before.count(a)==1
before=before.replace(a,b)
p.write_text(before+sep+after)
# Only control-flow change: after FULL epoch0 including Valid and checkpoint.
r=Path(wt)/"minigpt4/runners/runner_base.py"
t=r.read_text()
old="""                if self.evaluate_only:
                    break

                if self.config.run_cfg.distributed:
                    dist.barrier()
"""
new="""                if self.evaluate_only:
                    break

                # Controlled S2 one-epoch stop after completed Valid/checkpoint
                if cur_epoch == 0:
                    logging.info("[S2-SHUFFLE-STOP] after full epoch0, schedule max_epoch unchanged")
                    break

                if self.config.run_cfg.distributed:
                    dist.barrier()
"""
assert t.count(old)==1
r.write_text(t.replace(old,new))
print("[S2-SHUFFLE-STATIC] EXACT S2-L31 config; only output_dir changed. "
      "Only GNN node lookup rotated in MyLinear. No RNG calls.",flush=True)
PY
git -C "$WT" diff -- minigpt4/models/minigpt4rec_v2.py minigpt4/runners/runner_base.py > "$OUT/only_changes.diff"
python -m py_compile "$WT/minigpt4/models/minigpt4rec_v2.py" "$WT/minigpt4/runners/runner_base.py"
echo "[S2-SHUFFLE-START] $(date -Is) source=$SHA S2 L31 1 epoch wrong-node-injection"
(
 cd "$WT"
 python -u train_collm_mf_din.py --cfg-path "$CFG" 2>&1 | tee "$LOG"
)
grep -Fq "[S2-SHUFFLE-APPLIED] layer=31" "$LOG" || {
  echo "[S2-SHUFFLE-ERROR] wrong-node lookup not exercised"; exit 3;
}
grep -Fq "[S2-SHUFFLE-STOP] after full epoch0" "$LOG" || {
  echo "[S2-SHUFFLE-ERROR] epoch0 not completed"; exit 3;
}
python - "$LOG" "$OUT" <<'PY'
import re,sys
from pathlib import Path
log,out=sys.argv[1:]
lines=Path(log).read_text(errors="replace").splitlines()
val=[x for x in lines if 'Averaged stats:' in x and '***auc:' in x and '***uauc:' in x]
assert len(val)==1,f"expected 1 Valid, got {len(val)}"
def extract(p):
    m=re.search(p,val[0])
    assert m is not None,p
    return float(m.group(1))
auc=extract(r"\*\*\*auc:\s*([0-9.]+)")
uauc=extract(r"\*\*\*uauc:\s*\(?([0-9.]+)")
loss=extract(r"\bloss:\s*([0-9.]+)")
checkpoint=list((Path(out)/"shuffled_training_output").rglob("checkpoint_best.pth"))
assert len(checkpoint)==1,checkpoint
print(f"[S2-SHUFFLE-METRIC] epoch0 Valid AUC={auc:.9f} UAUC={uauc:.9f} "
      f"ValidLoss={loss:.9f} checkpoint={checkpoint[0]}",flush=True)
print("[S2-SHUFFLE-ANCHORS] Real-GNN S2-L31 AUC=0.72967525963; No-GNN S2-L31 AUC=0.633358 (earlier run)",flush=True)
print("[S2-SHUFFLE-COMPLETE] S2 strict 1 epoch, no Test, original checkpoint unchanged",flush=True)
PY
