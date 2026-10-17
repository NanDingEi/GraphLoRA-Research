#!/usr/bin/env bash
# Diagnostic 1: same S2 L31 checkpoint; paired per-example logits under gnn_scale 0/0.1/10.
# Source: historical S2 d818fa2 in its OWN temporary git worktree.
# No train or Test; no original checkpoint or worktree edits.
set -euo pipefail
MAIN="/home/bbx/user_lx/GraphLoRA"
SHA="d818fa2943ede4e9a810b032a3422e94d0c77ed0"
ID="${S2_RUN_ID:-20261017_143611}"
RUN="$MAIN/logs/S2_STRICT_${ID}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONUNBUFFERED=1
cd "$MAIN"
mkdir -p logs
exec 9>logs/S2_LOGIT_SENS.lock
flock -n 9 || { echo "[S2-LOGIT-ERROR] another sensitivity run is active"; exit 2; }
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$MAIN/logs/S2_LOGIT_SENS_${STAMP}"
WT="$MAIN/../GraphLoRA_S2_LOGIT_SENS_${STAMP}"
mkdir -p "$OUT"
test -f "$RUN/L31.yaml"
mapfile -t CKPTS < <(find "$RUN/L31_output" -maxdepth 3 -type f -name checkpoint_best.pth | sort)
[[ "${#CKPTS[@]}" -eq 1 ]] || { echo "[S2-LOGIT-ERROR] L31 epoch0 checkpoint not unique"; exit 2; }
CKPT="${CKPTS[0]}"
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
test -f "$WT/dataset/ml-1m/movie_graph_valid.npz"
# Instrument CURRENT active S2 evaluation (the only source change); no RNG consumed.
python - "$WT/minigpt4/tasks/rec_base_task.py" "$WT/minigpt4/models/minigpt4rec_v2.py" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]);s=p.read_text()
old="""                    users.extend(samples['UserID'].detach().cpu().numpy())
                    results_logits.extend(eval_output['logits'].detach().cpu().numpy())
                    labels.extend(samples['label'].detach().cpu().numpy())"""
new="""                    users.extend(samples['UserID'].detach().cpu().numpy())
                    results_logits.extend(eval_output['logits'].detach().cpu().numpy())
                    labels.extend(samples['label'].detach().cpu().numpy())
                    # S2 logit audit: write EXACT same logits used for AUC, before
                    # evaluation's in-place accuracy threshold. No RNG consumed.
                    import os as _s2_os
                    if _s2_os.environ.get('S2_SCORE_DUMP_PATH'):
                        _users = samples['UserID'].detach().cpu().reshape(-1).tolist()
                        _items = samples['TargetItemID'].detach().cpu().reshape(-1).tolist()
                        _labels = samples['label'].detach().cpu().reshape(-1).tolist()
                        _logits = eval_output['logits'].detach().cpu().reshape(-1).tolist()
                        assert len(_users)==len(_items)==len(_labels)==len(_logits)
                        with open(_s2_os.environ['S2_SCORE_DUMP_PATH'], 'a') as _f:
                            for _u,_i,_y,_z in zip(_users,_items,_labels,_logits):
                                _f.write(f'{int(_u)},{int(_i)},{int(_y)},{float(_z):.12g}\\n')
"""
assert s.count(old)==1, "S2 score-dump patch location not uniquely found"
p.write_text(s.replace(old,new))
q=Path(sys.argv[2]);t=q.read_text()
old2="""                    local_node_ids = self.local_node_ids[valid_mask]

                    if self.training and self.inj_diag:"""
new2="""                    local_node_ids = self.local_node_ids[valid_mask]

                    if not getattr(self, '_s2_signal_logged', False):
                        self._s2_signal_logged = True
                        with torch.no_grad():
                            _a = lora_a_output[batch_indices, token_positions].detach().float()
                            _z = self.gcn_output[local_node_ids].detach().float()
                            _a_norm = _a.norm(dim=-1).mean().item()
                            _z_norm = _z.norm(dim=-1).mean().item()
                            print(f'[S2-SIGNAL] layer={self.layer_idx} beta={self.gnn_scale:.5g} '
                                  f'active_tokens={len(local_node_ids)} '
                                  f'LoRA_A_mean_norm={_a_norm:.9g} '
                                  f'GNN_mean_norm={_z_norm:.9g} '
                                  f'betaGNN_over_A={self.gnn_scale*_z_norm/max(_a_norm,1e-12):.9g}',flush=True)

                    if self.training and self.inj_diag:"""
assert t.count(old2)==1, "S2 signal patch anchor not uniquely found"
q.write_text(t.replace(old2,new2))
print("[S2-LOGIT-PATCH] only eval logit dump and passive signal magnitude probe; no RNG change",flush=True)
PY
git -C "$WT" diff -- minigpt4/tasks/rec_base_task.py minigpt4/models/minigpt4rec_v2.py > "$OUT/instrumentation.diff"
python -m py_compile "$WT/minigpt4/tasks/rec_base_task.py" "$WT/minigpt4/models/minigpt4rec_v2.py"
for SCALE in 0 0.1 10; do
  SCORES="$OUT/gnn_${SCALE}.csv"
  LOG="$OUT/gnn_${SCALE}.log"
  : > "$SCORES"
  echo "[S2-LOGIT-START] epoch0 L31, same ckpt, scale=$SCALE"
  (
    cd "$WT"
    S2_SCORE_DUMP_PATH="$SCORES" \
    python -u train_collm_mf_din.py --cfg-path "$RUN/L31.yaml" --options \
      "model.ckpt=$CKPT" "model.gnn_scale=$SCALE" \
      "run.evaluate=true" "run.test_splits=[valid]" \
      "run.output_dir=$OUT/gnn_${SCALE}_eval" 2>&1 | tee "$LOG"
  )
  grep -Fq "[S2-SIGNAL] layer=31" "$LOG" || {
    echo "[S2-LOGIT-ERROR] L31 injection branch did not execute!";exit 3;
  }
done
python - "$OUT" <<'PY'
import csv,sys,numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
d=Path(sys.argv[1])
def read(scale):
    with (d/f"gnn_{scale}.csv").open() as f:
        arr=np.array([[float(x) for x in row] for row in csv.reader(f)],dtype=np.float64)
    assert arr.shape==(5200,4),f"Expected 5200 rows (5200,4), got {arr.shape}"
    assert np.isfinite(arr).all()
    return arr
a=read("0");base=read("0.1");big=read("10")
for tag,data in (("0",a),("0.1",base),("10",big)):
    assert np.array_equal(a[:,:3],data[:,:3]),"User/item/labels misaligned between runs"
    print(f"[S2-LOGIT-AUC] scale={tag} auc={roc_auc_score(data[:,2],data[:,3]):.9f}",flush=True)
for tag,data in (("0.1",base),("10",big)):
    delta=np.abs(data[:,3]-a[:,3])
    print(f"[S2-LOGIT-DELTA] scale={tag} vs0 n={len(delta)} "
          f"mean_abs={delta.mean():.12g} median_abs={np.median(delta):.12g} "
          f"p95={np.quantile(delta,.95):.12g} max={delta.max():.12g} "
          f"exactly_equal={np.count_nonzero(delta==0)}",flush=True)
print("[S2-LOGIT-COMPLETE] Valid only. No training. Checkpoint untouched.",flush=True)
PY
