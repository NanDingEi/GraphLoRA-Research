#!/usr/bin/env bash
# S2 / ML-1M: EXACTLY 8 epoch budgets in 6 sequential experimental arms.
# No collaborative injection at layer31: model.gnn_scale=0 in ALL arms.
# Alternate mechanisms: GNN CTR supervision into shared MF, GNN->ID-input
# embedding fusion, and LLM->GNN prediction distillation.
# Original S2 source; isolated worktree; ONLY first-party torch/pyyaml deps.
# Train fresh from same seed42/config on each arm; 1/2 complete epochs;
# max_epoch=3 always preserved to keep the original cosine schedule.
# Never Test; all Valid. Do NOT overwrite S2 historical checkpoints.
set -euo pipefail

MAIN="/home/bbx/user_lx/GraphLoRA"
S2_SHA="d818fa2943ede4e9a810b032a3422e94d0c77ed0"
S2_ID="${S2_RUN_ID:-20261017_143611}"
SOURCE="$MAIN/logs/S2_STRICT_${S2_ID}/L31.yaml"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONUNBUFFERED=1
cd "$MAIN"
mkdir -p logs
exec 9>logs/S2_NO31_NIGHT8.lock
flock -n 9 || { echo "[NIGHT8-ERROR] another run already owns night8 lock"; exit 2; }
test -f "$SOURCE" || { echo "[NIGHT8-ERROR] Original S2 L31.yaml not found"; exit 2; }
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$MAIN/logs/S2_NO31_NIGHT8_${STAMP}"
WT="/home/bbx/user_lx/GraphLoRA_NO31_NIGHT8_${STAMP}"
mkdir -p "$OUT"
echo "[NIGHT8-ROOT] $OUT"
echo "[NIGHT8-SOURCE] $S2_SHA"
git cat-file -e "$S2_SHA^{commit}"
git worktree add --detach "$WT" "$S2_SHA"
test "$(git -C "$WT" rev-parse HEAD)" = "$S2_SHA"
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

# Patch copied HISTORICAL S2 source ONCE. Original S2 worktree is not touched.
python - "$WT" "$OUT" <<'PY'
from pathlib import Path
import sys
wt=Path(sys.argv[1]);out=Path(sys.argv[2])
p=wt/"minigpt4/models/minigpt4rec_v2.py"
s=p.read_text()
def sub(old,new,n=1):
    global s
    assert s.count(old)==n, ("source anchor count",old[:100],s.count(old))
    s=s.replace(old,new)

# Passive env-dependent run controls, no new learnable heads except the
# existing gnn_conf_kappa when its loss is enabled in the arm's YAML.
sub(
"""        self.train_gnn_with_rec_loss = bool(train_gnn_with_rec_loss)
""",
"""        self.train_gnn_with_rec_loss = bool(train_gnn_with_rec_loss)
        # NIGHT8: separate GNN supervision and input fusion, never layer31.
        self.s2_input_alpha = float(os.environ.get("S2_INPUT_ALPHA","0"))
        self.s2_small_kd = float(os.environ.get("S2_SMALL_KD","0"))
        assert 0.0 <= self.s2_input_alpha <= 1.0
        assert self.s2_small_kd >= 0.0
        assert float(gnn_scale) == 0.0, "Night8 STRICTLY forbids layer31 injection"
        assert lora_gnn == "ngcf"
        print(f"[NIGHT8-MODEL] gnn_scale=0 alpha={self.s2_input_alpha} "
              f"bce={float(gnn_conf_loss_weight)} kd={self.s2_small_kd}",flush=True)
""")
sub(
"""        need_gnn_view = need_semantic or need_pref \\
            or self.rel_inject or self.rel_rank_mod
""",
"""        need_gnn_view = need_semantic or need_pref \\
            or self.rel_inject or self.rel_rank_mod \\
            or self.s2_input_alpha > 0 or self.gnn_conf_loss_weight > 0 \\
            or self.s2_small_kd > 0
""")
# Mix GNN penultimate 256-D with MF 256-D before projector, at ID-token
# positions. Unlike LoRA bottleneck injection, this gives L_rec->GNN->MF
# gradients THROUGH INPUT EMBEDDING, and works identically at inference.
marker="""    @projector_regularized_forward
    def forward_v2(self, samples):"""
add="""    def _s2_input_fusion(self, inputs_embeds, samples_encode, fb):
        alpha = self.s2_input_alpha
        if alpha == 0.0:
            return inputs_embeds
        assert "token_gnn" in fb, "Need 256-d graph penultimate embeddings"
        node_ids = samples_encode["node_ids"]
        local = fb["local_node_ids"]
        valid = local >= 0
        assert local.shape[0] == node_ids.shape[0], "node-ID order misaligned"
        assert bool(valid.any()), "No graph nodes matched the ID tokens"
        batch = fb["batch_indices"][valid]
        pos = fb["token_positions"][valid]
        assert batch.numel() == torch.unique(torch.stack([batch,pos],dim=1),dim=0).shape[0]
        mf_embeddings = samples_encode["mf_embeddings"].index_select(0,node_ids[valid]).float()
        g_emb = fb["token_gnn"][valid].float()
        assert mf_embeddings.shape == g_emb.shape, (mf_embeddings.shape,g_emb.shape)
        scaled_graph = F.normalize(g_emb,dim=-1,eps=1e-8) * mf_embeddings.norm(dim=-1,keepdim=True)
        corrected_mf = (1.0-alpha)*mf_embeddings + alpha*scaled_graph
        projector_dtype = next(self.llama_proj.parameters()).dtype
        corrected_token = self.llama_proj(corrected_mf.to(dtype=projector_dtype))
        assert corrected_token.numel() == batch.numel()*inputs_embeds.shape[-1], (
            "Input fusion requires proj_token_num=1",corrected_token.shape)
        corrected_token = corrected_token.reshape(-1,inputs_embeds.shape[-1])
        out = inputs_embeds.clone()
        out[batch,pos] = corrected_token.to(dtype=out.dtype)
        if not getattr(self,"_s2_fusion_logged",False):
            self._s2_fusion_logged=True
            print(f"[NIGHT8-FUSION] alpha={alpha} ID_positions={batch.numel()} "
                  f"MF_norm={mf_embeddings.norm(dim=-1).mean().item():.6g} "
                  f"graph_norm={g_emb.norm(dim=-1).mean().item():.6g} "
                  f"corrected_norm={corrected_mf.norm(dim=-1).mean().item():.6g}",
                  flush=True)
        return out

"""+marker
sub(marker,add)
# Training: call AFTER compute_feedback_gcn (so sem teacher sees unchanged
# MF prompt), BEFORE the LLM forward. GNN bottleneck still has scale zero.
sub(
"""            gcn_output = fb['gcn_output']
            # Keep the original train-path""",
"""            inputs_embeds = self._s2_input_fusion(inputs_embeds, samples_encode, fb)
            gcn_output = fb['gcn_output']
            # Keep the original train-path""")
# Evaluation: same operation after constructing the graph features.
sub(
"""                gcn_output = fb['gcn_output'].to(dtype=torch.float16)
                batch_indices = fb['batch_indices']""",
"""                inputs_embeds = self._s2_input_fusion(inputs_embeds, samples_encode, fb)
                gcn_output = fb['gcn_output'].to(dtype=torch.float16)
                batch_indices = fb['batch_indices']""")
# The original GNN-CTR helper scores the penultimate 256-d features.
# For tonight, supervise the FINAL 8-d GNN output so all three NGCF layers
# receive recommendation-task supervision even with layer31 injection OFF.
sub(
"""    def _row_gnn_scores(self, fb):""",
"""    def _s2_final_gnn_scores(self, fb):
        local = fb["local_node_ids"]
        valid = local >= 0
        assert bool(valid.any())
        z = fb["gcn_output"].float()
        token_final = torch.zeros((local.shape[0],z.shape[-1]),
                                  device=z.device,dtype=z.dtype)
        token_final[valid] = z[local[valid]]
        score_fb = dict(fb)
        score_fb["token_gnn"] = token_final
        return self._row_gnn_scores(score_fb)

    def _row_gnn_scores(self, fb):""")

# The existing GNN supervised BCE is otherwise only executed inside the
# pref_distill+'cand_group' branch, which DOES NOT run for plain CTR batches.
# This explicit task loss is independent of cand_group and sends gradients
# into the GNN and shared MF. Optional soft-LLM teacher trains the small model,
# NOT the LLM; logits.detach() is critical.
sub(
"""        if self.training and self.pref_distill and 'cand_group' in samples:""",
"""        if self.training and not self.pref_distill and (
            self.gnn_conf_loss_weight > 0 or self.s2_small_kd > 0
        ):
            assert "token_gnn" in fb
            g_scores = self._s2_final_gnn_scores(fb).float()
            labels = samples["label"].float()
            assert g_scores.shape == labels.shape, (g_scores.shape,labels.shape)
            if self.gnn_conf_loss_weight > 0:
                loss_g = F.binary_cross_entropy_with_logits(g_scores,labels)
                loss = loss + self.gnn_conf_loss_weight*loss_g
                ret["loss"]=loss
                ret["loss_gctr"]=loss_g.detach()
            if self.s2_small_kd > 0:
                # Distil LLM preferences into GNN, without backprop into LLM
                # from this auxiliary loss (but normal L_rec still trains LLM).
                with torch.no_grad():
                    target = torch.sigmoid(logits.float().detach())
                loss_kd = F.binary_cross_entropy_with_logits(g_scores,target)
                loss = loss + self.s2_small_kd*loss_kd
                ret["loss"]=loss
                ret["loss_gkd"]=loss_kd.detach()
            if not hasattr(self,"_s2_loss_step"):
                self._s2_loss_step=0
            self._s2_loss_step+=1
            if self._s2_loss_step==1 or self._s2_loss_step%500==0:
                print(f"[NIGHT8-LOSS] step={self._s2_loss_step} "
                      f"L_rec={loss_rec.detach().item():.6g} "
                      f"L_gctr={ret.get('loss_gctr',torch.tensor(0.)).item():.6g} "
                      f"L_gkd={ret.get('loss_gkd',torch.tensor(0.)).item():.6g}",
                      flush=True)

        if self.training and self.pref_distill and 'cand_group' in samples:""")
p.write_text(s)

# After evaluation/checkpoint, stop after 1 or 2 full epochs. Keep max_epoch=3
# so the baseline's warmup-cosine schedule remains UNCHANGED.
rp=wt/"minigpt4/runners/runner_base.py"
s=rp.read_text()
old="""                if self.evaluate_only:
                    break

                if self.config.run_cfg.distributed:
                    dist.barrier()"""
new="""                if self.evaluate_only:
                    break

                _night_epochs=int(__import__("os").environ["S2_NIGHT_EPOCHS"])
                assert _night_epochs in (1,2)
                if cur_epoch+1 >= _night_epochs:
                    logging.info("[NIGHT8-STOP] trained_epochs=%d max_epoch=%d "
                                 "Valid and checkpoint completed",cur_epoch+1,self.max_epoch)
                    break

                if self.config.run_cfg.distributed:
                    dist.barrier()"""
assert s.count(old)==1,("runner anchor",s.count(old))
rp.write_text(s.replace(old,new))
print("[NIGHT8-PATCH] ID-input GNN fusion + supervised small CTR/KD; "
      "NO layer31 GNN injection; per-arm 1/2 epochs, schedule max_epoch=3",flush=True)
PY
git -C "$WT" diff -- minigpt4/models/minigpt4rec_v2.py minigpt4/runners/runner_base.py > "$OUT/source_patch.diff"
python -m py_compile "$WT/minigpt4/models/minigpt4rec_v2.py" "$WT/minigpt4/runners/runner_base.py"

# The final two arms have two full epochs EACH, not resume from another arm.
# name epochs input-fusion-alpha GNN-hard-label-BCE-weight LLM->GNN-soft-target-weight
ARMS=(
  "E1_GNN_CTR_005 1 0.00 0.05 0.00"
  "E2_GNN_CTR_020 1 0.00 0.20 0.00"
  "E3_INPUT_010 1 0.10 0.00 0.00"
  "E4_INPUT_CTR_010 1 0.10 0.05 0.00"
  "E5_INPUT_CTR_025 2 0.25 0.05 0.00"
  "E6_INPUT_CTR_KD 2 0.10 0.05 0.05"
)
printf 'experiment\tplanned_epochs\tcompleted_epochs\tbest_epoch\tbest_auc\tbest_uauc\tbest_valid_loss\tcheckpoint\n' > "$OUT/summary.tsv"
echo "[NIGHT8-PLAN] six arms, 1+1+1+1+2+2 = eight epoch budget; serial one GPU"
echo "[NIGHT8-BASELINE] existing no-L31 S2 epoch0 Valid AUC=0.633358 UAUC=0.591457"
for ITEM in "${ARMS[@]}"; do
  read -r NAME EPOCHS ALPHA BCE KD <<< "$ITEM"
  ARMOUT="$OUT/$NAME"
  mkdir -p "$ARMOUT"
  CFG="$ARMOUT/recipe.yaml"
  LOG="$ARMOUT/train.log"
  python - "$SOURCE" "$CFG" "$ARMOUT" "$NAME" "$EPOCHS" "$ALPHA" "$BCE" "$KD" <<'PY'
from pathlib import Path
import yaml,copy,sys
source,dest,armout,name,epochs,alpha,bce,kd=sys.argv[1:]
base=yaml.safe_load(Path(source).read_text())
m,r=base["model"],base["run"]
assert m["gcn_layer"]==31 and float(m["gnn_scale"])==0.1
assert m["lora_gnn"]=="ngcf" and m["gnn_semantic_align"] is True
assert m["freeze_rec"] is False and m["train_gnn_with_rec_loss"] is True
assert not m.get("pref_distill",False)
assert not m.get("ckpt")
assert int(m["proj_token_num"])==1
assert r["max_epoch"]==3 and r["seed"]==42
assert r["batch_size_train"]==3 and r["accum_grad_iters"]==3
assert r["amp"] is True
assert int(epochs) in (1,2)
cfg=copy.deepcopy(base)
cfg["model"]["gnn_scale"]=0.0
cfg["model"]["gnn_conf_loss_weight"]=float(bce)
cfg["run"]["output_dir"]=str(Path(armout)/"output")
# No extra restarts, no Test, preserve original max_epoch for LR consistency.
assert {k for k in cfg["model"] if m.get(k)!=cfg["model"].get(k)} <= {"gnn_scale","gnn_conf_loss_weight"}
assert {k for k in cfg["run"] if r.get(k)!=cfg["run"].get(k)} == {"output_dir"}
Path(dest).write_text(yaml.safe_dump(cfg,sort_keys=False))
print(f"[NIGHT8-CONFIG] {name} epochs={epochs} gnn_scale=0 "
      f"alpha={alpha} gnn_BCE_weight={bce} KD_weight={kd} "
      f"unchanged_max_epoch={r['max_epoch']} seed={r['seed']}",flush=True)
PY
  echo "[NIGHT8-START] $NAME epochs=$EPOCHS alpha=$ALPHA bce=$BCE kd=$KD $(date -Is)"
  (
    cd "$WT"
    export S2_NIGHT_EPOCHS="$EPOCHS"
    export S2_INPUT_ALPHA="$ALPHA"
    export S2_SMALL_KD="$KD"
    python -u train_collm_mf_din.py --cfg-path "$CFG" > "$LOG" 2>&1
  ) || {
    echo "[NIGHT8-ERROR] $NAME crashed; see $LOG"
    tail -n 75 "$LOG" || true
    exit 3
  }
  grep -Fq "[NIGHT8-MODEL] gnn_scale=0" "$LOG" || {
    echo "[NIGHT8-ERROR] $NAME model gnn_scale was NOT 0";exit 3;
  }
  grep -Fq "[NIGHT8-STOP] trained_epochs=$EPOCHS" "$LOG" || {
    echo "[NIGHT8-ERROR] $NAME did not finish expected $EPOCHS epochs";exit 3;
  }
  if [[ "$ALPHA" != "0.00" ]]; then
    grep -Fq "[NIGHT8-FUSION]" "$LOG" || {
      echo "[NIGHT8-ERROR] $NAME ID-input fusion was not applied";exit 3;
    }
  fi
  if [[ "$BCE" != "0.00" ]]; then
    grep -Fq "[NIGHT8-LOSS]" "$LOG" || {
      echo "[NIGHT8-ERROR] $NAME GNN supervised loss was not applied";exit 3;
    }
  fi
  python - "$LOG" "$OUT/summary.tsv" "$NAME" "$EPOCHS" "$ARMOUT" <<'PY'
import re,sys
from pathlib import Path
log,summary,name,ne,armdir=sys.argv[1:]
ne=int(ne)
lines=Path(log).read_text(errors="replace").splitlines()
results=[x for x in lines if "Averaged stats:" in x and "***auc:" in x and "***uauc:" in x]
assert len(results)==ne,(name,"Valid summaries",len(results),"expected",ne)
def v(s,p):
    m=re.search(p,s)
    assert m is not None,(name,p,s[:250])
    return float(m.group(1))
auc=[v(x,r"\*\*\*auc:\s*([0-9.]+)") for x in results]
uauc=[v(x,r"\*\*\*uauc:\s*\(?([0-9.]+)") for x in results]
loss=[v(x,r"\bloss:\s*([0-9.]+)") for x in results]
best=max(range(ne),key=lambda k:auc[k])
ckpts=list((Path(armdir)/"output").rglob("checkpoint_best.pth"))
assert len(ckpts)==1,(name,ckpts)
for k in range(ne):
    print(f"[NIGHT8-METRIC] {name} epoch={k} ValidAUC={auc[k]:.9f} "
          f"ValidUAUC={uauc[k]:.9f} ValidLoss={loss[k]:.9f}",flush=True)
with open(summary,"a") as f:
    f.write(f"{name}\t{ne}\t{len(results)}\t{best}\t{auc[best]:.9f}\t"
            f"{uauc[best]:.9f}\t{loss[best]:.9f}\t{ckpts[0]}\n")
print(f"[NIGHT8-ARM-DONE] {name} best_epoch={best} best_auc={auc[best]:.9f}",flush=True)
PY
done
echo "[NIGHT8-COMPLETE] 6 serial jobs, 8 epochs, 0 layer31 injection, Valid only"
cat "$OUT/summary.tsv"
