# GraphLoRA NO31 + CTR/KD + CoRA

**Standalone GitHub branch:** `exp/no31-ctr-kd-cora-20261023`.

This is NOT built from the failed S2-CoRA nine/five-arm implementation. It is
built directly from the archived historical S2 commit
`d818fa2943ede4e9a810b032a3422e94d0c77ed0`, with the exact
CTR/KD/input-fusion model transformations originally embedded in
[run_S2_NO31_night8_serial.sh](https://github.com/NanDingEi/GraphLoRA-Research/blob/exp/layer-injection-study-20261016/run_S2_NO31_night8_serial.sh)
applied and committed to `minigpt4/models/minigpt4rec_v2.py`. The runner's
original post-epoch stop was also committed natively. **No historical
`source_patch.diff` is needed, and the branch contains the actual trainable
model source.**

## Historical result vs untested new method

Original validated *NO31 CTR/KD* seed-42 epoch-0 Valid:
AUC `0.719760366`, UAUC `0.685106415` (recorded in historical R5/E8 runner).
Historical preselected checkpoint Test AUCs: R5 two epochs epoch 1
`0.73393590`, E8 epoch 0 `0.73217842`; these are Test metrics,
NOT the Valid scores for a new CoRA model.

**No NO31+CoRA AUC/UAUC is claimed until you run this new branch.**

## Fixed controls

| Component | Setting |
| --- | --- |
| Foundation | pretrained Vicuna-7B |
| Recommender | original pretrained MF checkpoint, still trainable |
| NGCF | fresh model with original seed 42 |
| Original L31 GNN→LoRA injection | OFF, `gnn_scale=0` |
| Original GNN→input fusion | OFF, `S2_INPUT_ALPHA=0` |
| GNN hard-label CTR loss | ON, coefficient 0.05 |
| LLM→GNN soft-label KD loss | ON, coefficient 0.05; teacher logits detached |
| LLM→GNN semantic alignment | ON, coefficient 1e-4 |
| Projector regularization | ON, coefficient 5e-4 |
| S2 group LR scales | MF .5, NGCF .25, projector .5, LoRA 1.0 |
| Seed | 42, never changed |
| Optimizer | original batch 3, accumulate 3, 3-epoch cosine schedule |
| Checkpoints | no trained S2/R4/R5 checkpoint loaded |
| Validation | original valid split only, no Test |
| Training | 1 complete epoch by default, optionally 2 |

## CoRA implementation

`experiments/no31_cora/runtime.py` extends only L31 Q/V via an
**independent zero-initialized residual**. User/actual candidate-item features
are gathered from the original NO31 NGCF 256-D penultimate embeddings.
A collaborative query generator uses 4 learned queries, query self-attention,
user/item cross-attention and FFN to make separate 8-D Q/V rank factors.

1. The historical `MyLinear.forward` is executed without reimplementation.
2. At matched user/history/target *ID tokens only*:
   `result += 0.1 * B_c( s_ui * A(x_id) )`.
3. New `q_up` and `v_up` are **exactly zero** at initialization, so the
   initial extra prediction residual is zero.
4. NGCF input is `detach()` by default, so GNN optimization keeps its
   original CTR/KD/semantic pathways; regular MF→LLM gradients remain.
5. The historical `S2_INPUT_ALPHA` and `S2_SMALL_KD` controls are
   forced to `0` and `.05`; the original model rejects nonzero L31
   injection.

This is a controlled **CoRA-inspired** experiment, not a full implementation
of another paper's CoRA method.

## Download and run on server

```bash
git clone --single-branch --branch exp/no31-ctr-kd-cora-20261023 \
  https://github.com/NanDingEi/GraphLoRA-Research.git \
  /home/bbx/user_lx/GraphLoRA_NO31_CORA

cd /home/bbx/user_lx/GraphLoRA_NO31_CORA
conda activate minigpt4

# Syntax checks, CPU test, find historical E8 YAML, write matching configs.
# Does NOT train:
bash experiments/no31_cora/run.sh prepare 1

# Train CoRA on GPU0 in the FOREGROUND for 1 full epoch:
CUDA_VISIBLE_DEVICES=0 bash experiments/no31_cora/run.sh cora 1

# Or independently train historical NO31 CTR/KD control:
CUDA_VISIBLE_DEVICES=0 bash experiments/no31_cora/run.sh control 1

# Read AUC/UAUC without retraining:
bash experiments/no31_cora/run.sh summary
```

To train 2 epochs instead, prepare a **fresh checkout** and pass `2` to
both prepare and run. The script refuses to overwrite any previous run
directory.

### Exact E8 configuration preference

The preparation script **first** uses the actual archived server run config:

```text
/home/bbx/user_lx/GraphLoRA/logs/S2_NO31_NIGHT8_20261018_012408/E6_INPUT_CTR_KD/recipe.yaml
```

That run's original environment `alpha=.10` is overridden to `alpha=0`
for the successful E8 CTR/KD ablation, while the stored YAML's
`gnn_scale=0`, `CTR=.05` and the other historical settings are retained.
If unavailable, it next looks for the original S2 Strict `L31.yaml`,
then for the committed recovered S2 configuration; the preparation
script **warns** when it cannot use exact E8 YAML. You can also invoke
`python experiments/no31_cora/prepare.py --source-historical-yaml PATH`
for an explicit known YAML.

Model checkpoints and logs are saved under:

```text
output/NO31_CTR_KD_CORA_20261023/{CONTROL,CORA}/
logs/NO31_CTR_KD_CORA_20261023/
```

Everything is foreground and serial on CUDA_VISIBLE_DEVICES=0. It never
touches the original `GraphLoRA` or your S2/R4/R5 output directories.
