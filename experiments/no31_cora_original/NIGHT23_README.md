# NIGHT23: LightGCN hierarchical collaborative conditions for CoRA

Branch: `exp/night23-lightgcn-hierarchical-cora-20261028`
Parent: `exp/night20-prototype-qformer-20261027`

## Why the old NGCF is not used

The old NGCF in `minigpt4rec_v2.py` is a three-layer graph feature
projector aimed at the legacy single-layer LoRA injection (now `gnn_scale=0`).
It does not expose graph-community features or per-candidate within-community
history. It also forms dense diagonal matrices in its normalization path.
We leave that path unchanged and use a separately trained sparse LightGCN
graph encoder for new **CoRA conditions**, not direct LLM hidden-state injection.

## What this implementation actually does

1. **TRAIN only**: Construct positive user-item edges from
   `train_ood2.pkl`. Random-ID initialized LightGCN with 2 propagation
   layers is trained by BPR using TRAIN-positive edges and random non-positive
   samples (20 epochs, configurable, seed 42). Neither MF embeddings nor
   Valid/Test labels initialize this graph model.
2. Spectrally separate (by KMeans on normalized, propagated graph item
   embeddings) 16 graph-derived communities, compute community means
   `p_k`, and residuals `d_i = e_i - p_{g(i)}`.
3. At CoRA initialization load frozen graph user/item tables and these
   community artifacts. Global-RMS-scale graph user/items to match the
   existing frozen MF condition magnitude, without replacing MF elsewhere.
4. **Graph coarse**: [graph user, graph target community prototype,
   graph coarse dot score, leave-target-out smoothed user/community
   preference]; **Graph fine**: [target graph residual, within-community
   positive historical residual, graph residual dot score].
5. Replace the baseline **CoRA QFormer [user;target] input** with graph
   [user;target]. Add corresponding coarse/fine query-state residual heads
   **before the existing 8 group projectors and dynamic-A head**. Their
   heads are zero-initialized and individually capped at 10% of the base
   query-state norm. Rank16/alpha48/full QKVO L0-31 stays unchanged.
6. Original MF-based prompt, CTR/KD/semantic/projector-regularization
   pathways are retained. The old layer31 GNN injection stays disabled.

**Graph edge caveat:** LightGCN train-only positive edges include train
target edges. This is standard transductive training graph pretraining;
leave-target-out protection applies to derived user/community history counts,
not to graph node embedding propagation. Do not claim edge-level
leave-target-out for LightGCN without additional graph rebuilding.

The artifact's user/item indices MUST match the model's original
`rec_encoder` indices. The pretrained MF checkpoint is read for table
SHAPES only during graph preprocessing; actual graph embeddings are learned
from TRAIN interactions.

## Server commands

Use a new isolated worktree, no background training, GPU0 only:

```bash
cd /home/bbx/user_lx/GraphLoRA_NO31_CORA
git fetch https://github.com/NanDingEi/GraphLoRA-Research.git exp/night23-lightgcn-hierarchical-cora-20261028
git worktree add --detach /home/bbx/user_lx/GraphLoRA_NIGHT23 FETCH_HEAD
cd /home/bbx/user_lx/GraphLoRA_NIGHT23
conda activate minigpt4

mkdir -p logs/NO31_CORA_ORIGINAL_READY
cp /home/bbx/user_lx/GraphLoRA_NO31_CORA/logs/NO31_CORA_ORIGINAL_READY/train.yaml logs/NO31_CORA_ORIGINAL_READY/train.yaml

python experiments/no31_cora_original/test_night23_cpu.py

CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/graph_cf_pretrain.py \
  --train-file /home/bbx/user_lx/GraphLoRA/dataset/ml-1m/train_ood2.pkl \
  --shape-checkpoint /home/bbx/user_lx/GraphLoRA/checkpoints/0912_ml1m_oodv2_best_model_d256lr-0.001wd0.0001.pth \
  --output logs/NO31_CORA_GRAPH_PRETRAIN/lightgcn_k16_seed42.npz \
  --device cuda --seed 42 --dim 256 --layers 2 --epochs 20 --clusters 16

python experiments/no31_cora_original/night23.py --plan \
  --graph-artifact logs/NO31_CORA_GRAPH_PRETRAIN/lightgcn_k16_seed42.npz

CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/night23.py --run \
  --graph-artifact logs/NO31_CORA_GRAPH_PRETRAIN/lightgcn_k16_seed42.npz
```

**Do not run N23 while GPU0 is busy.** For first inspection use
`--only G1_LIGHTGCN_BASE`, then the other arms sequentially. Graph
pretraining saves a manifest with TRAIN file SHA256 and K16 sizes. The
runner validates the prepared YAML using `night8.prepared_config`,
ensuring CTR=.05, KD=.05, seed42, original MF checkpoint, batch2/accum4
and 3-epoch LR horizon. Each CoRA arm trains 1 epoch, Valid only.

## Experimental arms

| Arm | Base QFormer condition | Extra graph query condition | Schedule |
| --- | --- | --- | --- |
| G1_LIGHTGCN_BASE | graph [user;target] | none | all groups |
| G2_GRAPH_COARSE | graph [user;target] | coarse only | all groups |
| G3_GRAPH_FINE | graph [user;target] | fine only | all groups |
| G4_GRAPH_DUAL_UNIFORM | graph [user;target] | both | 0.5 / 0.5 |
| G5_GRAPH_DUAL_C2F | graph [user;target] | both | coarse [1,1,.75,.75,.25,.25,0,0], fine complement |

Reference: historical NIGHT17 group8 E0 Valid AUC=0.729928474,
UAUC=0.686612232; NIGHT20 MF-prototype four-arm results must be
obtained from actual GPU logs, not guessed.

N23 changes (i) the baseline CoRA condition from MF to graph and
(ii) optionally adds hierarchical graph features. G1 isolates the first
change. G2/G3/G4/G5 isolate additional graph features relative to G1.
One epoch alone does not establish a final model superiority claim.

**Current status:** Source committed only. CPU preflight and GPU trials
must be run on the user's server. No N23 Valid/Test metrics exist yet.
