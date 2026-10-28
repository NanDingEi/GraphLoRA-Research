# NIGHT24 — joint LightGCN + hierarchical CoRA + Vicuna optimization

Branch: `exp/night24-joint-lightgcn-cora-20261028`, branching from NIGHT23.
This is a NEW experiment, not a modification of historical frozen-graph
logs/checkpoints.

## Design

NIGHT23 freezes the graph embeddings after independent BPR pretraining.  In
NIGHT24, the **LightGCN ID embedding parameters and sparse graph propagation
are live in each train forward**. Their gradients come from the CoRA-conditioned
LLM recommendation loss and also an optional small graph BCE term.

* Positive training edges construct the normalized user–item adjacency.
* BPR-pretrained **raw graph ID table** initializes the trainable parameter;
  the artifact must include `graph_id_table` (NIGHT24 pretrainer exports it).
* Each mini-batch recomputes two LightGCN propagation steps with autograd.
* KMeans K16 graph-community memberships are fixed from TRAIN; graph community
  prototypes are **recomputed differentiably** as means of CURRENT graph item
  embeddings. Item residuals `r_i=z_i-p_{g(i)}` and sparse historical
  user/community positive-residual aggregates are also live, not frozen NPZs.
* User/community counts and priors are computed from TRAIN alone, excluding
  records for the current user–target pair (as in NIGHT20). Target-positive
  edge may still exist in the fixed graph adjacency during training: this is
  **transductive CF**, not per-example edge-masked graph training.
* Graph [user;candidate] replaces the frozen graph CoRA-QFormer condition,
  with the coarse/fine paths injected before original Group8 projectors.
  Original Vicuna, 32-layer Q/K/V/O CoRA rank16 alpha48 and zero-initialized
  CoRA P remain trainable as before.
* Original MF rec encoder remains in the prompt/CTR/KD/semantic paths, and
  existing projector regularization, training seed and scheduling are unchanged.

Objective (existing losses unchanged, plus graph):
`L_total = L_NO31+CoRA + graph_aux_weight * BCE(dot(graph_user,graph_item),label)`.
Set `--graph-aux-weight 0` to isolate learning from the LLM/CoRA losses.

Optimizer: `graph_joint.id_embedding` is a separate `graph_wd` AdamW
parameter group with `lr_scale_graph=0.1` relative to the common schedule.
Both the train-forward state and optimizer group are audited. Runtime prints
`[N24-GRAPH-GRAD]` for its first three actual backward calls. The runner
fails if graph gradients do not arrive.

This is true joint optimization of **graph ID table / graph propagation /
QFormer / CoRA / LLM adapter**, not simply trainable projection heads over
frozen graph embeddings. The LightGCN propagation operator itself is fixed,
as LightGCN has no learned message-passing transformation weights; its node
representation parameters are trainable.

## Experimental arms

| Arm | Trainable graph base | Live coarse/fine extra state | CoRA group schedule |
| --- | --- | --- | --- |
| J1_JOINT_GRAPH_BASE | yes | none | unchanged |
| J2_JOINT_DUAL_UNIFORM | yes | both | coarse/fine 0.5/0.5 |
| J3_JOINT_DUAL_C2F | yes | both | coarse 1,1,.75,.75,.25,.25,0,0; fine complement |

Compare J1 to NIGHT23 G1 (frozen graph), J2 to NIGHT23 G4,
and J3 to NIGHT23 G5. Only compare experiments initialized from the SAME
K16 graph artifact and with the same training setup. All reported metrics
are **Valid AUC/UAUC**, NOT Test.

## Commands on server (GPU0 only, foreground, no background run)

```bash
cd /home/bbx/user_lx/GraphLoRA_NO31_CORA
git fetch https://github.com/NanDingEi/GraphLoRA-Research.git exp/night24-joint-lightgcn-cora-20261028
git worktree add --detach /home/bbx/user_lx/GraphLoRA_NIGHT24 FETCH_HEAD
cd /home/bbx/user_lx/GraphLoRA_NIGHT24
conda activate minigpt4

mkdir -p logs/NO31_CORA_ORIGINAL_READY
cp /home/bbx/user_lx/GraphLoRA_NO31_CORA/logs/NO31_CORA_ORIGINAL_READY/train.yaml \
   logs/NO31_CORA_ORIGINAL_READY/train.yaml

python experiments/no31_cora_original/test_night24_cpu.py

# Re-export graph artifact with RAW ID table. The older NIGHT23 artifact
# lacks graph_id_table and deliberately fails joint-mode preflight.
CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/graph_cf_pretrain.py \
  --train-file /home/bbx/user_lx/GraphLoRA/dataset/ml-1m/train_ood2.pkl \
  --shape-checkpoint /home/bbx/user_lx/GraphLoRA/checkpoints/0912_ml1m_oodv2_best_model_d256lr-0.001wd0.0001.pth \
  --output logs/NO31_CORA_GRAPH_JOINT/lightgcn_k16_raw_seed42.npz \
  --device cuda --seed 42 --dim 256 --layers 2 --clusters 16 --epochs 20

python experiments/no31_cora_original/night24.py --plan \
  --graph-artifact logs/NO31_CORA_GRAPH_JOINT/lightgcn_k16_raw_seed42.npz

# First train one joint arm and inspect graph gradient + Valid quality.
CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/night24.py \
  --only J1_JOINT_GRAPH_BASE \
  --graph-artifact logs/NO31_CORA_GRAPH_JOINT/lightgcn_k16_raw_seed42.npz

# Only after J1 validation, run the remaining two arms serially.
CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/night24.py \
  --only J2_JOINT_DUAL_UNIFORM \
  --graph-artifact logs/NO31_CORA_GRAPH_JOINT/lightgcn_k16_raw_seed42.npz

CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/night24.py \
  --only J3_JOINT_DUAL_C2F \
  --graph-artifact logs/NO31_CORA_GRAPH_JOINT/lightgcn_k16_raw_seed42.npz
```

If GPU0 is occupied, do not start a GPU command. The runner checks GPU0.
Prepared config expects `gnn_scale=0`, `CTR=.05`, `KD=.05`,
`gnn_semantic_align=True`, projector regularization intact, seed42,
`batch_size_train=2`, `accum_grad_iters=4`, and 3-epoch LR
schedule with one actual trained epoch per arm.

## Required checks

- `[N24-CPU-PASS]`
- `[CORA-N23-GRAPH]`, `[N24-GRAPH-JOINT]`, `[N24-GRAPH-PROBE-OK]`
- `[lr-group] graph_wd ... scale=0.1`
- `[N24-GRAPH-GRAD] ... norm > 0`
- `[NIGHT8-RESULT] ... Valid AUC/UAUC` (not yet available)
- One checkpoint_0 per arm and no Test evaluation

Potential limitations: the graph adjacency is transductive and
fixed (trained on positives only), community assignments do not change
differentiably, sparse graph propagation is repeated every training batch,
and one GPU0 3090 may OOM. Any of these may require smaller graph
feature dimension or graph caching/checkpointing in a follow-up, with fair
baseline reruns. Do NOT substitute detached graph features just to solve
OOM; doing so breaks the LLM->graph gradient path.

Status: **code committed, NOT run**. No CPU/GPU experiment has been executed
by the assistant, and no N24 AUC/UAUC can be claimed yet.
