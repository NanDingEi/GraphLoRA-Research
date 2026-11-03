# CoRA Split-LoRA Experts: Six Controlled Experiments

Implementation: `cora_split_experts6.py`.

This is a **new** experiment protocol; it does not continue or overwrite the
unfinished `CORA_RANK4_EXPERTS5_20261103_FIX1` run.

## Controlled variants

| Trial | Activation / routing | Why |
|---|---|---|
| S1_SPLIT_ALL | Four rank-4 experts all active | Identity control equal to original full-rank-16 CoRA |
| S2_FIXED_TOP2 | Fixed two rank-4 experts per user | Measure loss of rank-8 sparsity |
| S3_SIM_TOP2 | Nearest two among four fixed train-only MF user centroids | Test user-similarity-based expert sharing |
| S4_SOFT4 | Softmax user router over four active experts; each weight initialized to 1 | Test user-conditioned soft route without capacity loss |
| S5_WARMUP_TOP2 | Soft4 warmup progressing to straight-through hard Top-2 | Test stable transition into sparse user experts |
| S6_SHARED_C2F | L1–16 all4, L17–24 shared+2 personalized, L25–32 shared+1 personalized | Test shared-vs-personalized coarse-to-fine experts |

All trials retain original CoRA `rank=16` parameters, divided into four
nonoverlapping contiguous rank-4 slices of the same `P` and dynamic
`W_{ui}` matrices. No extra P adapters are duplicated. For learned routers,
the small new user-gating MLP adds a few parameters.

S4 starts with user softmax probabilities of 1/4, scaled by four, so every
rank-4 expert begins at weight exactly 1. S5/S6 initialize with the same
dense identity and transition over 800 **training** microsteps (400 per epoch).
Validation always measures the final hard architecture, so checkpoints do
not depend on persisting a warmup counter. Dense/shared masks have no new
LoRA parameters.

**Baseline reference:** The prior independent RR1 run has Valid
AUC=0.732387 and UAUC=0.701999. S1 repeats the dense baseline in the same
split-expert runner as the other five variants. RR1 is historical context,
not a checkpoint reused by S1.

## Fixed setup

- Source: official CoRA + the already-used C6/D4 runtime code.
- Input: D4_TOPK_RES025, `neighbor_mix`, `personal_share=0.25`.
- Loss: original Yes-logit BCE + CTR weight `0.005` + gated KD `0.01`.
- Fine-tune: CoRA LR `3e-4`, MF LR `3e-5`.
- Seed 42, single physical GPU0, batch2, accumulation4, 400 microsteps per epoch.
- Serial foreground subprocesses; Valid-only early-stop patience 10, Test disabled.
- Every run has independent initialization, config/log directory, and
  atomic trainable-only best checkpoint.
- The inherited GPU0 preflight requires `>=22000 MiB` free.
- If any trial fails, later trials are not started. Existing trial logs are
  never overwritten.

## Install in local CoRA_official

```bash
cd /home/bbx/user_lx/CoRA_official
conda activate minigpt4
EXP=experiments/no31_cora_original
BASE=https://raw.githubusercontent.com/NanDingEi/GraphLoRA-Research/exp/cora-split-experts6-20261104/experiments/no31_cora_original

# The updated parent script includes the first-batch hasattr fix.
curl -fL "$BASE/cora_rank4_experts5.py" -o "$EXP/cora_rank4_experts5.py"
curl -fL "$BASE/cora_split_experts6.py" -o "$EXP/cora_split_experts6.py"

python "$EXP/cora_split_experts6.py" --cpu-test
python "$EXP/cora_split_experts6.py" --plan
```

After both pass and GPU0 is idle, run the six experiments serially:

```bash
python -u experiments/no31_cora_original/cora_split_experts6.py --run
```

To run a subset (only when they have not already created logs):

```bash
python -u experiments/no31_cora_original/cora_split_experts6.py --run \
  --only S1_SPLIT_ALL,S4_SOFT4,S5_WARMUP_TOP2,S6_SHARED_C2F
```

To check results without new training/GPU work:

```bash
python experiments/no31_cora_original/cora_split_experts6.py --summary
```

The result CSV is saved at
`experiments/no31_cora_original/out/CORA_SPLIT_EXPERTS6_20261104/rank4_expert_valid_summary.csv`.
Incomplete trials may have no AUC/UAUC until the first validation, and that
does **not** establish an algorithmic failure.
