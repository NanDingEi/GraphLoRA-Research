# Collaborative-aware CoRA LoRA-MoE: six experiments

**Training script:** `cora_collab_moe6.py`  
**Read-only results:** `cora_collab_moe6_summary.py`  
**Branch:** `exp/cora-collab-moe6-20261104`.

## Scope and scientific controls

Six independent Valid-only CoRA experiments. Every trial preserves the
original frozen Vicuna-7B backbone and the original **D4_TOPK_RES025**
collaborative mixed inputs, **C6_KD_BOTH** loss (CTR=0.005,
KD=0.010), total LoRA rank16 on each layer Q/K/V/O. Existing rank16 dynamic
weight generator and P adapters are partitioned into four contiguous,
**nonoverlapping rank4 experts**, not duplicated.

For router probability `p_{sample, group, expert}`:

```
rank_expert_weight = 1 + 0.2 * (4 * p - 1)
```

All four experts remain active. Every router begins at zero output logits, so
each expert weight initially equals **1**, identical to the original CoRA
rank16 forward. Unlike previous hard Top1/Top2, no rank channel is dropped.

Only the **router information** varies across trials. User/item/sequence
features are read from the same MF embedding tables used by CoRA. Router
inputs to MF embeddings are detached (no direct router-to-MF gradient), but
the original CoRA and C6 loss-to-MF paths are unchanged.

| Trial | Context / router | Learned parameters |
|---|---|---|
| M1_GLOBAL | Global; no user/item context | Four scalar global expert logits |
| M2_USER_CLUSTER | Fixed train-only MF user K4 clusters | 4x4 cluster expert logits |
| M3_ITEM_CLUSTER | Fixed train-only MF item K4 clusters | 4x4 cluster expert logits |
| M4_PAIR | Normalized user MF + target item MF + elementwise interaction | Small MLP |
| M5_MULTI_INTEREST | Three learned queries attend to true `samples['sas_seq']` history, selected by target item | Three query vectors + small MLP |
| M6_LAYER_PAIR | Same pair context as M4, with four independent 4-expert distributions for transformer groups L1–8, L9–16, L17–24, L25–32 | Four-head output MLP |

M2/M3 clusters are computed from **positive targets and histories in
`train_ood2.pkl` only**, never from Valid/Test labels or IDs. Cold IDs map
to the nearest train centroid using their MF vector. M5 reads the same
`sas_seq` sequence that the official CoRA model reads; it raises on missing
history rather than silently substituting user-only routing.

### Fixed training

- Seed=42, only GPU0, six fresh independent processes, serial foreground.
- CoRA LR=3e-4, MF LR=3e-5.
- Batch size 2, gradient accumulation 4, 400 iterations/epoch.
- Validation AUC early stopping patience 10.
- Valid only, no Test; per-run trainable-only atomic checkpoint.
- Existing logs and checkpoints from RR, S and E experiments are untouched.
- 22,000 MiB free GPU0 preflight inherited from prior runs.
- Additional mean probability balance regularizer of 1e-3.
- Logs record router mean/std probabilities and entropy per 100 microsteps;
  unlike expert activation counts, these reveal true soft-route differences.

## Installation

```bash
cd /home/bbx/user_lx/CoRA_official
conda activate minigpt4
EXP=experiments/no31_cora_original
RAW=https://raw.githubusercontent.com/NanDingEi/GraphLoRA-Research/exp/cora-collab-moe6-20261104/experiments/no31_cora_original

# Make sure the parent runner includes the previous first-batch hasattr fix.
curl -fL "$RAW/cora_rank4_experts5.py" -o "$EXP/cora_rank4_experts5.py"
curl -fL "$RAW/cora_collab_moe6.py" -o "$EXP/cora_collab_moe6.py"
curl -fL "$RAW/cora_collab_moe6_summary.py" -o "$EXP/cora_collab_moe6_summary.py"
```

The pre-existing `cora_joint8.py`, `cora_combo8.py`,
`cora_rankgran7.py`, `cora_struct10.py`, `cora_latevote4.py` remain
unchanged in your local CoRA_official directory.

## Validate the CPU routes and generate configs

```bash
python "$EXP/cora_collab_moe6.py" --cpu-test
python "$EXP/cora_collab_moe6.py" --plan
```

These steps do not load the 7B LLM or run GPU training.

## Recommended one-epoch smoke run, separate protocol

```bash
python -u "$EXP/cora_collab_moe6.py" \
  --run --screen-epochs 1 \
  --protocol CORA_COLLAB_MOE6_SMOKE_20261104
```

This starts real CoRA model loading/training and costs GPU resources, but each
trial stops after one validation epoch. It cannot guarantee that longer-run
numerical behavior will be stable. Smoke outputs do not overlap the full run.

## Full six-trial serial run

Run only once GPU0 is free:

```bash
python -u "$EXP/cora_collab_moe6.py" --run
```

To run a subset, choose trial names not yet present in the protocol's log
directory:

```bash
python -u "$EXP/cora_collab_moe6.py" --run \
  --only M4_PAIR,M5_MULTI_INTEREST,M6_LAYER_PAIR
```

The launcher preserves each trial's log, config and checkpoint and refuses to
overwrite existing logs. It stops at the first worker failure.

## Results

```bash
python "$EXP/cora_collab_moe6.py" --summary
python "$EXP/cora_collab_moe6.py" --summary --debug
```

Reads both `***auc / ***uauc` training logs and optional `valid_log.txt`
records. It scans actual run output directories for `checkpoint_best.pth`
independently, avoiding the empty-summary issue in older experiment scripts.

Summary CSV:
`experiments/no31_cora_original/out/CORA_COLLAB_MOE6_20261104/collab_moe6_valid_summary.csv`.

Logs:
`experiments/no31_cora_original/logs/CORA_COLLAB_MOE6_20261104/M*.log`.

AUC/UAUC are **Valid-only**; UAUC must be paired with the best-AUC epoch.
Checkpoint existence/ZIP format does not prove model loadability. All six
experiments use fixed seed42; small improvements cannot be treated as
statistically significant without additional evaluation.
