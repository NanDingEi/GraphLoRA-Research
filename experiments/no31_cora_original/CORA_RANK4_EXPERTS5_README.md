# CoRA Rank-4 Expert Partition: Tonight's 5 Experiments

**Implementation:** `cora_rank4_experts5.py` on branch `exp/cora-rank4-experts-20261103`.

## Core implementation

For each existing CoRA PLora rank-16 Q/K/V/O adapter and sample-specific rank-16
dynamic matrix, partition the rank dimension into four **nonoverlapping**
rank-4 experts. The original `lora_P.weight[:, :16]` and generated
`W_ui[..., :16]` are **not duplicated** or replaced. Multiplying rank
channels by a 16-dimensional mask selects four rank-4 column blocks, then
calls the same original CoRA PLora forward. Thus the original adapter/generator
parameter budget is maintained.

Use `sqrt(4/k)` when k experts are active, matching the expected variance
of dense rank-16 adaptation under independent equal-variance components.
The dense all-four mask of ones is identical to unsplit rank-16 mathematically
and is checked by the CPU test.

## 5 experiments

| Trial | Active expert slots | Router | Hypothesis |
|---|---|---|---|
| E1_HASH_TOP1 | 1 of 4, all layers | user ID mod 4 | Sparse-rank negative control |
| E2_SIM_TOP1 | 1 of 4, all layers | train-only MF user KMeans K4 | Similar users benefit from shared expert |
| E3_USER_TOP1 | 1 of 4, all layers | trainable user-only straight-through Top1 | Learned user personalization helps |
| E4_USER_TOP2 | 2 of 4, all layers | trainable user-only straight-through Top2 | More active rank reduces sparsity loss |
| E5_DEPTH_C2F | L1-16: 4; L17-24: 2; L25-32: 1 | user-only trainable TopK at middle/deep groups | Coarse-to-fine expert activation across depth |

The existing **RR1_L32_BASE** is the separate reference:
Valid AUC=0.732387, UAUC=0.701999. It uses the same D4 input and C6 losses.
It is **not** re-trained or asserted to be a checkpoint matched to these 5 trials.

**New-user replacement prototype:** E2 assigns train-seen users to four
KMeans-derived expert slots. A cold user uses their own CoRA-generated
user-conditioned rank projection in the **least-similar** expert slot. This is
*per-example, non-destructive slot substitution*; no shared P weights are
overwritten and no persistent online-expert eviction is performed. The log
includes `cold_users_window`. If no cold users exist in the data,
this new-user behavior cannot be evaluated by E2.

No new LoRA P matrices are instantiated. E3/E4/E5 add only a small user router.
The total **LoRA rank budget remains 16**, although total trainable parameters
are slightly larger in learned-router trials because of those router MLPs.

## Fixed training controls

- Same official CoRA + D4_TOPK_RES025 mixed neighbor inputs as RR1
- C6_KD_BOTH: `CTR=.005`, `KD=.010`, original Yes-logit BCE retained
- CoRA LR `3e-4`, MF LR `3e-5`, seed `42`
- Batch `2`, accumulation `4`, `400` iterations/epoch
- GPU0 only, serial foreground workers, patience `10`, Valid only
- Independent pretrained MF initialization and independent best checkpoints
- Run-time optimizer inclusion and first-backward gradient checks for routers
- GPU0 memory preflight needs at least `22000 MiB` free
- Training logs and config/output artifacts isolate each protocol and trial
- No original source, prompt, old log, or checkpoint overwrites

## Install the new script into existing CoRA_official

From `/home/bbx/user_lx/CoRA_official`:

```bash
cd /home/bbx/user_lx/CoRA_official
conda activate minigpt4
curl -fL \
  https://raw.githubusercontent.com/NanDingEi/GraphLoRA-Research/exp/cora-rank4-experts-20261103/experiments/no31_cora_original/cora_rank4_experts5.py \
  -o experiments/no31_cora_original/cora_rank4_experts5.py
```

This experiment depends on the existing `cora_joint8.py`,
`cora_combo8.py`, `cora_rankgran7.py`, `cora_struct10.py`, and
`cora_latevote4.py` in the same directory. These were already used by the
previous residual-routing experiment. This installer does **not** modify them.

## Basic validation

```bash
python experiments/no31_cora_original/cora_rank4_experts5.py --cpu-test
python experiments/no31_cora_original/cora_rank4_experts5.py --plan
```

Expected `[R4E CPU PASS]`, then `[R4E PLAN PASS]`.

## Full five-trial run (only after CPU and config preflight)

```bash
python -u experiments/no31_cora_original/cora_rank4_experts5.py --run
```

Or run only E3 and E5:

```bash
python -u experiments/no31_cora_original/cora_rank4_experts5.py \
  --run --only E3_USER_TOP1,E5_DEPTH_C2F
```

One worker at a time, stop on first failure. A trial with an existing log is
not overwritten. For an independent two-epoch smoke screen, use a **different
protocol** so the complete training logs remain available:

```bash
python -u experiments/no31_cora_original/cora_rank4_experts5.py \
  --run --screen-epochs 2 \
  --protocol CORA_RANK4_EXPERTS5_SMOKE_20261103
```

## Read results

```bash
python experiments/no31_cora_original/cora_rank4_experts5.py --summary
```

This reads `valid_log.txt` and writes a read-only best-AUC + paired-UAUC CSV
at `experiments/no31_cora_original/out/CORA_RANK4_EXPERTS5_20261103/rank4_expert_valid_summary.csv`.
It does not use GPU or run Test.

When investigating experts, search individual training logs for
`[R4E TRAIN]`, `expert_counts_window`, `[R4E OPT PASS]`, and
`[R4E GRAD PASS]` in
`experiments/no31_cora_original/logs/CORA_RANK4_EXPERTS5_20261103/`.
The expert-count window aggregates 100 microsteps to avoid interpreting
single batch=2 counts as a user-level trend.

## Fix 2026-11-03: Module attribute check

The first E1 run under `CORA_RANK4_EXPERTS5_20261103` failed at the
first training batch, because a model module's auxiliary tensor was checked
using container membership instead of an attribute check. The script on
this branch now uses `hasattr(generator, "_r4e_aux")`.

The prior failed log is retained. After re-downloading the script, use a
**new protocol** to re-run all five trials without overwriting that record:

```bash
python -u experiments/no31_cora_original/cora_rank4_experts5.py \
  --run --protocol CORA_RANK4_EXPERTS5_20261103_FIX1
```

The `--run` entry generates missing configs automatically. To view the
new results, append `--protocol CORA_RANK4_EXPERTS5_20261103_FIX1` to
`--summary`.
