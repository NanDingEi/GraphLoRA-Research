# NIGHT10: eight seed-42 CoRA experiments

Goal: improve Valid AUC/UAUC without changing random seeds, while separating
configuration sensitivity from two low-risk residual-branch interventions.
All arms are one fresh epoch, run serially on GPU0 in the foreground. They
use the same prepared NO31 configuration and identical seed 42; no Test split
is enabled. Existing R2/R7/F4 checkpoints and logs are never overwritten.

## Why this plan

- The current best is F4: full-depth QKVO, alpha=48, Valid AUC=0.730121895,
  UAUC=0.688041011.
- Resuming F4 for epoch 1 produced 0.726915968 / 0.687095218, so the immediate
  goal is not simply to add epochs; it is to find a stable first-epoch recipe.
- V/O alpha=32 reached 0.725721127 / 0.682916847, while V/O alpha=48 in the
  interrupted N4 run reached only 0.628508184 / 0.608556160. This is why G1
  rechecks R2 exactly instead of assuming the difference is normal.
- Alpha=52/56/60 refine the observed high-performing QKVO interval 48–64.
- G6 lowers only the CoRA-P learning-rate multiplier to 0.5.
- G7 caps the CoRA residual per token at 0.5 times the base projection-output
  norm. The cap is inactive when the residual is small.
- G8 keeps all 32 layers and QKVO active, but applies a smooth fixed profile:
  scale(layer)=0.5+0.5*layer/31. This is an opt-in model intervention; the
  original uniform CoRA path remains the default.

## Eight arms

| ID | Projections | Alpha | P LR multiplier | Residual cap | Depth profile | Purpose |
|---|---|---:|---:|---:|---|---|
| G1 | V/O | 32 | 1.0 | off | uniform | Exact R2 anchor; diagnose the N4 discrepancy |
| G2 | Q/K/V/O | 48 | 1.0 | off | uniform | Exact F4 anchor; confirm the current best recipe |
| G3 | Q/K/V/O | 52 | 1.0 | off | uniform | Refine alpha |
| G4 | Q/K/V/O | 56 | 1.0 | off | uniform | Refine alpha |
| G5 | Q/K/V/O | 60 | 1.0 | off | uniform | Refine alpha |
| G6 | Q/K/V/O | 48 | 0.5 | off | uniform | Reduce P-update magnitude |
| G7 | Q/K/V/O | 48 | 1.0 | 0.5 | uniform | Norm-bounded residual branch |
| G8 | Q/K/V/O | 48 | 1.0 | off | late_linear | Attenuate early-layer residuals smoothly |

Common settings: seed=42; train batch=2; eval batch=3; accumulation=4;
init LR=1e-4; max_epoch schedule horizon=3; one completed epoch; CTR=.05;
KD=.05; semantic correction weight=1e-4; projector regularization=5e-4;
autocast-cache fix ON; rank=16; MF/NGCF and the rest of the NO31 configuration
unchanged. All runs use fresh initialization and Valid-only evaluation.

## Update the existing checkout

    cd /home/bbx/user_lx/GraphLoRA_NO31_CORA
    conda activate minigpt4
    git status --short
    git fetch https://github.com/NanDingEi/GraphLoRA-Research.git \
      exp/no31-ctrkd-cora-20261023
    git merge --ff-only FETCH_HEAD

Do not reset the worktree or remove prior experiment output. If the merge
cannot fast-forward because of local changes, preserve them and inspect diff.

## Preflight

    python experiments/no31_cora_original/night10.py --plan
    python experiments/no31_cora_original/test_night10_cpu.py
    python experiments/no31_cora_original/test_night8_cpu.py

The full run additionally checks the historical fixed R0 checkpoint's early
CoRA-P weights. If that checkpoint was intentionally moved, only skip this
check if early-layer gradients have already been independently verified.

## Run serially

    CUDA_VISIBLE_DEVICES=0 \
    python -u experiments/no31_cora_original/night10.py --run

Optional split to manage the night:
- First four: add --first 4
- Start from G5 in a new invocation: use --start G5 (creates a new timestamped
  output directory). This starts G5 fresh; it does not continue G4's weights.
- Run one configuration only: --only G7

These are independent fresh runs, not checkpoint resumes. All invocations use
seed 42.

## Outputs

Each invocation writes a unique directory:
logs/NO31_CORA_NIGHT10_<timestamp>/, containing per-arm train.log, train.yaml,
experiment.json, result.json, plus summary.csv and results.jsonl. Checkpoints
go under the corresponding output/ directory. The summary records alpha,
projections, P LR multiplier, residual cap, depth profile, seed, Valid AUC/UAUC,
and checkpoint paths.

Interpret results conservatively: the anchors must be close to R2/F4 before
the alpha/LR/model variants can be compared. If G1 or G2 differs substantially
from its reference, inspect seed/data/config/initialization before interpreting
the other arms. No improvement is guaranteed by these experiments.
