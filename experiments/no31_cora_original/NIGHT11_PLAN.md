# NIGHT11: diagnose the alpha cliff and stabilize CoRA injection

This is a follow-up to NIGHT10's observed G2/G3-G5 results. It stays in the
existing experiments/no31_cora_original/ directory and does not alter NO31's
MF/GNN/Projector/LoRA initialization, seed, CTR/KD losses, or data split.

## Evidence motivating the design

- G2, QKVO with alpha=48: Valid AUC=0.730121895, UAUC=0.688041011.
- G3, alpha=52: AUC=0.643987935, UAUC=0.588966578.
- G4, alpha=56: AUC=0.649289699, UAUC=0.609540741.
- G5, alpha=60: AUC=0.636835898, UAUC=0.598218977.
- G1, V/O alpha=32: AUC=0.725721127, UAUC=0.682916847.

The abrupt gap between alpha=48 and alpha=52 is too large to treat as a
routine monotonic scaling trend without a repeat. NIGHT11 first checks G2 and
repeats alpha=52; it then searches alpha=49/50/51. Two further arms test
conservative updates and whether Q/K residuals should be weaker than V/O.

## Eight one-epoch arms

| ID | Alpha | CoRA-P LR multiplier | Residual cap | Projection residual multipliers | Purpose |
|---|---:|---:|---:|---|---|
| N1 | 48 | 1.0 | off | Q/K/V/O = 1 | Exact G2 anchor |
| N2 | 52 | 1.0 | off | Q/K/V/O = 1 | Repeat G3; validate the performance cliff |
| N3 | 49 | 1.0 | off | Q/K/V/O = 1 | Fine search near alpha=48 |
| N4 | 50 | 1.0 | off | Q/K/V/O = 1 | Fine search near alpha=48 |
| N5 | 51 | 1.0 | off | Q/K/V/O = 1 | Fine search near alpha=48 |
| N6 | 48 | 0.25 | off | Q/K/V/O = 1 | Reduce the update magnitude of CoRA-P |
| N7 | 48 | 1.0 | 0.25 x base-output norm | Q/K/V/O = 1 | More conservative residual bound |
| N8 | 48 | 1.0 | off | Q/K = 0.5; V/O = 1 | Preserve strong V/O path while damping Q/K |

All runs: seed=42; fresh initialization; train batch=2; eval batch=3;
gradient accumulation=4; init_lr=1e-4; scheduler horizon=3; one completed
epoch; CTR=.05; KD=.05; semantic alignment weight=1e-4; projector regularizer
weight=5e-4; rank=16; all 32 layers; concatenated frozen MF user/item
features; Valid-only evaluation. No Test split is used. Do not change seed.

N1 must remain within 0.02 of historical G2 on both AUC and UAUC. N2 must
remain within 0.02 of the historical G3 result. The runner aborts automatically
if either anchor drifts outside tolerance, preserving all logs/checkpoints and
avoiding waste on an invalid comparison.

## Update and preflight on the existing checkout

    cd /home/bbx/user_lx/GraphLoRA_NO31_CORA
    conda activate minigpt4
    git status --short
    git fetch https://github.com/NanDingEi/GraphLoRA-Research.git \
      exp/no31-ctrkd-cora-20261023
    git merge --ff-only FETCH_HEAD

Do not use git reset --hard. If local changes block a fast-forward, inspect
and preserve them.

    python experiments/no31_cora_original/night11.py --plan
    python experiments/no31_cora_original/test_night11_cpu.py
    python experiments/no31_cora_original/test_night10_cpu.py
    python experiments/no31_cora_original/test_night8_cpu.py

## Run

    CUDA_VISIBLE_DEVICES=0 \
    python -u experiments/no31_cora_original/night11.py --run

Run serially in the foreground on GPU0. The script refuses to overlap with
another GPU process. For a shorter initial diagnostic, run N1 and N2 only:

    CUDA_VISIBLE_DEVICES=0 \
    python -u experiments/no31_cora_original/night11.py --run --first 2

After reviewing those anchors, the remaining runs can be started in a new
timestamped output directory with --start N3.

For one arm only, use --only N8, for example. This always starts a fresh run;
it is not checkpoint continuation.

## Interpretation

- If N2 repeats the collapse, the alpha cliff is reproducible under the fixed
  setup; use N3-N5 to locate a safer boundary.
- If N2 does not repeat, stop treating alpha as the explanation and inspect
  run determinism, data ordering, optimizer state, and configuration drift.
- If N3-N5 are close to N1, prefer the best stable alpha based on both AUC and
  UAUC, not a single noisy point.
- N6, N7, and N8 isolate a smaller CoRA-P update, stricter residual cap, and
  projection-wise residual scaling respectively. They are hypotheses, not
  guaranteed improvements.

Logs and result summaries go under logs/NO31_CORA_NIGHT11_<timestamp>/;
checkpoints go under the corresponding timestamped output/ directory. Prior
NIGHT10 results are not overwritten.
