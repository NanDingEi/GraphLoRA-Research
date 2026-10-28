# NIGHT22 — EXACT NIGHT13 ranked-CoRA four-loss ablation (8 training runs)

**Correct baseline:** `experiments/no31_cora_original/night13.py`,
`GROUPED_RANK_4_6_8_10`; not NIGHT17 group8-rank16, NIGHT19, NIGHT20
prototype paths, or unrelated CoRA R0.

The user's *reported* prior rank410 Valid metrics are AUC **0.727748** and
UAUC **0.683007**. They are not read from a verified archived raw log by this
runbook. **R0 retrains the same all-auxiliary-on CoRA first**; it is the
actual baseline for the seven ablations. When R0 differs from the reported
history by more than **0.015 on either AUC or UAUC**, the runner aborts before
the remaining seven GPU runs.

## The eight runs (including independent baseline retraining)

| ID | CTR | KD | Semantic | Item-projector preservation |
|---|---:|---:|---:|---:|
| R0_BASE_ALL_ON | .05 | .05 | .0001 | .0005 |
| R1_NO_CTR | 0 | .05 | .0001 | .0005 |
| R2_NO_KD | .05 | 0 | .0001 | .0005 |
| R3_NO_CTR_KD | 0 | 0 | .0001 | .0005 |
| R4_NO_SEM | .05 | .05 | 0 | .0005 |
| R5_NO_PROJ | .05 | .05 | .0001 | 0 |
| R6_NO_SEM_PROJ | .05 | .05 | 0 | 0 |
| R7_ALL_OFF | 0 | 0 | 0 | 0 |

This includes the full 2x2 CTR x KD factorial *conditioned on semantic and
projector on*, and 2x2 semantic x projector factorial *conditioned on CTR/KD
on*; R0 is shared, plus the all-off anchor. Four factors' complete 2^4
factorial would require 16 runs, not 8, so do not claim all interactions
are identified.

Architecture (exact NIGHT13):
- frozen pretrained MF concatenated user/item embedding -> original Q-Former
- single rank10 generator with normalized rank prefixes; L0-7 rank4,
  L8-15 rank6, L16-23 rank8, L24-31 rank10
- alpha48, effective alpha/r scaling=3.0 every group
- **group_projectors=off**; full Q/K/V/O L0..31, seed42, fresh CoRA
- unchanged NO31/no L31 GNN injection and no input-level GNN mixing
- original training YAML optimizer/LR/batch, train one epoch (LR schedule
  uses unchanged 3-epoch horizon), same Valid only, no Test
- one process at a time on GPU0 in foreground, checkpoint_0 per run.

Control for unrelated random-state and architecture changes:
- all runs retain a trainable `gnn_conf_kappa` parameter even if CTR and KD
  are both off
- semantic teacher is still computed when its *weight* is zero to preserve
  the same model/teacher path
- random 64-item projector sampling is still computed if projector *weight*
  is zero (shadow, no gradient from projector loss)
- all training starts independently from pretrained MF/Vicuna, never
  warm-start from previous ablation
- structural and actual-loss checks reject invalid combinations.

## Server instructions (separate worktree, no background training)

    cd /home/bbx/user_lx/GraphLoRA_NO31_CORA
    git fetch https://github.com/NanDingEi/GraphLoRA-Research.git exp/night22-rank410-loss-factorial-20261027
    git worktree add --detach /home/bbx/user_lx/GraphLoRA_NIGHT22 FETCH_HEAD
    cd /home/bbx/user_lx/GraphLoRA_NIGHT22
    conda activate minigpt4

Copy ONLY the original all-on prepared training config:

    mkdir -p logs/NO31_CORA_ORIGINAL_READY
    cp /home/bbx/user_lx/GraphLoRA_NO31_CORA/logs/NO31_CORA_ORIGINAL_READY/train.yaml logs/NO31_CORA_ORIGINAL_READY/train.yaml

CPU checks and plan:

    python experiments/no31_cora_original/test_night13_cpu.py
    python experiments/no31_cora_original/test_night22_cpu.py
    python experiments/no31_cora_original/night22.py --plan

Run all eight (R0 first, fail-fast on drift):

    CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/night22.py --run

Outputs:

    logs/NO31_CORA_NIGHT22_<timestamp>/summary.csv
    logs/NO31_CORA_NIGHT22_<timestamp>/R0_BASE_ALL_ON/train.log
    logs/NO31_CORA_NIGHT22_<timestamp>/R0_BASE_ALL_ON/result.json
    logs/NO31_CORA_NIGHT22_<timestamp>/R1_NO_CTR/...
    ...
    output/NO31_CORA_NIGHT22_<timestamp>/R0_BASE_ALL_ON/.../checkpoint_0.pth

Inspect:

    LOG_DIR=$(ls -td logs/NO31_CORA_NIGHT22_* | head -1)
    cat "$LOG_DIR/summary.csv"
    grep -E 'NIGHT22-LOSS-CONFIG|NIGHT22-LOSS-AUDIT-PASS|NIGHT22-RERUN-DELTA' "$LOG_DIR"/R*/train.log

Note: `NIGHT22-LOSS-AUDIT-PASS` and `NIGHT22-RERUN-DELTA` are
**runner stdout**, not necessarily inside per-run train.log. Use terminal
output or save a foreground tee record:

    CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/night22.py --run 2>&1 | tee night22_console.log

Check the complete source-side model audit:
`[NIGHT22-LOSS-CONFIG]`, `[CORA-NIGHT22-ADAPTER-CONFIG]`,
`[semantic]`, `[projector-reg]`, and (when CTR or KD is enabled)
`[NIGHT8-LOSS]`.

The generated `factorial_interactions.json` quantifies how CTR behaves with
KD on versus KD off (`CTR_x_KD_interaction`), and the analogous semantic x
projector contrast, for both AUC and UAUC. These are descriptive
**single-seed** contrasts, not significance tests.

Each `summary.csv` row has its actual `loss_ctr/loss_kd/loss_sem/loss_proj`
weights and `delta_vs_retrained_rank410_auc/uauc`. This **current
retrained R0**, not the historical NIGHT17 E0 / generic CoRA R0, is the
primary comparison.

If R0 fails the historical range, STOP. Check actual NIGHT13 archived
`train.yaml`, code commit, pretrained MF and Vicuna file versions,
validation split, GNN/CoRA parameter trainability and optimizer behavior
before interpreting ablations.

**Status:** Submitted to GitHub, not run on user's GPU. No ablation results
are claimed in this document.
