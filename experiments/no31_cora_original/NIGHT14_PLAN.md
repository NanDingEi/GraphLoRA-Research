# NIGHT14 — parameter-matched grouped-rank CoRA

## Research question

Does distributing the same overall CoRA-P rank budget toward later Vicuna layers improve ML-1M OOD2 Valid AUC and UAUC, compared with uniform rank 16 on all 32 layers?

Historical reference: G2 (Q/K/V/O, 32 layers, uniform rank 16, alpha 48):
AUC 0.7301218951846701, UAUC 0.68804101129963.
Previous compressed group ranks 4/6/8/10 achieved AUC 0.7277484056529369 and UAUC 0.6830067108807387, while using only 43.75% of G2's CoRA-P parameters; that result does **not** test a matched parameter budget.

## Exact matched P budget

| Layer range | Group rank | Number of P matrices | Parameters |
|---|---:|---:|---:|
| L0–L7 | 12 | 32 | 1,572,864 |
| L8–L15 | 14 | 32 | 1,835,008 |
| L16–L23 | 18 | 32 | 2,359,296 |
| L24–L31 | 20 | 32 | 2,621,440 |
| **Total** | Mean rank 16 | **128** | **8,388,608** |
| G2 uniform 16 | Mean rank 16 | 128 | 8,388,608 |

All four groups retain Q/K/V/O projections. Each \`cora_lora_P.weight\` is (4096, group's rank), initialized to zero. Scaling is held at \`s = alpha_base/rank_base = 48/16 = 3.0\`, equivalent to local alpha values 36/42/54/60. Thus changing rank does not inadvertently amplify low-rank layers by a factor \`48/r_g\`.

## Why the generator must also be budget-matched

A naive shared rank-20 dense head \`Linear(512,4096*20)\` needs 8,388,608 additional weights **plus bias** compared to the G2 rank-16 generator. Equal P parameters would then conceal a substantial increase in generator parameters.

NIGHT14 therefore preserves the original G2-compatible \`final: Linear(512,4096*16)\` and its initialization. For layers requiring rank 18 or 20, it adds four independent nonlinear channels using a compact additional branch:

    extra_rank_head = Linear(512,32) -> GELU -> Linear(32,4096*4)

- Extra trainable parameters in this branch: **557,088**.
- Combined G2 P and final head (excluding unchanged shared QFormer/others): **42,008,576**.
- NIGHT14's combined P, final head, extra branch: **42,565,664**, or **+1.326%**.
- Including the unchanged QFormer and other trainable components makes the proportion smaller.
- Group ranks 12/14/18/20 select a common prefix of 12/14/18/20 channels from the generated rank-20 basis, and normalize each selected basis once.
- New channels depend nonlinearly on the QFormer states, so they are not just a 20-column linear reparameterization of a rank-16 tensor.

Note: This is **not exactly equal total trainable parameter count**; it is equal CoRA-P count and about +1.3% of P plus main-head count. The branch keeps a true rank-above-16 extension possible without a large dense generator.

## Controls

The experiment reuses the original input \`[MF user; MF item]\`, QFormer (8 layers / 4 queries), all 32 Q/K/V/O injection locations, zero-initialized P, batch 2 / eval batch 3 / gradient accumulation 4, training seed 42, CTR/KD losses 0.05/0.05, semantic and projector regularization, no direct GNN injection (NO31), the 3-epoch LR schedule with training terminated after one epoch, and Valid-only evaluation. No Test split is queried.

This experiment changes *rank allocation and only the necessary low-cost high-rank generation mechanism*. No alpha sweep, no LR sweep, no token gate, no interaction token. Prior logs and checkpoints are retained.

## Run on existing checkout (no new clone/worktree)

    cd /home/bbx/user_lx/GraphLoRA_NO31_CORA
    conda activate minigpt4
    git status --short
    git fetch https://github.com/NanDingEi/GraphLoRA-Research.git exp/no31-ctrkd-cora-20261023
    git merge --ff-only FETCH_HEAD

    python experiments/no31_cora_original/night14.py --plan
    python experiments/no31_cora_original/test_night14_cpu.py
    python experiments/no31_cora_original/test_night8_cpu.py
    nvidia-smi

    CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/night14.py --run

The runner writes an isolated timestamped \`logs/NO31_CORA_NIGHT14_*/summary.csv\`, detailed \`BUDGET_MATCHED_RANK_12_14_18_20/train.log\`, and a corresponding checkpoint in \`output/NO31_CORA_NIGHT14_*/\`. It refuses overlapping GPU0 work.

Compare against both G2 (AUC 0.730121895, UAUC 0.688041011) and the previous compressed profile (AUC 0.727748406, UAUC 0.683006711). Do not claim a reliable accuracy improvement from a single seed / one epoch without replication.
