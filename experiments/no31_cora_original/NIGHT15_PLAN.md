# NIGHT15: group-specific collaborative representation projectors

## Purpose and controls

Compare the existing G2 dynamic CoRA-A generator against a shared collaborative projector and four independent depth-group projectors. All other G2 choices are held fixed: frozen MF user/item concat condition, 8-layer/4-query QFormer, rank16 alpha48, all 32 Q/K/V/O layers, 128 original independent CoRA-P matrices, CTR=.05, KD=.05, semantic alignment, projector regularization, seed42, LR=1e-4, train batch2, eval batch3, accumulation4, 3-epoch LR horizon with training stopped after one complete epoch, Valid only (no Test split).

Historical G2 reference: Valid AUC 0.7301218951846701, Valid UAUC 0.68804101129963.

| Arm | Mechanism | Extra parameters |
|---|---|---:|
| P0_G2_CONTROL | Original shared QFormer-to-dynamic-A head | 0 |
| P1_SHARED_PROJECTOR | One shared residual MLP between QFormer and original final head | 33,312 |
| P2_FOUR_GROUP_PROJECTORS | Four independent residual MLPs for L0–7, L8–15, L16–23, L24–31 | 133,248 |

Each MLP is 512 -> 32 -> 512 with GELU. Its last linear layer starts at exactly zero: Z_g = Z + MLP_g(Z) equals Z at initialization. The original head remains 512 -> (4096*16). The projectors transform the four QFormer query states before dynamic matrix generation, rather than replacing the 128 already independent per-layer CoRA-P modules.

For each group g and layer l in the group:

    Z = QFormer(concat(MF_user,MF_target))    # [B,4,512]
    Z_g = Z + Projector_g(Z)
    A_g = reshape(normalize(mean(Final(Z_g),queries)))  # [B,4096,16]
    output(l,p) = W(l,p)@x + 3 * P(l,p)(dropout(x)@A_g)

P1 uses the same A for all groups. P2 learns potentially distinct A across groups. The original QFormer, final rank16 generator head, and CoRA-P are shared exactly across variants. Compared with the existing P plus final head parameter budget (~42.01M), P1 adds 0.079% and P2 adds 0.317%.

P0 reruns G2 and automatically aborts further arms if its Valid AUC or UAUC differs from G2 historical results by over 0.02. This is a compatibility check, not a claim of statistical significance.

## How to run (existing checkout, GPU0, foreground serial)

    cd /home/bbx/user_lx/GraphLoRA_NO31_CORA
    conda activate minigpt4
    git status --short
    git fetch https://github.com/NanDingEi/GraphLoRA-Research.git exp/no31-ctrkd-cora-20261023
    git merge --ff-only FETCH_HEAD

    python experiments/no31_cora_original/night15.py --plan
    python experiments/no31_cora_original/test_night15_cpu.py
    python experiments/no31_cora_original/test_night8_cpu.py
    nvidia-smi

    CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/night15.py --run

Run only the G2 control if desired:

    CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/night15.py --run --first 1

Then run the other two in serial in a new timestamped directory:

    CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/night15.py --run --start P1_SHARED_PROJECTOR

Logs go under logs/NO31_CORA_NIGHT15_<timestamp>/; checkpoints go under the matching output/NO31_CORA_NIGHT15_<timestamp>/ directory. One summary.csv file is written for each batch. Do not run other GPU jobs concurrently. Keep previous NIGHT10 through NIGHT14 results and checkpoints unchanged.

After training, [CORA-GROUP-PROJECTORS] prints terminal projection weight norms for P1 and P2. Positive norms support the presence of learned changes but are not themselves evidence of improvement. Prefer both Valid AUC and UAUC, and compare P2 vs P1 to isolate the benefit of depth-specific projector specialization.
