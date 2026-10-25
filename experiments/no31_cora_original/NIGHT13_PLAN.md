# NIGHT13: grouped-rank CoRA

## Model change

This experiment adds one new architecture on top of the current best original-CoRA setup (G2: alpha=48, rank=16, Q/K/V/O in all 32 layers).

- Transformer layers 0-7: rank 4
- Transformer layers 8-15: rank 6
- Transformer layers 16-23: rank 8
- Transformer layers 24-31: rank 10

The original MF user/item concatenation, frozen MF conditioning snapshot, shared 8-layer QFormer, all-token residual injection, all Q/K/V/O targets, NO31 CTR/KD/semantic losses, data split, seed, optimizer settings, and one-epoch valid-only evaluation remain unchanged.

The QFormer and one rank-10 output head are shared across all four groups. Each group takes the first 4, 6, 8, or 10 rank channels per hidden dimension from this common basis, then renormalizes the flattened group-specific dynamic A matrix. This preserves the original CoRA design of a shared user-item-conditioned basis and changes only the rank allocation, without introducing four independent generator heads. Each layer/projection has a zero-initialized P matrix whose input width matches its group's rank. The forward remains:

    y = W x + 3.0 * P_(layer,projection) (x @ A_(user,item,group))

The effective scale alpha/r is held at 3.0, matching the G2 rank-16 / alpha-48 baseline. Therefore the corresponding group alpha values are 12, 18, 24, and 30. This avoids increasing the residual scale in lower-rank groups merely because their rank is smaller.

This is a single architecture experiment, not a parameter sweep. It tests whether allocating more collaborative rank capacity to deeper Transformer layers helps the LLM use user-item information.

## Checks and run commands

    cd /home/bbx/user_lx/GraphLoRA_NO31_CORA
    conda activate minigpt4
    git status --short
    git fetch https://github.com/NanDingEi/GraphLoRA-Research.git exp/no31-ctrkd-cora-20261023
    git merge --ff-only FETCH_HEAD

    python experiments/no31_cora_original/night13.py --plan
    python experiments/no31_cora_original/test_night13_cpu.py

    CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/night13.py --run

The run is foreground and serial. The runner refuses to start if GPU 0 is occupied. It saves the per-run `summary.csv` at `logs/NO31_CORA_NIGHT13_<timestamp>/summary.csv`, detailed `train.log` and `result.json` under `logs/NO31_CORA_NIGHT13_<timestamp>/GROUPED_RANK_4_6_8_10/`, and `checkpoint_0.pth` under the corresponding timestamped `output/NO31_CORA_NIGHT13_<timestamp>/GROUPED_RANK_4_6_8_10/` directory.

The summary reports AUC/UAUC and deltas against NO31, the original CoRA R0, and the strongest alpha-48 rank-16 G2 reference (AUC 0.7301218952, UAUC 0.6880410113).
