# NO31 + Original CoRA: four arms, two total epochs

This uses the existing \`/home/bbx/user_lx/GraphLoRA_NO31_CORA\` checkout and
the existing branch \`exp/no31-ctrkd-cora-20261023\`. Do not create a new
worktree and do not overwrite the R2/R7/F4 historical logs/checkpoints.

## Exact protocol

All arms use seed 42, GPU0, serial foreground training, Valid-only evaluation,
batch size 2, eval batch size 3, gradient accumulation 4, original 3-epoch
LR schedule horizon (\`max_epoch=3\`), CTR=.05, KD=.05, semantic correction
weight 1e-4, projector regularization 5e-4, autocast cache fix ON, rank16,
CoRA-P LR multiplier 1.0. No Test split is enabled.

| Arm | Start | CoRA targets | Alpha | Epoch accounting |
|---|---|---|---:|---|
| N1 | Resume R2 checkpoint_0 | V/O, layers 0–31 | 32 | historical epoch 0 + newly trained epoch 1 = 2 total |
| N2 | Resume R7 checkpoint_0 | Q/K/V/O, layers 0–31 | 64 | historical epoch 0 + newly trained epoch 1 = 2 total |
| N3 | Resume F4 checkpoint_0 | Q/K/V/O, layers 0–31 | 48 | historical epoch 0 + newly trained epoch 1 = 2 total |
| N4 | Fresh initialization | V/O, layers 0–31 | 48 | train epoch 0 and epoch 1 = 2 total |

N1/N2/N3 restore model weights, AdamW state and AMP scaler from the original
epoch-0 checkpoints. \`max_epoch=3\` is intentionally retained as the LR schedule
horizon; \`S2_NIGHT_EPOCHS=2\` caps the number of completed epochs. This means
the resumed arms run only epoch 1, not two *additional* epochs.

The existing checkpoint format does not store RNG states. Resumed runs preserve
the configured seed and all saved optimizer/model/scaler state, but cannot
claim bitwise-identical RNG continuation.

## Update the existing checkout

    cd /home/bbx/user_lx/GraphLoRA_NO31_CORA
    conda activate minigpt4
    git status --short
    git fetch https://github.com/NanDingEi/GraphLoRA-Research.git \
      exp/no31-ctrkd-cora-20261023
    git merge --ff-only FETCH_HEAD

If local modifications block a fast-forward, preserve them and inspect the
diff. Do not use \`git reset --hard\` or delete old output folders.

## Preflight without GPU training

    python experiments/no31_cora_original/test_continue4_cpu.py
    python experiments/no31_cora_original/continue4_two_epochs.py --plan
    python experiments/no31_cora_original/continue4_two_epochs.py --check-checkpoints

The checkpoint preflight checks the exact experiment metadata, epoch index,
active CoRA-P tensors (nonzero for L0–L7), AdamW state, AMP scaler and the
saved NO31 config. It stops before training if anything is inconsistent.

## Run all four arms, serially on GPU0

    CUDA_VISIBLE_DEVICES=0 \
    python -u experiments/no31_cora_original/continue4_two_epochs.py --run

To run a single arm:

    CUDA_VISIBLE_DEVICES=0 \
    python -u experiments/no31_cora_original/continue4_two_epochs.py --run --only N3

The script refuses to overlap another GPU0 compute process. If any arm fails,
it stops and preserves all completed logs/checkpoints. To retry, fix the
reported cause and run the failed arm selectively; it creates a new timestamp.

## Output

    logs/NO31_CORA_CONTINUE4_TWOEPOCHS_<timestamp>/
      N1/{train.yaml,experiment.json,train.log,result.json}
      N2/{train.yaml,experiment.json,train.log,result.json}
      N3/{train.yaml,experiment.json,train.log,result.json}
      N4/{train.yaml,experiment.json,train.log,result.json}
      summary.csv
      results.jsonl
      plan.json

Checkpoints are written under the matching \`output/NO31_CORA_CONTINUE4_TWOEPOCHS_<timestamp>/N*/.../\`.
\`summary.csv\` records epoch-0 and epoch-1 AUC/UAUC and their changes. For N1–N3,
epoch 0 is the historical checkpoint's recorded validation result; for N4,
both values are measured in the new run.

The assistant has updated the repository scripts and has not executed GPU training.
