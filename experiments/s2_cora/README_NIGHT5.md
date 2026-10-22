# S2 recovery + 4 CoRA model experiments (2 epochs each)

## Historical starting point

All experiments use code derived from the archived S2 commit
`d818fa2943ede4e9a810b032a3422e94d0c77ed0`, not the more heavily
modified R4/R5 trainer. Historical `S2_groups_acc3` recorded Valid AUC
0.729675 at epoch 0 and 0.732884 at epoch 1. The recovered YAML is based
on effective parameters logged at training time.

## Five total runs

| Run | Mode | Seed | Architecture change | Epochs |
| --- | --- | --- | --- | --- |
| 01_S2_CONTROL | original S2 | 42 | none, reproduction gate | 0 and 1 |
| 02_CORA_QV | CoRA | 42 | full collaborative user/item Q+V adapter | 0 and 1 |
| 03_CORA_Q_ONLY | CoRA | 42 | condition only L31 Query projection | 0 and 1 |
| 04_CORA_V_ONLY | CoRA | 42 | condition only L31 Value projection | 0 and 1 |
| 05_CORA_INTERACTION | CoRA | 42 | user+item+user×item tokens to collaborative cross-attention; L31 Q+V | 0 and 1 |

Every arm uses original pretrained MF, the same fresh NGCF/Projector/LLM
LoRA seed-42 initialization, original semantic alignment 1e-4, original
Projector regularization 5e-4, original group LR multipliers, gradient
accumulation 3, and the same validation dataset. No warmed-up S2 GraphLoRA
checkpoint and no R4/R5 weights are loaded.

Each arm trains exactly **two full epochs** (the epoch-0 and epoch-1 passes),
with `run.max_epoch=3` kept for the historical LR schedule and an explicit
stop after the second completed epoch. Validate AUC/UAUC and save a checkpoint
after **each** epoch. `checkpoint_best.pth` is chosen by Valid AUC only.
No Test data are evaluated.

## Fairness guards

- S2 reproduction must achieve best Valid AUC >= 0.72 or the queue stops.
- CoRA is zero-output at initialization: its Q/V up matrices are zero.
- Before each CoRA training run, the entrypoint builds the model and hashes
  the MF, NGCF, Projector, and LoRA weights. All must match the initial
  S2 control SHA256 exactly, or the run is refused.
- Both full train epochs, both validations, `checkpoint_0.pth`,
  `checkpoint_1.pth` and best checkpoint are required.
- A partially failed run is not silently overwritten. The original S2
  checkpoint, graphs, dataset, and R4/R5 outputs remain untouched.

## Deployment to an isolated worktree

```bash
cd /home/bbx/user_lx/GraphLoRA_collab_v1
git fetch https://github.com/NanDingEi/GraphLoRA-Research.git exp/s2-cora-night5-20261023
git worktree add -b s2-cora-five-local-20261023 /home/bbx/user_lx/GraphLoRA_S2_CORA_FIVE FETCH_HEAD
cd /home/bbx/user_lx/GraphLoRA_S2_CORA_FIVE
conda activate minigpt4

# No GPU training in this step
bash experiments/s2_cora/run_night5.sh prepare

# 5 runs × 2 full epochs, serially, GPU0, foreground
CUDA_VISIBLE_DEVICES=0 bash experiments/s2_cora/run_night5.sh run
```

Or run `bash experiments/s2_cora/run_night5.sh all` from an unprepared
working tree. The launcher does not schedule background training.

## Results

- Logs/configurations and `summary.csv`:
  `logs/S2_CORA_FIVE_20261023/`
- Checkpoints:
  `output/S2_CORA_FIVE_20261023/`
- Comparison columns: epoch-0 AUC/UAUC; epoch-1 AUC/UAUC;
  best epoch; best Valid AUC/UAUC; difference versus reproduced S2.

Runtime is substantial: five methods times two full epochs represents
ten complete epochs plus validation. The entire sequence may not finish
within a single night on one RTX 3090. No actual GPU results are claimed
until the server runs the scripts.
