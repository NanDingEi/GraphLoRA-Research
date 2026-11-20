# HP24 Warm-start Routing Improvements

Branch: `exp/cora-hp24-warmstart-20261120`

This experiment family makes **three single-model, HP24-initialized** routing improvements. It does not instantiate a second LLM, teacher/student model, or ensemble.

## Source model

- Source protocol: `CORA_HIER_FULLSTACK_20261113`
- Source trial: `HP24_B07_T05`
- Expected source checkpoint directory:
  `experiments/no31_cora_original/output/CORA_HIER_FULLSTACK_20261113/HP24_B07_T05/`
- The source checkpoint must be the compact `trainable_model_only_v1` checkpoint produced by the HP24 run. The script verifies the format, seed, tensor names, and shapes before loading.
- Frozen Vicuna weights are reused from the existing configured base model. The HP24 trainable LoRA, MF, and hierarchical-router tensors are loaded from the HP24 checkpoint. Optimizer state is intentionally reset.

## Three trials

| Trial | Change | HP24-preserving initialization |
|---|---|---|
| `W1_PAIR_AWARE` | Candidate-aware personal-expert logit using `[u, i, u*i]` | Zero-initialized final layer; initial route exactly matches HP24 |
| `W2_ADAPTIVE_BETA` | Per-user coarse/fine blend `beta_u=0.5+0.4*sigmoid(MLP(u))` | Zero-initialized final layer; initial beta is 0.70 |
| `W3_LAYER_PERSONAL` | Four layer-group-specific residual offsets to the personal-expert logit | Four zero offsets; original scalar HP24 `personal_logit` key and shape are preserved |

All trials keep rank 16 (four Rank4 slices), K1=3, K2=8, route temperature 0.50, base beta 0.70, personal target 0.40, residual lambda 0.20, D4 grouped neighbor inputs, and C6 losses.

## Warm-start schedule

1. Build the exact HP24 runtime and load the HP24 trainable checkpoint.
2. Add only the selected zero-initialized routing module.
3. Train the new router for two epochs while the original HP24 parameters remain frozen. The MF optimizer group is materialized first to satisfy the inherited Joint8 optimizer audit; MF parameters are frozen during this phase.
4. At epoch 2, unfreeze the original HP24 trainable parameters and add them to the optimizer. Continue joint fine-tuning with a fresh optimizer state.
5. Best checkpoints always save the complete source HP24 trainable state plus the new routing parameters, including during router-only warmup.

## Fixed controls

- Seed 42; physical GPU0 only; serial trials
- CoRA LR 1e-4; MF LR 1e-5 (0.1x)
- Batch size 2; accumulation 4; 400 iterations/epoch
- Router-only warmup: 2 epochs / 800 steps
- Min LR 2e-5; warmup LR 5e-6
- Patience 10; Valid-only checkpoint selection
- Test split disabled during training

## Server commands

Run in the existing `minigpt4` environment from the CoRA repository root:

```bash
python experiments/no31_cora_original/cora_hp24_warmstart.py --check
python -m py_compile experiments/no31_cora_original/cora_hp24_warmstart.py
python experiments/no31_cora_original/cora_hp24_warmstart.py --cpu-test
python experiments/no31_cora_original/cora_hp24_warmstart.py --plan
```

Start all three serial GPU0 experiments:

```bash
CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/cora_hp24_warmstart.py --run --only W1_PAIR_AWARE,W2_ADAPTIVE_BETA,W3_LAYER_PERSONAL
```

Read Valid-only results:

```bash
python experiments/no31_cora_original/cora_hp24_warmstart.py --summary
tail -n 60 experiments/no31_cora_original/logs/CORA_HP24_WARMSTART_20261120/W1_PAIR_AWARE.log
```

After choosing a checkpoint from Valid results, evaluate only the explicitly selected trial on Test:

```bash
CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/cora_hp24_warmstart.py --test --only W1_PAIR_AWARE
```

The test command requires an explicit `--only` selection and refuses to overwrite an existing test report. It is not called by the training command.

## Outputs

- Configs: `local_configs/CORA_HP24_WARMSTART_20261120/`
- Logs: `logs/CORA_HP24_WARMSTART_20261120/`
- Checkpoints: `output/CORA_HP24_WARMSTART_20261120/`
- Valid summary: `out/CORA_HP24_WARMSTART_20261120/hp24_warmstart_valid_summary.csv`
- Test reports: `out/CORA_HP24_WARMSTART_TEST_20261120/`
