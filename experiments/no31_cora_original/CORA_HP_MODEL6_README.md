# CoRA HP Model6: Six Single-Model Architecture Experiments

Branch: `exp/cora-hp-model6-20261116`

This experiment family builds on the HP Fullstack implementation and compares six **single-7B-model** architecture changes. It does not instantiate a second LLM, does not resume RR5/HP12/HP24 checkpoints, and does not use teacher-student distillation. The pretrained base LLM and pretrained MF initialization remain the same as the established CoRA training protocol; LoRA/router parameters are newly initialized for every trial.

## Shared controls

- Seed: 42
- GPU: CUDA device 0 (set `CUDA_VISIBLE_DEVICES=0`)
- Dynamic rank: 16, represented as four Rank4 slices
- Hierarchical user clusters: K1=3 coarse groups, K2=8 fine groups per coarse group
- D4 multi-granularity inputs and C6 auxiliary-loss path retained from HP Fullstack
- CTR weight: 0.005; MF-KD weight: 0.010
- CoRA LR: 3e-4; MF LR ratio: 0.1
- Batch/micro-batch: 2; accumulation: 4; 400 iterations per epoch
- Valid-only model selection, AUC early stopping, patience 10
- No test evaluation during training

## Six variants

| Trial | Change | Initialization / control |
|---|---|---|
| `M1_DUAL_TEMP` | Computes T=0.25 and T=0.50 routing distributions and learns a user-conditioned convex mixture | Both personal logits are independently calibrated to the same target personal share; mixture starts at 0.5/0.5 |
| `M2_USER_TEMP` | Predicts a per-user temperature in [0.25, 0.50] | Starts near T=0.47; personal logit calibrated at that temperature |
| `M3_USER_BETA` | Predicts per-user coarse/fine blend coefficient beta in [0.50, 0.90] | Starts at beta=0.70 |
| `M4_PAIR_PERSONAL` | Uses [user, item, user*item] to adjust the personal-expert logit | Zero-initialized residual, so initial route matches the HP24-style prior |
| `M5_LAYER_PERSONAL` | Learns a separate personal-expert logit for each of four 8-layer groups | Fixes the prior route_variant plumbing issue; starts with equal group logits |
| `M6_HETERO_ADAPTER` | Adds a learned affine adapter to each Rank4 slice in each layer group before soft routing | Identity initialization; rank remains 16; 4x4x4 gain and bias parameters |

**M6 clarification:** the current implementation is a lightweight expert-specific affine adapter over the four Rank4 slices of the existing dynamic generator. It is not four independent weight-generator networks. This keeps the underlying generator and rank budget intact while testing whether expert-specific transformations help.

## Server commands

Run from the CoRA repository root in the existing `minigpt4` environment:

```bash
python experiments/no31_cora_original/cora_hp_model6.py --cpu-test
python experiments/no31_cora_original/cora_hp_model6.py --plan
CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/cora_hp_model6.py --run --only M1_DUAL_TEMP,M2_USER_TEMP,M3_USER_BETA,M4_PAIR_PERSONAL,M5_LAYER_PERSONAL,M6_HETERO_ADAPTER
python experiments/no31_cora_original/cora_hp_model6.py --summary
```

Default protocol: `CORA_HP_MODEL6_20261116`.

## Outputs

- Logs: `experiments/no31_cora_original/logs/CORA_HP_MODEL6_20261116/`
- Checkpoints/configs: handled by the inherited CoRA runner under the protocol-specific output directory
- Summary: `experiments/no31_cora_original/out/CORA_HP_MODEL6_20261116/hp_model6_valid_summary.csv`
- Manifest: `experiments/no31_cora_original/logs/CORA_HP_MODEL6_20261116/hp_model6_manifest.json`

The summary reports the best validation AUC, the UAUC at that epoch, and epoch counts. Do not select weights by test metrics. After valid-only comparison, test only the preselected checkpoint(s).
