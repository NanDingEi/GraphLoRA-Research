# CoRA Residual Routing: six controlled training experiments

Script: `cora_residual_routing6.py`

This suite runs six independent end-to-end CoRA training trials, each in a
fresh foreground subprocess on physical GPU0. All trials use the same
D4_TOPK_RES025 / neighbor_mix input path, C6_KD_BOTH losses, pretrained MF
initialization, seed 42, CoRA LR 3e-4, MF LR 3e-5, batch 2, accumulation 4,
400 iterations per epoch, Valid-only early stopping (patience 10), and
trainable-only atomic best checkpoints. Test splits are disabled.

## Trials

| ID | Structure | Purpose |
|---|---|---|
| RR1_L32_BASE | (s=z_{32}) | C6 baseline |
| RR2_GLOBAL_R31 | (s=z_{32}+\alpha(z_{31}-z_{32})) | Global single-layer residual |
| RR3_FIXED_MULTI | (s=z_{32}+\alpha\,\mathrm{mean}(r_{29},r_{30},r_{31})) | Fixed multi-layer residual |
| RR4_GLOBAL_ROUTER | (s=z_{32}+\alpha\sum_l\pi_l r_l) | Global learned layer router |
| RR5_USER_GATE_R31 | (s=z_{32}+g(u)r_{31}) | User-conditioned single-layer gate |
| RR6_USER_ROUTER_MULTI | (s=z_{32}+g(u)\sum_l\pi_l(u)r_l) | User-conditioned multi-layer router |

All residuals are defined as (r_l=z_l-z_{32}). User-conditioned trials use
only the user half of the raw MF UI embedding. The gate prior constrains only
the batch mean, not individual users. There is no trainable affine
residualizer in this suite.

## Install into CoRA_official

Run from the root of the local CoRA source checkout:

```bash
mkdir -p experiments/no31_cora_original
curl -fL \
  https://raw.githubusercontent.com/NanDingEi/GraphLoRA-Research/exp/cora-residual-gate-train-20261103/experiments/no31_cora_original/cora_residual_routing6.py \
  -o experiments/no31_cora_original/cora_residual_routing6.py
```

## Validate and prepare

```bash
conda activate minigpt4
python experiments/no31_cora_original/cora_residual_routing6.py --cpu-test
python experiments/no31_cora_original/cora_residual_routing6.py --plan
```

`--plan` validates the local CoRA source API and all six configs, and writes
the manifest. It does not load the 7B model or allocate GPU memory.

## Run all six sequentially

```bash
python -u experiments/no31_cora_original/cora_residual_routing6.py --run
```

The launcher checks that GPU0 has at least 22,000 MiB free before each worker;
each worker checks again before model construction. It runs in the foreground,
writes one log per trial, stops at the first failure, and never overwrites an
existing trial log.

To run only selected trials:

```bash
python -u experiments/no31_cora_original/cora_residual_routing6.py --run \
  --only RR5_USER_GATE_R31,RR6_USER_ROUTER_MULTI
```

Do not run a short screen and then reuse the same protocol for a full run:
existing logs are intentionally protected from overwrite. Use a fresh protocol
name for a clean rerun.
