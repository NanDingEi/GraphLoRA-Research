# Trainable user-conditioned late-layer residual gate

## Repository / directory distinction

This experiment branch is hosted in `NanDingEi/GraphLoRA-Research`. It is **not** a branch of the upstream `VanillaCreamer/CoRA` repository. Therefore, do not run `git fetch origin exp/cora-residual-gate-train-20261103` from a clone whose `origin` is `VanillaCreamer/CoRA`.

The experiment script is intended to run inside the user's local `CoRA_official` source tree, with the experiment modules stored together under:

```
CoRA_official/
  train_collm_mf_din.py
  minigpt4/
  experiments/no31_cora_original/
    cora_joint8.py
    cora_combo8.py
    cora_rankgran7.py
    cora_struct10.py
    cora_latevote4.py
    cora_residual_gate_train.py
    local_configs/cora_mf_ood2_official.yaml
```

The base YAML is local experiment configuration and is not published by this branch. Keep the existing local YAML; do not replace it with a guessed config.

## Install into an existing CoRA_official clone

From the root of `CoRA_official`, download only the new script:

```bash
mkdir -p experiments/no31_cora_original
curl -fL \
  https://raw.githubusercontent.com/NanDingEi/GraphLoRA-Research/exp/cora-residual-gate-train-20261103/experiments/no31_cora_original/cora_residual_gate_train.py \
  -o experiments/no31_cora_original/cora_residual_gate_train.py
```

The script imports the five existing experiment modules listed above. If any are absent, fetch the matching versions from this branch before running. Existing files are not automatically overwritten.

## Validation sequence

```bash
conda activate minigpt4
python experiments/no31_cora_original/cora_residual_gate_train.py --cpu-test
python experiments/no31_cora_original/cora_residual_gate_train.py --plan
```

Only after both checks pass should training be considered:

```bash
CUDA_VISIBLE_DEVICES=0 python -u \
  experiments/no31_cora_original/cora_residual_gate_train.py \
  --run --screen-epochs 2
```

The script separates the CoRA source root (model/training code) from the experiment directory (configs/logs/output). It does not modify the CoRA Git remote, upstream source files, or existing checkpoints. Training is Valid-only; Test is disabled.
