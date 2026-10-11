# GraphLoRA training controls v1

For GraphLoRA versions whose Runner/Task/Scheduler still use the upstream construction patterns. Works alongside user-local semantic alignment and Projector Reg patches: **does not edit `minigpt4/models/minigpt4rec_v2.py`**.

## What it changes

1. `minigpt4/runners/runner_base.py`: optimizer parameter grouping by **MF embedding (`rec`) / input Projector (`proj`) / GNN (`gnn`) / LoRA (`lora`) / other**, preserving weight decay vs. no-weight-decay. Parameter groups have an `lr_scale` and a logged `group_name`. The GNN group covers `gcn.*`, `gnn.*`, `Wneck` and `gnn_proj`, and the custom Layer31 q/v projection bias if present. Unknown trainable parameters go to `other`.
2. `minigpt4/common/optims.py`: each of the **three** scheduler writers sets `param_group['lr'] = base_lr * param_group.get('lr_scale', 1.0)`; groups with no `lr_scale` remain identical to upstream.
3. `minigpt4/tasks/base_task.py`: each forward loss divided by actual window size, optimizer stepped only after a complete window or last batch; non-divisible tail is supported. AMP `GradScaler` remains intact, reported training loss remains unscaled.

Original files are timestamp-backed-up on `--apply`. Check performs no modifications; installer refuses already-patched code and unfamiliar source shapes. Make sure the `--check` passes **on your customized local repo**.

## Installation

Place ZIP in `/home/bbx/user_lx/GraphLoRA` and unzip in project root:

```bash
cd /home/bbx/user_lx/GraphLoRA
unzip -o graphlora_train_controls_v1.zip
python graphlora_train_controls_v1/install_training_controls.py --check
python graphlora_train_controls_v1/install_training_controls.py --apply
python -m py_compile minigpt4/runners/runner_base.py minigpt4/common/optims.py minigpt4/tasks/base_task.py
python -m unittest discover -s graphlora_train_controls_v1/tests -v
```

If local source differs from upstream and the check fails, **stop**; send error and those specific files. Do not force a patch or replace your custom files with upstream.

## New `run` configuration keys

All default to backward-compatible values, so existing commands behave normally unless explicitly overridden:

| Key | Default | Meaning |
|---|---:|---|
| `run.lr_scale_rec` | 1.0 | MF embedding LR relative to scheduler base LR |
| `run.lr_scale_proj` | 1.0 | `llama_proj` LR relative to scheduler base LR |
| `run.lr_scale_gnn` | 1.0 | GNN/network/bottleneck LR relative to scheduler base LR |
| `run.lr_scale_lora` | 1.0 | LoRA matrices LR relative to scheduler base LR |
| `run.lr_scale_other` | 1.0 | other trainable params LR relative to scheduler base LR |
| `run.accum_grad_iters` | 1 | microbatch count per optimizer update |

The model-specific loss coefficients **already** exist in the user's custom code:

| Key | Role |
|---|---|
| `model.gnn_semantic_weight` | Multiply semantic correction loss gradients (`L_sem`) |
| `model.proj_reg_weight` | Multiply projector cosine loss gradients (`L_proj`) |

**A loss does not have an independent learning rate in a shared optimizer.** A GNN module LR scale affects *all* losses that update that GNN, including recommendation and semantic alignment. Projector LR scales affect both recommendation and projector regularization updates. To set a truly independent optimizer/gradient scale for *only* the semantic loss on overlapping parameters, the local semantic loss computation would need to expose the raw scalar; the installer intentionally does not attempt that without the custom model file.

## Suggested first controlled experiment

From S2 configuration, start **fresh** (model.ckpt=null, resume_ckpt_path=null) with base LR = `1e-4`, Projector Reg `5e-4`, semantic loss `1e-4`, and **effective batch size = 3×3 = 9**. GNN LR×0.25, Projector×0.5, MF×0.5, LoRA×1.0. This changes accumulation *and* module learning rates; compare to S2 as an exploratory run, then perform isolated ablations if it improves AUC.

```bash
cd /home/bbx/user_lx/GraphLoRA
unset GLORA_GRAD_PROBE
mkdir -p logs
python -u train_collm_mf_din.py \
  --cfg-path output/overnight_auc_sweep_3/configs/S2_sem_proj_0005.yaml \
  --options \
  run.init_lr=1e-4 run.min_lr=5e-5 run.warmup_lr=1e-5 \
  run.accum_grad_iters=3 \
  run.lr_scale_rec=0.5 run.lr_scale_gnn=0.25 \
  run.lr_scale_proj=0.5 run.lr_scale_lora=1.0 \
  model.gnn_semantic_weight=1e-4 model.proj_reg_weight=5e-4 \
  model.ckpt=null run.resume_ckpt_path=null \
  run.output_dir=/home/bbx/user_lx/GraphLoRA/output/S2_groups_acc3 \
  2>&1 | tee logs/S2_groups_acc3.log
```

### Isolating semantic loss coefficient

To halve only the **semantic loss gradient contribution**, change `model.gnn_semantic_weight=5e-5` while leaving `run.lr_scale_gnn` fixed. To halve only projector regularization gradient contribution, change `model.proj_reg_weight=2.5e-4` while leaving `run.lr_scale_proj` fixed.

### Training/checkpoint cautions

- Validation AUC/UAUC selects the checkpoint; never select model hyperparameters using Test AUC.
- Scheduler `iters_per_epoch` is still the microbatch count (11297 in the logged ML-1M run). The scheduler advances every microbatch, while optimizer steps once per accumulation window. This preserves the existing schedule's elapsed-data semantics, but **warmup_steps=200 counts microbatches**, or about 67 optimizer updates when accumulation=3.
- The runner's reported `lr` is the **first param group's effective LR**, not necessarily the base LR. Look for `[lr-group]` startup logs.
- With 11297 microbatches and accumulation 3, optimizer steps 3766 times per epoch (last update covers 2 microbatches).
- A batch size of 3 with accumulation 3 approximates physical batch size 9 for independent-sample losses. If other losses depend on each microbatch (e.g., random graph/neighborhood samples, pairwise regularization), full-batch equivalence is not exact for those losses.
- Restarting training with `model.ckpt` or `resume_ckpt_path` set changes the experiment. Save to an entirely new `output_dir`.
- If `--apply` succeeds, do not invoke it a second time. Roll back a file only using its own matching timestamp backup, after stopping any training.
