# Collaborative injection: guarded 3-epoch experiments

Branch: `exp/collab-injection-e3`

This branch adds a config generator and a serial runner. It does not change the model implementation from `exp/collab-injection-v1`.

## 1. Pull the branch

If you already have the isolated worktree, fetch and check out the new branch there:

```bash
cd /home/bbx/user_lx/GraphLoRA_collab_v1
git fetch https://github.com/NanDingEi/GraphLoRA-Research.git exp/collab-injection-e3
git checkout -B exp/collab-injection-e3 FETCH_HEAD
```

If this worktree is detached or contains uncommitted edits, back up those edits first. In particular, do not overwrite local logs or configs you need.

## 2. Generate matched configs from the actual S2-L31 checkpoint

```bash
cd /home/bbx/user_lx/GraphLoRA_collab_v1
python experiments/collab_injection_v1/prepare_e3_configs.py
```

Default checkpoint:
`/home/bbx/user_lx/GraphLoRA/logs/S2_STRICT_20261017_143611/L31_output/20261017143/checkpoint_best.pth`

If the checkpoint is stored elsewhere, pass `--checkpoint /absolute/path/checkpoint_best.pth`.

The script copies the saved config, changes only the experiment mode/required scales, disables conflicting relation injection flags, sets seed 42 and 3 epochs, and gives each run a separate output directory. It clears `model.ckpt` and `run.resume_ckpt_path`; training starts from the configured pretrained model/LoRA initialization rather than resuming the champion's trained weights. Other settings, including warmup, semantic alignment, projector regularization, dataset paths, and learning rates, are preserved from the checkpoint config unless explicitly noted in the script.

Generated configs:
- `tran_configs/collab_v1/e3/S2_L31_CONTROL.yaml`
- `tran_configs/collab_v1/e3/S2_GATED_L31.yaml`
- `tran_configs/collab_v1/e3/S2_CONDITIONAL_LORA.yaml`

## 3. Start the guarded serial runner

Check GPU availability first:

```bash
nvidia-smi
pgrep -af '[t]rain_collm_mf_din.py' || true
```

Then run:

```bash
bash experiments/collab_injection_v1/run_e3.sh
```

The runner:
1. Runs a 12-mini-batch Gated-L31 preflight first.
2. Checks that `collab_gate` parameters were saved and changed. If not, it stops before the long runs.
3. Runs control, gated L31, and conditional LoRA serially, 3 epochs each.
4. Checks that the custom parameters are present and nonzero in the best checkpoint.
5. Writes logs to `logs/collab_v1/` and checkpoints to distinct `output/*_E3/` directories.

The preflight's validation metrics are diagnostic only. Full-run model selection must use validation AUC/UAUC. Do not select a checkpoint based on test metrics; evaluate test only after selecting the checkpoint by validation.

## Important

- The preflight only verifies that the gate can update; it is not a performance result.
- A nonzero parameter norm is necessary but not sufficient evidence of a correct learning signal.
- If the runner stops with a gate/conditioner update failure, preserve the log and checkpoint; do not report that run as a valid method comparison.
- The runner does not automatically evaluate the test split.
- No GPU training has been executed by GitHub; all runs happen on your machine.
