# Collaborative injection: guarded 3-epoch experiments

Fix branch: `fix/collab-gated-optimizer` (based on `exp/collab-injection-e3`).

## What changed after the failed preflight

The previous log showed LoRA-B gradients but zero saved LoRA-B/gate parameters. That alone did not prove the cause, so this branch adds optimizer-boundary diagnostics to report:
- GradScaler scale before/after each optimizer update;
- non-finite gradient tensor/value counts and GradScaler's `found_inf` flag;
- before/after norms and exact max absolute parameter deltas for layer-31 LoRA-B, gate, and conditioner parameters.

The E3 configs now set `amp_init_scale: 1024` instead of PyTorch's default 65536, to reduce early fp16 loss-scaling overflow risk. The default for all other runs is unchanged. This is a targeted mitigation, not considered verified until the preflight log shows actual parameter deltas and no skipped optimizer updates.

The runner now stops after a successful preflight by default. Full training requires explicit opt-in.

## 1. Pull the fix branch

```bash
cd /home/bbx/user_lx/GraphLoRA_collab_v1
git fetch https://github.com/NanDingEi/GraphLoRA-Research.git fix/collab-gated-optimizer
git checkout -B fix/collab-gated-optimizer FETCH_HEAD
```

If this worktree contains uncommitted changes, back them up before switching branches.

## 2. Generate matched configs

```bash
python experiments/collab_injection_v1/prepare_e3_configs.py
```

Default checkpoint used only as a source of configuration:
`/home/bbx/user_lx/GraphLoRA/logs/S2_STRICT_20261017_143611/L31_output/20261017143/checkpoint_best.pth`

The generator copies the saved configuration, then clears `model.ckpt` and `run.resume_ckpt_path`. It does not resume trained champion weights. Dataset paths, model initialization, learning rates, semantic alignment, projector regularization, and other saved settings are preserved unless the script explicitly changes them.

## 3. Run only the guarded preflight first

```bash
nvidia-smi
pgrep -af '[t]rain_collm_mf_din.py' || true
bash experiments/collab_injection_v1/run_e3.sh
```

The script runs a 12-mini-batch gated preflight with optimizer diagnostics. Logs are written to `logs/collab_v1/SMOKE_GATED_PREFLIGHT.log`. It exits without launching the full 3-epoch experiments.

Inspect the update records:

```bash
grep -E '\[AMP\]|\[OPT-UPDATE-DIAG\]|\[FAIL\]|\[PASS\]' \
  logs/collab_v1/SMOKE_GATED_PREFLIGHT.log
```

Interpretation:
- `found_inf=1`, non-finite gradients, and a falling scale indicate GradScaler skipped the update because of overflow.
- `found_inf=0` but zero deltas for all watched parameters indicate the problem is elsewhere in the optimizer/parameter wiring.
- LoRA-B should move after the first optimizer update; the gate may only start moving after LoRA-B becomes nonzero, so inspect multiple update boundaries.
- The preflight AUC/UAUC is diagnostic only, not a method result.

## 4. Run the full three-way comparison only after the preflight is healthy

```bash
RUN_FULL_E3_AFTER_PREFLIGHT=1 \
  bash experiments/collab_injection_v1/run_e3.sh
```

This reruns the guarded preflight, then runs control, gated L31, and conditional LoRA serially for 3 epochs each. Outputs are isolated under `output/*_E3/`; logs are under `logs/collab_v1/`.

Use validation AUC/UAUC to select checkpoints. Test evaluation remains a separate step after checkpoint selection.
