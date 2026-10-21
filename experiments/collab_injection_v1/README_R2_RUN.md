# R2 8-Round Execution: GraphLoRA internal collaborative attention

Implementation commit series on branch `fix/collab-gated-optimizer`.
Unlike the earlier R2 design-only README, these are **actual scripts**.

## Files

- `r2_collab_runtime.py`: Opt-in runtime hooks, no core-model source edit.
- `r2_01_cpu_score_audit.py`: CPU-only analysis of existing R1 archive.
- `r2_probe.py`: R2-02 to R2-05: same checkpoint, fixed 512 Valid samples.
- `r2_runtime_entry.py`: Required patched model entry for R2 training.
- `r2_train.py`: independent 24-step smoke and one full epoch for 06/07/08.
- `test_r2_cpu.py`: shape/role/weighting invariant tests.
- `run_r2_eight_foreground.sh`: sequential driver, aborts at first error.

## Before running

```bash
cd /home/bbx/user_lx/GraphLoRA_collab_v1
git status --short
git fetch https://github.com/NanDingEi/GraphLoRA-Research.git fix/collab-gated-optimizer
git merge --ff-only FETCH_HEAD
conda activate minigpt4
export CUDA_VISIBLE_DEVICES=0
```

**Do not merge if git status shows uncommitted changes or merge cannot
fast-forward.** No experiment modifies historical S2, R1, or P2B results.

P2B source config expected at:
`logs/collab_v1/S2_L30_AUX_P0_20261021_165700/P2B_S2_CACHE_FIXED_FULL.yaml`.
Source checkpoint must resolve to:
`/home/bbx/user_lx/GraphLoRA/logs/S2_STRICT_20261017_143611/L31_output/20261017143/checkpoint_best.pth`.

Original R1 archive expected at:
`logs/collab_v1/S2_R1_QV_ABLATION.npz`.

Create S2 evaluation config if missing:
```bash
test -f tran_configs/collab_v1/e3/S2_L31_CONTROL.yaml || \
  python experiments/collab_injection_v1/prepare_e3_configs.py
```

## Single command (foreground, strictly serial)

```bash
bash experiments/collab_injection_v1/run_r2_eight_foreground.sh
```

The launcher checks Python syntax and CPU tensor/role invariants;
R2-01 does not use GPU; R2-02 to R2-05 collect 512 Valid predictions
without training, recording actual pathway activations and paired native
Yes logits; R2-06 to R2-08 each run a fresh 24-step smoke and then one
full epoch starting from the SAME original S2 checkpoint. They do not
inherit optimizer weights from previous arms. Every training arm enables
the P2-A autocast cache fix. No Test runs.

### Manually execute one slot at a time instead

```bash
python -m py_compile experiments/collab_injection_v1/r2_*.py
python experiments/collab_injection_v1/test_r2_cpu.py
python -u experiments/collab_injection_v1/r2_01_cpu_score_audit.py
python -u experiments/collab_injection_v1/r2_probe.py --round 02
python -u experiments/collab_injection_v1/r2_probe.py --round 03
python -u experiments/collab_injection_v1/r2_probe.py --round 04
python -u experiments/collab_injection_v1/r2_probe.py --round 05

# IMPORTANT: only after each probe passes predeclared pathway gates
python -u experiments/collab_injection_v1/r2_train.py --round 06
python -u experiments/collab_injection_v1/r2_train.py --round 07
python -u experiments/collab_injection_v1/r2_train.py --round 08
```

## Five predeclared probes

| Round | Runtime mode(s) | Outcome |
| --- | --- | --- |
| 01 | CPU ON vs OFF from R1 | Yes-logit movement distribution |
| 02 | audit | actual graph rank/V projection norm distributions |
| 03 | off vs decision_q | last-Prompt Q perturbation sensitivity |
| 04 | off vs history_k | pre-RoPE historical K perturbation sensitivity |
| 05 | off vs relation_v vs relation_shuffle | correct vs wrong candidate graph relation |

Here `off` means **R2 modification disabled**, NOT R1's GNN
ON/OFF switch. It preserves **original S2 L31 Q+V** injection.

## Three matched training experiments

| Round | Mode | Difference from original S2 |
| --- | --- | --- |
| 06 | relation_v | per-history-item V graph feature weighted by candidate-conditioned normalized graph similarity |
| 07 | relation_v_q OR relation_v_raw | add decision-Q residual if R2-03 native score response > predeclared threshold, else test no mean normalization as fallback |
| 08 | relation_shuffle | same as 06 but substitute mismatched candidate GNN representation ONLY to compute V history weights |

No parameters are added. Existing GNN + LoRA optimize from S2 with
native BCE and same original P2B schedule. Each experiment outputs a
unique config and manifests under
`logs/collab_v1/R2_EIGHT_20261022/R2_XX/`.
Model weights go under
`output/R2_EIGHT_20261022/R2_XX/FULL_E1/.../checkpoint_best.pth`.

The model's `model.r2_mode` YAML key is experiment **metadata**;
the vanilla model does NOT restore runtime hooks from this metadata.
**Evaluate R2-trained checkpoints via `r2_runtime_entry.py` with
matching `--mode`**. Evaluating with vanilla
`train_collm_mf_din.py` would silently compute different predictions.

## Guardrails / reporting

- Initial 512 baseline for 03/04/05 must match R1's first 512 IDs,
  item IDs, labels, and Yes logits (max absolute tolerance 2e-4).
- R2-02 must observe >0 V injection tokens.
- R2-05 must produce >0 changed Yes predictions and meaningful
  correct-vs-mismatched candidate distinction, else stop.
- The training controller refuses input configurations with wrong
  checkpoint, wrong S2 mode, wrong LR, seed, batch, accumulation or
  missing FX autocast fix. It refuses existing run folders.
- Smoke must report actual candidate-weighted V activation. Full
  training must yield one valid AUC/UAUC and checkpoint.
- No Valid-based threshold tuning on a 512-subset, no Test leakage,
  no automatic retries or background jobs.
- 1-epoch Valid deltas are exploratory, not statistically established.

Expected output lines:
`[R2-01-COMPLETE]`, `[R2-02-COMPLETE]`, ..., `[R2-05-COMPLETE]`,
`[R2-06-RESULT]`, `[R2-07-RESULT]`, `[R2-08-RESULT]`.

**Implementation was authored against GitHub repository source but has
not been executed on the user's 3090 server.** Before trusting any
result, verify the CPU preflight, R1 anchor and 24-step runtime smoke.
