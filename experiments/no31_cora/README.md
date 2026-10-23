# Archived NO31 + CTR/KD + CoRA

## What is preserved

This branch is based on the **original GraphLoRA S2 commit**
`d818fa2943ede4e9a810b032a3422e94d0c77ed0`.
The historical **NO31** source transformations were extracted unchanged
from the original script
[`run_S2_NO31_night8_serial.sh`](https://github.com/NanDingEi/GraphLoRA-Research/blob/exp/layer-injection-study-20261016/run_S2_NO31_night8_serial.sh)
in archival commit `74c7e3ce2d17e1ab428a73175effc941062d22a0`.
The historical patched code is now committed directly into
`minigpt4/models/minigpt4rec_v2.py` and
`minigpt4/runners/runner_base.py`, **so NO external source_patch.diff is needed**.

The archival training mechanism remains:
- NO layer-31 collaborative feature injection: `model.gnn_scale=0.0`.
  LoRA at L31 remains normally trainable.
- No input embedding fusion: `S2_INPUT_ALPHA=0`.
- **GNN CTR** hard-label BCE loss weight `0.05`.
- **LLM→GNN KD** soft-label BCE loss weight `0.05`,
  with teacher Yes logit detached before GNN supervision.
- Original GNN semantic alignment weight `1e-4`,
  projector regularization weight `5e-4`, pretrained MF checkpoint,
  original Vicuna-7B, original data and graph files.
- Original S2 cosine schedule max_epoch=3, batch3, gradient accumulation3,
  seed42, group LR ratios MF0.5/GNN0.25/Projector0.5/LoRA1.
- Original NO31 1- or 2-full-epoch stop implemented in the runner,
  without changing the 3-epoch learning-rate schedule.

Old NO31 result to match is `R5_E8_CTR_KD_2EPOCH` (Valid metrics are
different from its Test metrics). **No new GPU results are claimed**.

## Added CoRA

The CoRA implementation lives in `experiments/no31_cora/adapter.py`.
It reuses the same collaborative query generator (learnable queries,
self-attention, user/item cross-attention, rank-8 conditioning).
Q/V up-projections are initialized to **exactly zero**, so at
initialization the CoRA correction is zero.

Unlike the previous S2-CoRA which added an extra low-rank residual to
**all prompt tokens**, this version adds it **only at ID-token positions**:

```text
original NO31:       LoRA-B(LoRA-A(H)), no direct L31 GNN injection
NO31 + CoRA:         original NO31 + gain * CoRA-Up(s(user,item) * LoRA-A(H_ID))
default gain:        0.1
default GNN grad:    DETACHED from CoRA (GNN still learns from original CTR/KD/sem)
CoRA LR scale:      0.25 (relative to S2 scheduler; its own optimizer group)
```

This is an **experimental CoRA-inspired conditional LoRA adaptation**,
not a claim to exactly reproduce a published CoRA architecture.

## Two independent experiment arms

| Arm | NO31 historical CTR/KD | CoRA |
|---|---|---|
| `no31` | yes | off |
| `cora` | yes | ID-only L31 Q+V, gain=.1, condition.detach() |

Both use **identical YAML** except `output_dir`, identical seed42, same
pretrained MF weights and new LoRA/NGCF/projector initializations.
Before any full training they compare SHA256 of their MF/GNN/projector/LoRA
shared initial weights. A mismatch stops the training.

Default: **1 complete epoch per arm**, preserving 3-epoch scheduler horizon;
Valid AUC/UAUC only. Set `NO31_EPOCHS=2` to run 2 per arm.
Both configurations automatically record checkpoint(s) and the best Valid
checkpoint. The model and the trained checkpoint are not resumed from S2.

## Download and launch on original GPU0 server

```bash
cd /home/bbx/user_lx/GraphLoRA_collab_v1

git fetch https://github.com/NanDingEi/GraphLoRA-Research.git \
  exp/no31-ctrkd-cora-20261023

git worktree add -b no31-cora-local-20261023 \
  /home/bbx/user_lx/GraphLoRA_NO31_CORA FETCH_HEAD

cd /home/bbx/user_lx/GraphLoRA_NO31_CORA
conda activate minigpt4

# Validate and create both YAML configs; no GPU training.
bash experiments/no31_cora/run.sh prepare

# Serial one-epoch NO31 control + CoRA, GPU0, foreground only.
bash experiments/no31_cora/run.sh run
```

Or from a fresh worktree: `bash experiments/no31_cora/run.sh all`.

For 2 epochs per arm, use `NO31_EPOCHS=2` on BOTH
`prepare` and `run`.

The original NO31 optimizer groups (MF/GNN/Projector/LoRA) are unchanged.
The additional `r3_adapter.*` parameters are assigned their **own**
`cora` LR group (multiplier `0.25`, configurable via
`run.lr_scale_cora`). Without CoRA, no such parameter group is created.
CPU checks verify zero-initialized identity, ID-only locality and
nonzero Q/V up-projection gradients.

The prepare step checks the historical MF/Vicuna paths and uses the
original dataset graph NPZ files. It creates symlinks in the isolated
worktree if required, checking their exact SHA256 against the originals.

Training is fail-closed: original NO31 baseline epoch0 AUC below 0.69
blocks the CoRA run. Results of interrupted runs are **never overwritten**
and no process starts in the background.

Results go to:
- `logs/NO31_CTRKD_CORA_20261023/{no31,cora}_train.log`
- `logs/NO31_CTRKD_CORA_20261023/{no31,cora}_result.json`
- `output/NO31_CTRKD_CORA_20261023/`

## Resume the finished CoRA epoch0 for exactly one more epoch

The first NO31+CTR/KD+CoRA experiment produced:
`Valid AUC=0.635651530`, `Valid UAUC=0.595876840` at epoch0.
The original NO31 CTR/KD control achieved `Valid AUC=0.719760366`.
The CoRA run has **completed** epoch0 and saved
`output/NO31_CTRKD_CORA_20261023/cora/**/checkpoint_0.pth`.

To continue **only the CoRA model** for **epoch1** (no rerun of epoch0,
no new NO31 control training), run this in the existing worktree:

```bash
cd /home/bbx/user_lx/GraphLoRA_NO31_CORA
git fetch https://github.com/NanDingEi/GraphLoRA-Research.git exp/no31-ctrkd-cora-20261023
git merge --ff-only FETCH_HEAD
conda activate minigpt4

# Read and verify checkpoint epoch, CoRA params, AdamW moments, AMP scaler,
# historical CTR/KD, seed and cosine schedule: no training.
bash experiments/no31_cora/resume_one.sh --dry-run

# Resume exactly epoch 1 on GPU0, foreground.
bash experiments/no31_cora/resume_one.sh
```

Script: `experiments/no31_cora/resume_one.py`.
It sets `run.resume_ckpt_path` to the original `checkpoint_0.pth`,
and `S2_NIGHT_EPOCHS=2` while preserving `run.max_epoch=3`.
The runner restores the actual trainable model params, optimizer AdamW
moments, AMP GradScaler, and starts at checkpoint epoch+1 = 1.
**Setting NO31_EPOCHS=2 alone would retrain from scratch; do not do that.**

New output and log roots use unique timestamps:
`output/NO31_CTRKD_CORA_RESUME_E1_<timestamp>/` and
`logs/NO31_CTRKD_CORA_RESUME_E1_<timestamp>/`.
The original epoch0 checkpoint and logs are never modified. The script
verifies the saved new `checkpoint_1.pth` and prints its Valid AUC/UAUC
relative to the original epoch0.

**Reproducibility note**: this checkpoint does **not** include the
Python/NumPy/PyTorch random-number-generator states or dataloader state.
The resumed epoch uses restored model/optimizer/scaler and the correct
epoch1 learning rate, but it is not a bitwise-identical continuation of
an uninterrupted 2-epoch run. It should be considered a genuine
optimizer-state resume, not an exact RNG replay.

## Resume after the fixed probe-only counter bug

If you ran the initial `run` launcher and it stopped immediately after
`[NO31-PROBE-OK] mode=cora` with
`RuntimeError: CoRA branch q never executed`, **there was no training**.
The cause was a training-only counter assertion after an init-only probe.
It is fixed: the probe exits without checking Q/V forwards, and the runner
safely reuses the original NO31 and CoRA probe hashes. Only the **exact**
legacy false-failure signature is recognized; other tracebacks stop the run.

From the already created `GraphLoRA_NO31_CORA` worktree:

```bash
cd /home/bbx/user_lx/GraphLoRA_NO31_CORA
git status --short
git fetch https://github.com/NanDingEi/GraphLoRA-Research.git exp/no31-ctrkd-cora-20261023
git merge --ff-only FETCH_HEAD
conda activate minigpt4
CUDA_VISIBLE_DEVICES=0 bash experiments/no31_cora/run.sh run
```

Do **not** run `prepare` again, delete any existing probe logs, or use
`git reset --hard`. The launcher validates and reuses the existing hashes.
`test_probe_resume.py` checks this CPU-side, including rejection of
unrelated failures.

## Provenance / verification

The only changes beyond the archival NO31 patch to S2 are:
1. A guarded `MyLinear.forward` CoRA hook, inactive in the NO31 control.
2. A new CoRA adapter module/entry/runner and CPU zero-init tests.
3. Separate unchanged-parameter initialization fingerprints.

Do not compare the historical Test AUC `0.733936` directly with a
new Valid AUC. The standard original NO31 E8 epoch0 Valid reference was
approximately 0.719760 / UAUC 0.685106.
