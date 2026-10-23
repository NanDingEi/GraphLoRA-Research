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

## Provenance / verification

The only changes beyond the archival NO31 patch to S2 are:
1. A guarded `MyLinear.forward` CoRA hook, inactive in the NO31 control.
2. A new CoRA adapter module/entry/runner and CPU zero-init tests.
3. Separate unchanged-parameter initialization fingerprints.

Do not compare the historical Test AUC `0.733936` directly with a
new Valid AUC. The standard original NO31 E8 epoch0 Valid reference was
approximately 0.719760 / UAUC 0.685106.
