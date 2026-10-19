# Collaborative injection experiments (v1)

Branch: `exp/collab-injection-v1`

The code in `minigpt4/models/minigpt4rec_v2.py` is **already modified** on this branch. Do not run `apply_collab_injection.py` after checking out the branch (it would attempt to patch a second time).

## Isolated checkout on server

```bash
cd /home/bbx/user_lx/GraphLoRA
git fetch https://github.com/NanDingEi/GraphLoRA-Research.git exp/collab-injection-v1
git worktree add --detach ../GraphLoRA_collab_v1 FETCH_HEAD
cd ../GraphLoRA_collab_v1
python -m py_compile minigpt4/models/minigpt4rec_v2.py
```

The detached worktree allows evaluation without changing the existing S2 checkout. Use separate output directories and the **validated S2-L31 configuration** as the common control.

In a copy of your S2 L31 YAML, set model keys:

* Control: `collab_mode: legacy`, `gnn_scale: 0.1`, `gcn_layer: 31`.
* Gated L31: `collab_mode: gated_l31`, `gnn_scale: 0.1`, `gcn_layer: 31`.
* Conditional LoRA: `collab_mode: conditional_lora`, `gnn_scale: 0.0`, `gcn_layer: 31`.

For both new modes set `rel_inject: false` and `rel_rank_mod: false` to avoid confounding them with earlier experiments.

## Important pre-training checks

1. Verify the `collab_gate` / `collab_conditioner` parameters are included in the optimizer.
2. Run a forward/backward mini-batch and inspect parameter gradients; run one optimizer update and check nonzero parameter changes.
3. Verify the checkpoint save **and load** paths preserve the two new modules. PEFT-only adapter saves may omit them.
4. Confirm the legacy mode behaves identically to the unmodified branch with the same inputs/seeds.
5. Confirm that `gnn_scale: 0.0` for the conditional experiment *does not* prevent GNN features being constructed; this pathway reads `gcn_output`.
6. Compare validation AUC/UAUC per epoch, never use Test to pick hyperparameters/checkpoints.

No remote training or evaluation was performed by this commit. GitHub `main` was not modified.

## Known limitation

Current `inj_diag` logs the pre-gating injection ratio, so inspect effective gate outputs separately before interpreting its values.
