# Tonight: 9 DISTINCT model structures, fixed seed 42, TWO full epochs each

## Immutable historical S2 training recipe

Archived source: `d818fa2943ede4e9a810b032a3422e94d0c77ed0`.
Historical valid S2: Epoch 0 AUC 0.729675 / UAUC 0.684167;
Epoch 1 AUC 0.732884 / UAUC 0.687953.

All 9 arms retain **seed 42**, pretrained historical MF, original Vicuna,
NGCF, Projector regularization weight 5e-4, semantic loss weight 1e-4,
batch 3, gradient accumulation 3, subgroup LRs
(MF 0.5 / GNN 0.25 / Projector 0.5 / LoRA 1.0),
gcn_layer 31, gnn_scale 0.1 and matching original data.
No S2/R4/R5 joint model checkpoint. Model configuration never uses
a different seed or pretrained recommender for CoRA.

`max_epoch=3` for the original cosine LR horizon and
`stop_after_n_epochs=2` to stop after completing Epoch 0 AND Epoch 1.
Both epochs are evaluated on Valid, and `checkpoint_0.pth`,
`checkpoint_1.pth`, `checkpoint_best.pth` are required.
No Test-set evaluation. Runs execute SERIAL and in the foreground on GPU0.

## Nine distinct methods, not random-seed experiments

| ID | Method | CoRA target | Conditioning context / mechanism |
| --- | --- | --- | --- |
| 01_S2_S42 | Reproduced S2 baseline | none | Original S2, hard gate best Valid AUC >= 0.72 |
| 02_CORA_QV | Full CoRA | Q and V | Query self-attn + user/item cross-attn |
| 03_CORA_Q | Q-only | Q | Does learning collaboration through Q alone help? |
| 04_CORA_V | V-only | V | Is modifying collaborative value representation enough? |
| 05_CORA_HALF | Low-strength CoRA | Q and V | Residual gain 0.5 instead of 1.0 |
| 06_CORA_USER | User-only conditioning | Q and V | Only user collaborative context, item is zero-masked |
| 07_CORA_ITEM | Item-only conditioning | Q and V | Only target-item context, user is zero-masked |
| 08_CORA_INTERACT | Interaction-aware conditioning | Q and V | Additional elementwise user×item interaction token |
| 09_CORA_NO_CROSS | No cross-attention | Q and V | Replace learned cross-attention output with pooled CF context |

A full CoRA `B_c diag(s_ui) A` correction is initially zero,
so the original GraphLoRA S2 output is not replaced.
Feature/attention and Q/V ablations use the same model baseline, seed and
optimizer schedule. Each CoRA arm checks exact SHA256 equality of
the pretrained MF, NGCF, projector and LoRA initialization against S2
**before training**, again during training start. Any mismatch fails closed.

## Recommended isolated working tree and launch

```bash
cd /home/bbx/user_lx/GraphLoRA_collab_v1
git fetch https://github.com/NanDingEi/GraphLoRA-Research.git exp/s2-rollback-cora-minimal-20261022

# If you have not created the separate S2 worktree yet:
git worktree add -b s2-cora-local-20261022 \
  /home/bbx/user_lx/GraphLoRA_S2_CORA FETCH_HEAD

cd /home/bbx/user_lx/GraphLoRA_S2_CORA
conda activate minigpt4
bash experiments/s2_cora/run_night9.sh prepare
CUDA_VISIBLE_DEVICES=0 bash experiments/s2_cora/run_night9.sh run
```

If that worktree already exists and is clean, update it with
`git -C /home/bbx/user_lx/GraphLoRA_S2_CORA pull --ff-only`
after checking that its local branch tracks the correct remote branch.
Do not reset or overwrite active experiments.

Alternative for a NEW experiment worktree only:
`CUDA_VISIBLE_DEVICES=0 bash experiments/s2_cora/run_night9.sh all`.

The **first** S2 experiment must reproduce best Valid AUC >= 0.72
(or no CoRA arm will run). The rest are one-to-one fixed-seed mechanism
comparisons; a good CoRA AUC by itself is not meaningful without S2.

## Results and practical limits

- Per-arm full raw logs: `logs/S2_CORA_NINE_20261022/<ID>.log`
- Per-arm initialization parity preflight: `<ID>.preflight.json`
- Both-epoch AUC/UAUC and best-epoch comparison: `summary.csv`
- Checkpoints: `output/S2_CORA_NINE_20261022/<ID>/<job_id>/checkpoint_{0,1,best}.pth`

Nine methods × two full epochs = **18 training epochs**
(11,297 microbatches per epoch, plus validation). On one RTX 3090,
the whole schedule may run beyond one night. The scripts do not create
background tasks; if the process stops, completed arms can be skipped,
while partial runs require deliberate recovery and are never overwritten.
