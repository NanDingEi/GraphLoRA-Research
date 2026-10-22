# S2 rollback + minimal CoRA

## Archived starting point

- Parent commit: d818fa2943ede4e9a810b032a3422e94d0c77ed0 (NOT R4/R5).
- Historical S2: logs/S2_groups_acc3.log and tran_configs/S2_groups_acc3_recovered.yaml.
- Valid AUC 0.729675 (epoch 0), 0.732884 (epoch 1); epoch-1 UAUC 0.687953.
- The reconstructed YAML reflects the logged effective parameters, not an exact archival copy of the original invocation; actual baseline reproduction remains to be verified.
- MF initialized from its historical pretrained checkpoint. NGCF, Projector and LLM LoRA are newly initialized and optimized.
- No GraphLoRA S2 checkpoint or R4/R5 checkpoint is loaded.
- batch_size_train=3, accum_grad_iters=3, seed=42, lr=1e-4, 3-epoch cosine horizon.
- lr_scale_rec=0.5, lr_scale_gnn=0.25, lr_scale_proj=0.5, lr_scale_lora=1.0.
- gnn_semantic_weight=1e-4 and proj_reg_weight=5e-4, gcn_layer=31, gnn_scale=0.1.
- Original fix_autocast_cache default (false) is preserved; no silent optimizer or teacher changes.

## Exact code change scope

- Base model minigpt4/models/minigpt4rec_v2.py: only a guarded NGCF penultimate feature export for active CoRA; without CoRA hooks the baseline executes the S2 branch.
- experiments/s2_cora/s2_cora_adapter.py: CoRA-inspired extra zero-init conditional low-rank B_c diag(s_user_item) A path on L31 Q and V; original S2 GNN-LoRA pathway is retained.
- The optimizer, runner, scheduler, tasks, tokenizer, MF, GNN architecture and original loss implementation remain exactly at the parent commit.
- The CoRA implementation is an R3-inspired adapter, not a reproduction of the full published CoRA training protocol.

## Run without touching the existing R4/R5 working directory

```bash
cd /home/bbx/user_lx/GraphLoRA_collab_v1
git fetch https://github.com/NanDingEi/GraphLoRA-Research.git exp/s2-rollback-cora-minimal-20261022
git worktree add -b s2-cora-local-20261022 /home/bbx/user_lx/GraphLoRA_S2_CORA FETCH_HEAD
cd /home/bbx/user_lx/GraphLoRA_S2_CORA
conda activate minigpt4
bash experiments/s2_cora/run.sh prepare
```

Prepare does not train. It checks the saved S2 effective recipe, MF checkpoint, original graph NPZ SHA256, compiles Python files, runs a CPU-only CoRA zero-init test and creates two identical experiment YAMLs apart from output_dir.

## Manual opt-in experiment stages, all foreground on GPU0

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/s2_cora/run.sh probe
CUDA_VISIBLE_DEVICES=0 bash experiments/s2_cora/run.sh baseline
CUDA_VISIBLE_DEVICES=0 bash experiments/s2_cora/run.sh cora
```

- probe: build each model but do not train; compare MF/GNN/Projector/LoRA parameter SHA256.
- baseline: train S2 with recorded schedule; stop the sequence if best Valid AUC < 0.72.
- cora: train the only extra CoRA arm with exactly the same original config and initialization; allowed only after successful S2 baseline and parameter-parity probes.
- Each run uses an independent output directory and never overwrites historical S2/R4/R5 checkpoints.
- Running all three train phases takes multiple hours; nothing is scheduled or launched automatically by this GitHub change.
