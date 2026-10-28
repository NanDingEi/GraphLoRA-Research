# NIGHT21 — Loss ablations on the healthy NIGHT17 CoRA E0

**Do not use NIGHT20 T1 or any of the failed prototype/fine-residual arms.**
All four experiments use the original NIGHT17 E0_UNIFORM_4LAYER_CONTROL
architecture and the prepared historical baseline config.

Historical E0 Valid: AUC=0.7299284744316927, UAUC=0.6866122315446266.
These are recorded reference values, not rerun or Test results.

## Four trainable arms (one loss removed per run)

| Arm | CTR | KD | Semantic | Projector cosine preservation |
|---|---:|---:|---:|---:|
| E0 historical (NOT rerun) | .05 | .05 | .0001 | .0005 |
| A1_NO_CTR | 0 | .05 | .0001 | .0005 |
| A2_NO_KD | .05 | 0 | .0001 | .0005 |
| A3_NO_SEM | .05 | .05 | 0 | .0005 |
| A4_NO_PROJ | .05 | .05 | .0001 | 0 |

- A1 retains trainable `gnn_conf_kappa` for KD instead of accidentally
  changing the model architecture.
- A3 keeps the semantic teacher forward so its random-consumption path
  stays the same; **only the semantic loss weight is zero**.
- A4 computes the same randomly sampled 64-item projector structure
  statistic but gives it zero loss weight, likewise preserving sampling.
- Projector regularization is on `llama_proj` (MF item-item cosine
  similarity preservation), not CoRA's group8 projector.
- KD is LLM -> GNN with stopped LLM-teacher gradients, not GNN -> LLM.
- No NIGHT19 output residual, NIGHT20 prototype branch or layer31 GNN
  embedding injection. The baseline group8 CoRA mapping still runs.

Common: only GPU0, seed42, 1 epoch per trial, foreground serial, Valid only,
Vicuna-7B, frozen pretrained MF pair, full L0-31 Q/K/V/O, group8 (eight
independent projectors each serving four layers), rank16 alpha48, BS2
accum4, original optimizer/LR and input mixing disabled. Fresh initialization
for each trial, no resumed experiment checkpoints.

## Isolated server commands

    cd /home/bbx/user_lx/GraphLoRA_NO31_CORA
    git fetch https://github.com/NanDingEi/GraphLoRA-Research.git exp/night21-cora-e0-loss-ablation-20261027
    git worktree add --detach /home/bbx/user_lx/GraphLoRA_NIGHT21 FETCH_HEAD
    cd /home/bbx/user_lx/GraphLoRA_NIGHT21
    conda activate minigpt4

The original prepared E0 YAML is an untracked runtime asset. Copy **only**
the YAML, no checkpoint, into the isolated worktree:

    mkdir -p logs/NO31_CORA_ORIGINAL_READY
    cp /home/bbx/user_lx/GraphLoRA_NO31_CORA/logs/NO31_CORA_ORIGINAL_READY/train.yaml logs/NO31_CORA_ORIGINAL_READY/train.yaml

CPU/config checks:

    python experiments/no31_cora_original/test_night21_cpu.py
    python experiments/no31_cora_original/night21.py --plan

Run all four sequentially on GPU0, **in foreground**:

    CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/night21.py --run

Or execute one experiment in isolation:

    CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/night21.py --only A1_NO_CTR

Continue from another arm in a **new output session**:

    CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/night21.py --start A2_NO_KD

Note: --run, --start and --only are mutually exclusive modes. A repeat
does NOT resume weights; each arm independently initializes fresh.

## Logs

    logs/NO31_CORA_NIGHT21_<timestamp>/
      A1_NO_CTR/train.yaml
      A1_NO_CTR/train.log
      A1_NO_CTR/result.json
      ...
      summary.csv
      results.jsonl
      plan.json

Model checkpoints are saved under output/NO31_CORA_NIGHT21_<timestamp>/.
Each summary row records the four applied loss coefficients and
delta_vs_healthy_e0_auc/uauc.

Required logs include:

- [NIGHT21-LOSS-CONFIG] actual active scalar coefficients.
- [NIGHT21-GRAD-PATH] numbers of trainable parameters in CoRA generator,
  GNN and llama_proj, and whether kappa is trainable.
- [NIGHT8-LOSS] raw L_rec, L_gctr, L_gkd.
- [semantic] raw semantic loss and effective corr_w.
- [projector-reg] raw cosine MSE and weighted contribution (0 for A4).
- [NIGHT21-LOSS-AUDIT-PASS] per-arm log consistency.
- [NIGHT8-STOP] exactly one trained epoch.
- [NIGHT21-E0-DELTA] Valid AUC/UAUC vs historical normal E0.

**Interpretation caveat:** logged parameter trainability and auxiliary loss
values do not by themselves prove a nonzero gradient to a particular LLM
module. NO31 gnn_scale=0 and input alpha=0 prevent the GNN from injecting
into LLM hidden states. If CTR/KD/semantic removal changes Valid but does
not have a direct gradient path to LLM, check RNG/order or shared parameters
before attributing causal improvement. Seed42 provides one run, not a
confidence interval.

**Status:** committed on GitHub; no assistant-side GPU/CPU runs executed.
