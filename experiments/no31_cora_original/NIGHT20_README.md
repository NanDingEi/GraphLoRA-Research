# NIGHT20: four trainable coarse/fine CoRA experiments

This branch implements T1–T4 for **Vicuna-7B + original CoRA group8**. It
continues the user's NIGHT17 E0 / NIGHT18–19 research but introduces
collaborative conditions **before group8 QFormer-state projectors**, not as
output-side dynamic-LoRA-A additive gates.

No baseline retrain, no Test split. The four arms start from the same original
Vicuna + pretrained MF state and fresh zero-CoRA-P initialization.

## Four trainable arms

| ID | Collaborative QFormer state | Group schedule |
|---|---|---|
| T1_PROTO_COARSE_ONLY | user MF + K16 prototype + coarse MF score + leave-target-out smoothed user/group preference | coarse=1, fine=0, all 8 groups |
| T2_PROTO_FINE_ONLY | candidate prototype residual + within-group positive historical residual + user-specific fine MF score | coarse=0, fine=1, all 8 groups |
| T3_PROTO_DUAL_UNIFORM | both coarse and fine | coarse=.5, fine=.5, all 8 groups |
| T4_PROTO_DUAL_C2F | both, progressive layer routing | coarse=[1,1,.75,.75,.25,.25,0,0], fine=1-coarse |

Common: CoRA Q/K/V/O full L0–31, rank=16, alpha=48, group8 (each group 4
Transformer layers), seed 42, NO31 GNN insertion off, CTR=.05 KD=.05, original
optimizer/batch/LR schedule (1 trained epoch of a 3-epoch LR horizon), GPU0
**serial, foreground**, Valid only.

The original frozen [user; target] MF pair still enters the released CoRA
QFormer unchanged. Both additional small mapper heads exist in all T1–T4 arms
(identical extra parameter count) and their final projection layers are
zero-initialized. New query-state additions are independently norm-clipped to
at most 10% of the unmodified QFormer query-state norm. At initialization,
the combined CoRA dynamic A is exactly the original group8 E0 path.

Training histories are computed only from train_ood2.pkl and exclude **all**
records of the current (user, target) pair from user–group counts and positive
history aggregates. Inference uses the same target-excluded rule. The group
partition and MF embeddings were learned from training data separately.

## Important input requirements

- K=16 NPZ from completed CPU prototype probe. Default:
  /home/bbx/user_lx/GraphLoRA_CF_PROTO_PROBE/logs/CF_PROTOTYPE_PROBE_20261027_103711/prototypes_k16.npz
- Train source default:
  /home/bbx/user_lx/GraphLoRA/dataset/ml-1m/train_ood2.pkl
- **Original E0-prepared training YAML**:
  logs/NO31_CORA_ORIGINAL_READY/train.yaml (under the *running worktree*)
- Existing pretrained MF and Vicuna paths referenced by that config.

NPZ is checked against the actual loaded MF item table (max abs
reconstruction error <= 1e-4), and K16/256/cluster shapes are validated.
No files are overwritten in NIGHT17–NIGHT19.

## Recommended server instructions

Use a fresh worktree; do not switch branches over uncommitted GPU experiments:

    cd /home/bbx/user_lx/GraphLoRA_NO31_CORA
    git fetch https://github.com/NanDingEi/GraphLoRA-Research.git exp/night20-prototype-qformer-20261027
    git worktree add --detach /home/bbx/user_lx/GraphLoRA_NIGHT20 FETCH_HEAD
    cd /home/bbx/user_lx/GraphLoRA_NIGHT20
    conda activate minigpt4

This worktree does not automatically share untracked YAML/log files. If the
prepared E0 config exists in the original worktree, copy just that config:

    mkdir -p logs/NO31_CORA_ORIGINAL_READY
    cp /home/bbx/user_lx/GraphLoRA_NO31_CORA/logs/NO31_CORA_ORIGINAL_READY/train.yaml logs/NO31_CORA_ORIGINAL_READY/train.yaml

CPU synthetic preflight:

    python experiments/no31_cora_original/test_night20_cpu.py

Verify actual input paths and training configuration (no training):

    python experiments/no31_cora_original/night20.py --plan

Run four serial trials on GPU0 in the foreground:

    CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/night20.py --run

Or start with T1 only if you want to inspect metrics before other runs:

    CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/night20.py --only T1_PROTO_COARSE_ONLY

Override the NPZ / TRAIN input paths if they differ:

    python experiments/no31_cora_original/night20.py --plan --prototype /path/to/prototypes_k16.npz --train-file /path/to/train_ood2.pkl

Each trial trains exactly one epoch with original train/Valid split and fresh
initialization. The 4-arm sweep writes into:
logs/NO31_CORA_NIGHT20_<timestamp>/

- per arm: train.yaml / experiment.json / train.log / result.json
- overall: plan.json / results.jsonl / summary.csv
- model checkpoints: output/NO31_CORA_NIGHT20_<timestamp>/<arm>/...

Log checks:
- [CORA-N20-DATA] K16, MF reconstruction, train-only history
- [CORA-N20-PROBE-OK] zero-init heads / layer schedule
- [CORA-ORIGINAL-PROBE-OK] original group8 rank16, 128 target modules
- [NIGHT8-STOP] trained_epochs=1
- [CORA-N20-HEAD-NORMS] coarse/fine trained head norms
- [NIGHT8-RESULT] Valid AUC/UAUC

The historical reference is NIGHT17 E0 **Valid AUC=0.729928474,
UAUC=0.686612232** (not re-evaluated or retrained here).
Do not treat these as Test metrics. Check all four against the **same**
validation split. The CPU prototype probe's 10,401-row full valid results
are not directly comparable to GraphLoRA's 5,200-row valid_small if the run
uses that smaller split.

## Research comparison

T1 vs E0: evidence for group-level collaborative preferences.
T2 vs E0: evidence for personalized within-community signal.
T3 vs T1/T2: complementary information without layer scheduling.
T4 vs T3: does a low-coarse / high-fine depth split outperform uniform mixing?

If T1/T2 fall back to ~0.63 AUC, inspect optimizer/initialization and halt
rather than blindly spending extra GPU time on T3/T4. This advice is not an
automatic sweep stopping rule.

Status: source submitted to GitHub; synthetic preflight and full GPU runs
have **not** been run by the assistant. Do not infer results from code.
