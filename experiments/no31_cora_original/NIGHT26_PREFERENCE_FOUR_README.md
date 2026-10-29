# NIGHT26: R1-R4 recommendation preference CoRA residual routing

All code is on the pre-existing NIGHT26 Git branch. Use the existing
/home/bbx/user_lx/GraphLoRA_NIGHT26 worktree; no new worktree or project.

## Motivation

Stop online LightGCN/MF lambda searches after N1=.613915 AUC and
N2=.628094 AUC. Reuse the historical strong G2 Group8 CoRA model,
without building JointLightGCN or computing K16 dynamic graph features.

## Four new experiments (one epoch, seed42, GPU0 only, Valid only)

| ID | Output gates | Route dimension |
| --- | --- | --- |
| R1_GLOBAL_PREF | Bx1 | one global example-specific CoRA residual factor |
| R2_DEPTH8_PREF | Bx8 | Group8 depth-specific factors (four LLM layers per group) |
| R3_QK_VO_PREF | Bx2 | Q/K factor and V/O factor |
| R4_DEPTH8_QKVO_PREF | Bx32 | one factor per Group8 x Q/K/V/O |

At each target attention projection, the original dynamic CoRA residual
is scaled by 1 + 0.1*tanh(router_logits[user,item,history,group,projection]).
Last-layer weight and bias are initialized to zero, yielding exact
unit multipliers. Existing Vicuna, CoRA-P, 8-layer QFormer, group8 projectors,
CTR=.05, KD=.05, semantic and projector losses are UNCHANGED.

Features are 515 dims: normalized frozen MF user vector (256),
normalized frozen MF target vector (256), user-target cosine (1),
mean candidate-to-history cosine (1), valid history fraction (1).
History is taken from InteractedItemIDs_pad, excluding padding, invalid
IDs and the current target even if present. No graph file or extra
pretraining is used. Existing NO31 losses/paths are preserved.

The router has a 515->64 GELU -> 1/8/2/32 head. Initialization
consumes no later global CPU RNG: saved/restored during model init.
Parameters are optimized in existing AdamW 'other' group at the
original learning-rate schedule. No detached gradient in active route.

## Update the SAME NIGHT26 project

    cd /home/bbx/user_lx/GraphLoRA_NIGHT26
    git status --short
    git fetch https://github.com/NanDingEi/GraphLoRA-Research.git exp/night26-mf-graph-lambda-adaptive-20261028
    git switch --detach FETCH_HEAD
    export PYTHONPATH="$PWD"

CAUTION: Preserve any local changes shown by git status before switching.

## Run CPU synthetic preflight

    python -m py_compile experiments/no31_cora_original/preference_router.py experiments/no31_cora_original/adapter.py experiments/no31_cora_original/night26_preference_four.py experiments/no31_cora_original/test_night26_preference_cpu.py
    python experiments/no31_cora_original/test_night26_preference_cpu.py
    python experiments/no31_cora_original/night26_preference_four.py --plan

Expected: [N29-CPU-ROUTE-PASS] x4 and [N29-CPU-PASS].
The CPU test verifies no target leakage in history; exact CoRA forward and
base CoRA parameter gradient parity for unit gain; 4 dimensions and trained
gate gradients, with bounded outputs.

## GPU0 foreground serial one-epoch Valid training

    CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/night26_preference_four.py --run

Optional single arm:

    CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/night26_preference_four.py --run --only R2_DEPTH8_PREF

No background, no parallel training, no legacy baseline rerun.
Logs: logs/NO31_CORA_NIGHT26_PREF4_<timestamp>/summary.csv,
results.jsonl, R1.../train.log, etc. Checkpoints remain under
output/NO31_CORA_NIGHT26_PREF4_<timestamp>/R*/...
End-of-epoch checks: [N29-ROUTER-FIRST], [N29-ROUTER-GRAD],
[N29-ROUTER-END] plus standard original CoRA / CTR/KD checks.

CUDA cache clearing is disabled (empty_cache_interval=0) in these four
new experiments to reduce synchronization/allocator overhead. If
memory behavior is bad, use --legacy-cache to revert to per-batch
torch.cuda.empty_cache() without changing router equation.
No performance or model metric improvement is claimed before GPU runs.
