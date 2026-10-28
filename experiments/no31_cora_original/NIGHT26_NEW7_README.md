# NIGHT26 NEW7 — seven new LightGCN + CoRA experiments

All changes are in the existing GraphLoRA_NIGHT26 worktree and GitHub NIGHT26 branch.
No baseline reruns, no new project or git worktree. Output folders only isolate logs and checkpoints.

| Experiment | Fusion | 3-hop layer aggregation | Added feature |
| --- | --- | --- | --- |
| E1_L3_LAMBDA0025 | fixed lambda=.025 | uniform | weaker graph |
| E2_L3_LAMBDA0200 | fixed lambda=.2 | uniform | stronger graph |
| E3_DUAL_UI_GATE | learnable user/item lambda; init .1 | uniform | independent user/item gating |
| E4_DEGREE_UI_GATE | degree-aware user/item lambda; init .1 | uniform | train positive-degree conditioning |
| E5_SHALLOW_FIXED | fixed lambda=.1 | fixed [.4,.3,.2,.1] | shallow-first propagation |
| E6_SHALLOW_LEARNED | fixed lambda=.1 | learned from [.4,.3,.2,.1] | adaptive depth |
| E7_GRAPH_C2F_GROUP8 | fixed lambda=.1 | uniform | K16 coarse/fine Graph Group8 routing |

All arms use exactly the same 3-layer MF-initialized LightGCN ID table and train-positive adjacency.
No graph BPR pretraining is repeated. Lambda gates are sigmoid-limited to (0,.5),
trained by the downstream LLM recommendation objective. E4 uses normalized log(1+degree)
from unique positive TRAIN edges only; both user/item gates start at exactly .1.
E5 and E6 start with identical shallow-prior graph propagation; E6 trains 4 softmax logits.
E7 uses existing Graph8 coarse/fine head schedule and audits both zero-output
initialized heads after training. The direct L31 GNN injection remains OFF.

Unchanged: seed42, GPU0, ML-1M OOD2, Vicuna-7B, mixed MF prompt, rank16 alpha48,
Q/K/V/O all 32, original semantic, projector, CTR=.05, KD=.05 losses, 1 train epoch,
3-epoch LR schedule horizon, Valid AUC/UAUC only, no graph BCE auxiliary loss.
FP32 graph/CoRA precision island and AMP init scale256 remain as in NIGHT25/26.

## Update code in the existing NIGHT26 project

    cd /home/bbx/user_lx/GraphLoRA_NIGHT26
    git status --short
    git fetch https://github.com/NanDingEi/GraphLoRA-Research.git exp/night26-mf-graph-lambda-adaptive-20261028
    git switch --detach FETCH_HEAD
    export PYTHONPATH="$PWD"

Do not use git switch if you have uncommitted changes you need to preserve.

## CPU check and plan

    python experiments/no31_cora_original/test_night26_new7_cpu.py
    python experiments/no31_cora_original/night26_seven.py --plan

## One GPU0 foreground serial run of all seven

    CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/night26_seven.py --run

Optional: pass --graph-artifact PATH to reuse a specific existing NIGHT26 MF L3 NPZ.
The runner otherwise reuses the latest verified NIGHT26 graph artifact (or creates it once).
Run --only E4_DEGREE_UI_GATE or --start E5_SHALLOW_FIXED to selectively rerun/restart.
GPU0 must be idle; the runner refuses concurrent training and stops upon any failure.

Results: logs/NO31_CORA_NIGHT26_NEW7_<timestamp>/summary.csv and results.jsonl,
per-arm train.log/result.json, checkpoint paths under output/NO31_CORA_NIGHT26_NEW7_*/.
Checks: N27-ARTIFACT-PARITY-PASS, N27-CPU-PASS, N27-GATE-GRAD, N27-GATE-END,
N26-LAYER-LOGIT-GRAD, N26-LAYER-WEIGHTS-END, N27-C2F-HEADS-END, N24-AMP-SUMMARY.

Note the known limitation: positive target edges from TRAIN still appear in graph
propagation during train, while validation target edges are not inserted into the graph.
Historical G2 and NIGHT25 M2 metrics are references only, not rerun.
