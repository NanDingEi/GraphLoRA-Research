# NIGHT26 — matched FP32 MF-only control, 3-layer lambda 0.05, adaptive depth

Three **foreground-serial** ml-1m OOD2 Valid-only experiments.  Same
pretrained MF source, same K16 graph artifact, same 3 propagation steps,
same group8 CoRA rank16/alpha48 full Q/K/V/O all 32, original Vicuna-7B,
same seed42, optimizer schedule, GPU0, accumulation=4, one trained LLM
epoch.  No historical checkpoint overwritten; NIGHT26 uses a separate worktree.

| Order | ID | MF graph node init | L0..3 weights | CoRA lambda | graph BCE |
| --- | --- | --- | --- | --- | --- |
| C0 | C0_MF_ONLY_FP32 | Same pretrained MF file (not used in forward) | fixed uniform (graph not called) | 0 | 0 |
| C1 | C1_L3_LAMBDA005 | Exact copied MF | fixed uniform 1/4 each | 0.05 | 0 |
| C2 | C2_L3_ADAPTIVE_LAMBDA005 | Same exact MF | learned softmax of 4 zero-initialized logits | 0.05 | 0 |

CoRA condition is **(1-lambda) * frozen MF_pair + lambda * LightGCN_pair**.
For C0, lambda=0 and graph auxiliary=0, so no graph forward or graph
gradient; this isolates the trained MF-only output under the SAME FP32
CoRA numerical policy as C1/C2. C0 is not being claimed as an exact
historical G2 replay. The graph module remains installed and unused for
setup/optimizer parity and existing code paths, including original hybrid
MF prompt, CTR=0.05, KD=0.05, semantic loss and no direct L31 injection.

C1 and C2 differ **only in propagation aggregation**. For C2:

  w = softmax(graph_joint.layer_logits), layer_logits = [0,0,0,0]
  E_graph = w0 * E0 + w1 * A E0 + w2 * A^2 E0 + w3 * A^3 E0

No gradient detach. The extra four learnable scalar logits enter the
existing graph optimizer parameter group (same LR multiplier=0.1).
The 3-layer graph is initially identical to C1, up to FP32 reduction
round-off. At the end of training C2 must report four positive weights
summing to 1 and observed nonzero layer-logit gradients (up to the first three are logged).
If weights do not move from uniform, the run fails rather than presenting
an illusory adaptive result.

## Set up a separate worktree

    cd /home/bbx/user_lx/GraphLoRA_NIGHT25
    git status --short
    git fetch https://github.com/NanDingEi/GraphLoRA-Research.git exp/night26-mf-graph-lambda-adaptive-20261028
    git worktree add --detach /home/bbx/user_lx/GraphLoRA_NIGHT26 FETCH_HEAD
    cd /home/bbx/user_lx/GraphLoRA_NIGHT26
    conda activate minigpt4
    mkdir -p logs/NO31_CORA_ORIGINAL_READY
    cp /home/bbx/user_lx/GraphLoRA_NIGHT25/logs/NO31_CORA_ORIGINAL_READY/train.yaml logs/NO31_CORA_ORIGINAL_READY/train.yaml
    export PYTHONPATH="$PWD"

The runner forces the current worktree to the start of sys.path and
PYTHONPATH, so stale NIGHT24/NIGHT25 imports must not contaminate results.

## Plan and CPU preflight

    python experiments/no31_cora_original/test_night26_cpu.py
    python experiments/no31_cora_original/night26.py --plan

## Run ALL THREE on GPU0, one foreground serial invocation

    CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/night26.py --run

For selective reruns (a *new* log root is created; already finished
results are not overwritten):

    CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/night26.py --run --only C2_L3_ADAPTIVE_LAMBDA005

Or restart from the failed arm:

    CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/night26.py --run --start C1_L3_LAMBDA005

**Do not run if GPU0 is occupied**. A preflight checks this and stops
before all GPU compute. One step failing stops the later steps. Training
is not detached or placed in a background process.

## Output

A new `logs/NO31_CORA_NIGHT26_<timestamp>/` directory contains the
single common MF-initialized graph artifact, its manifest and pretrain
log (0 epochs), a separate experiment.json, train.yaml, train.log and
result.json for each arm, results.jsonl, summary.csv, and a failed.json
on errors. Actual checkpoints are under `output/NO31_CORA_NIGHT26_.../`.

Main audit markers:

- [N25-MF-WEIGHT-PARITY-PASS] exact pretrained MF node weights
- [N26-MF-ONLY] graph bypass for C0
- [N26-MF-ONLY-AUDIT-PASS] no graph gradients expected in C0
- [N24-GRAPH-GRAD] nonzero in C1 and C2 after CoRA-P updates
- [N26-LAYER-LOGIT-GRAD] nonzero learned logits gradients in C2
- [N26-LAYER-WEIGHTS-END] learned weights and deviation from uniform
- [N24-AMP-SUMMARY] successful/skipped updates
- [N26-ARM-DONE] and [N26-COMPLETE] for serial status

**Interpretation:** C0 vs C1 quantifies 3-layer MF-initialized graph
injection at lambda=0.05, without comparing across different precision
policies. C1 vs C2 tests whether learned aggregation improves on equal
aggregation with the same residual injection. Historical G2 and N25
M2 (lambda=0.1) remain contextual comparisons, not matched controls.

Known caveat: original TRAIN positive target edges are present in the
LightGCN adjacency during training; this iteration does not yet run a
leave-target-out adjacency ablation. AUCs are VALID and should not be
represented as TEST.
