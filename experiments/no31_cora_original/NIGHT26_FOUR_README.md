# NIGHT26: four new experiments + opt-in training-speed patch

Continue in the **existing** /home/bbx/user_lx/GraphLoRA_NIGHT26 worktree.
No new project, Git branch or worktree. Log and checkpoint directories
still have per-run timestamps so previous experiments are never overwritten.

## Four new one-epoch experiments; no historical baseline rerun

| ID | graph fusion | 3-hop aggregation | Purpose |
| --- | --- | --- | --- |
| N1_L3_L015 | user=item=.15 | uniform | interpolate E2 and NIGHT25 M2 |
| N2_L3_L025 | user=item=.25 | uniform | test above E2 lambda=.2 |
| N3_USER01_ITEM03 | user=.10, item=.30 | uniform | stronger collaborative item information |
| N4_USER03_ITEM01 | user=.30, item=.10 | uniform | stronger collaborative user information |

All four use identical MF-initialized raw LightGCN embeddings, 3 layers,
seed42, graph LR scale=.1, zero graph BCE, AMP initial loss scale256,
CoRA group8, rank16 alpha48, Q/K/V/O all 32 Transformer layers,
original MF mixed prompt, CTR=.05, KD=.05, semantic alignment enabled.
One epoch each and VALID AUC/UAUC (not Test).
Historical E2 Valid AUC=.7297545888776531 UAUC=.6885904780046014
is a read-only reference, not rerun.

## Speed optimizations enabled only for these four runs

1. graph_base forward returns only batch-selected graph user/item embeddings.
   It skips the unused K16 community prototype mean, item residuals,
   user/community history sparse aggregation and unused graph BCE.
   Full 3-hop differentiable graph propagation is NOT removed or detached.
2. Gradient finite/norm diagnostic probes inspect the first eight graph
   backwards and every 256 thereafter instead of syncing on every batch.
   AMP GradScaler still checks nonfinite gradients on every optimizer step.
3. Per-batch dynamic-CoRA matrix finite checks run on the first 16 batches
   and every 256 thereafter, avoiding repeated GPU synchronization.
4. PyTorch empty_cache is not called every microbatch; the historical
   behavior is retained as the default and can be restored with --legacy-perf.
5. The graph fast path trusts IDs already validated by the CoRA adapter,
   avoiding four duplicate GPU->CPU range-check synchronizations per step.

These changes should improve runtime but no speedup percentage is claimed
without actual GPU timing. The key O(N+E) full-graph propagation and
FP32 CoRA remain. Peak reserved CUDA memory may rise when caching is enabled.
Use --legacy-perf if memory behavior or stability regresses.

## Pull into existing NIGHT26 worktree

    cd /home/bbx/user_lx/GraphLoRA_NIGHT26
    git status --short
    git fetch https://github.com/NanDingEi/GraphLoRA-Research.git exp/night26-mf-graph-lambda-adaptive-20261028
    git switch --detach FETCH_HEAD
    export PYTHONPATH="$PWD"

Do not overwrite any uncommitted local changes; commit or preserve first.

## Preflight

    python -m py_compile experiments/no31_cora_original/night26_four.py experiments/no31_cora_original/night24_joint_graph.py experiments/no31_cora_original/adapter.py experiments/no31_cora_original/test_night26_four_cpu.py
    python experiments/no31_cora_original/test_night26_four_cpu.py
    python experiments/no31_cora_original/night26_four.py --plan

Expect [N28-CPU-FAST-PARITY-PASS], [N28-CPU-ASYMMETRIC-PASS]
and [N28-CPU-PASS]. CPU tests compare fast/legacy selected user/item
graph embeddings and graph-ID parameter gradients, plus fixed asymmetry.

## Foreground GPU0 serial command

    CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/night26_four.py --run

Optionally provide --graph-artifact with an existing verified N26 MF-L3
NPZ file. Otherwise it autodiscovers a verified graph artifact or creates
one MF-init artifact once, with zero BPR pretraining.

    CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/night26_four.py --run --only N3_USER01_ITEM03

On GPU OOM or an unexpected numerical issue, rerun selectively with
--legacy-perf to disable all four optimization controls while keeping
the same model mathematics.

Logs: logs/NO31_CORA_NIGHT26_FOUR_<timestamp>/summary.csv
and results.jsonl, plus each arm's train.log, experiment.json and YAML.
Checkpoints: output/NO31_CORA_NIGHT26_FOUR_<timestamp>/N*/...
summary.csv includes wall_seconds (model load, training and Valid eval),
Valid AUC/UAUC, assigned U/I coefficients and optimizer AMP statistics.

Note: changing λ or separate U/I fixed coefficients can change performance;
this runner does not claim these methods beat E2 until Valid results exist.
