# Collaborative Prototype–Residual MF Probe

This CPU-only experiment replaces the unsuccessful handmade MF Hadamard/history-interaction signals. It does NOT modify CoRA, LoRA, or Vicuna, and does NOT evaluate the Test split.

## Method

The pretrained MF score is s(u,i) = e_u^T e_i. We construct a positive-interaction item co-occurrence graph exclusively from TRAIN, normalize users' contributions by 1/positive_degree(user), and apply spectral embedding plus KMeans (K=16/32 by default).

Each item belongs to a community g(i) with prototype c_k = mean item embedding. The specific-item residual is d_i = e_i - c_{g(i)}. Define s_coarse(u,i) = e_u^T c_{g(i)} and s_fine(u,i) = e_u^T d_i. Their sum equals original MF exactly.

**Important:** Inside a fixed (user, community), the coarse term is constant, so the within-group AUC of the full MF score and of the fine residual MUST be identical. This is an identity, NOT a performance improvement. Evaluate coarse signal versus random size-matched communities, and evaluate coverage and item-popularity baseline for within-community rankings.

Unconnected items receive the nearest community under frozen MF cosine similarity (without using Valid labels). Padding item ID 0 remains unassigned. Only valid labels are used to report diagnostics.

## Run on the server

Use an isolated worktree so existing GPU training directories remain unchanged.

    cd /home/bbx/user_lx/GraphLoRA_NO31_CORA
    git fetch https://github.com/NanDingEi/GraphLoRA-Research.git exp/cf-prototype-residual-probe-20261027
    git worktree add --detach /home/bbx/user_lx/GraphLoRA_CF_PROTO_PROBE FETCH_HEAD
    cd /home/bbx/user_lx/GraphLoRA_CF_PROTO_PROBE
    conda activate minigpt4
    python experiments/cf_prototype_probe/test_probe_cpu.py
    python -u experiments/cf_prototype_probe/probe.py --k-values 16 32 --seed 42

Default data directory: /home/bbx/user_lx/GraphLoRA/dataset/ml-1m
Default MF checkpoint: /home/bbx/user_lx/GraphLoRA/checkpoints/0912_ml1m_oodv2_best_model_d256lr-0.001wd0.0001.pth
Override the defaults using --data-dir and --mf-checkpoint.

## Outputs

Under logs/CF_PROTOTYPE_PROBE_<timestamp>:

- manifest.json: paths, settings, sample counts.
- summary.csv: MF and coarse/fine AUC/UAUC, random community comparison, within-/between-community pairwise results, residual energy ratio.
- metrics_k16.json / metrics_k32.json: P1/P2/P3 statistics.
- prototypes_k16.npz / prototypes_k32.npz: item_to_cluster (padding -1), prototypes, mf_residual, item_connected_in_graph.

P1 tests if community-level collaboration meaningfully outperforms a random community assignment with exactly the same sizes. Cross-community positive-negative pairwise accuracy is separately reported.

P2 evaluates ranking within the same user/community against training-set item popularity. The fine residual and original MF rankings must match by construction. Report eligible users, rows and pairs to rule out tiny subgroup coverage.

P3 checks maximum score reconstruction error below 1e-8, embedding-residual energy, score variance, graph connectivity and cluster imbalance.

No fresh LLM validation or Test metrics are produced. Continue to CoRA mapping ONLY if the prototype-level collaborative signal is useful and group coverage is sufficient.
