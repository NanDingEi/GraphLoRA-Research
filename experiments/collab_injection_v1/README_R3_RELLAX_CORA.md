# R3 — ReLLaX CFLoRA and CoRA-inspired parameter-injection trials

These are **mechanism-level transfers onto GraphLoRA S2**, not full reproductions of either paper. No direct modification of Yes logits or final LM head. No Test-set selection.

## Methods

**R3-CFLORA (ReLLaX inspiration).** ReLLaX CFLoRA generalizes vanilla LoRA BA to BWA with a per-sample rank-by-rank component interaction W. We preserve the original S2 ID-token GNN injection and modify L31 Q and V with W_ui = I + 0.1*tanh(MLP([g_u,g_i,g_u*g_i,abs(g_u-g_i)])). Graph penultimate embeddings g are 256-dimensional and rank r=8. MLP final layer is zero-initialized, so before training W_ui=I. SUBR and SPA are deliberately not added.

**R3-CORA (CoRA inspiration).** Original CoRA uses learned collaborative queries, self-/cross-attention with user/item collaborative features, and low-rank incremental weights in the LLM parameter space without new prompt tokens. Here 4 learned queries (width 64) attend to user+candidate GNN penultimate features; pooled queries generate 2 rank-8 condition vectors for L31 Q/V. An additional B_c diag(s_ui) A_S2 weight update is used, with B_c initialized to zero. Unlike original CoRA, the existing LoRA and GNN learning policy follows matched P2B.

Both preserve the original GraphLoRA Yes/No task, pretrained Vicuna, existing L31 Q/V GNN injection and S2's semantic alignment. No candidate-cosine scalar gate from R2 is used.

Paper references: https://doi.org/10.1145/3766553 (ReLLaX), https://ojs.aaai.org/index.php/AAAI/article/view/33334 (CoRA).

## Files

- r3_paper_adapters.py: runtime injection modules installed before checkpoint loading
- r3_entry.py: run original model and optimizer under the proper R3 runtime
- r3_probe.py: 32-example exact initial S2 Yes score parity anchor
- r3_train.py: independent 24-step smoke + full 1 epoch Valid Small per method
- test_r3_cpu.py: small-module zero-init and autograd sanity checks
- run_r3_paper_trials.sh: GPU0 foreground-only serial runner

## Commands on the server

    cd /home/bbx/user_lx/GraphLoRA_collab_v1
    git status --short
    git fetch https://github.com/NanDingEi/GraphLoRA-Research.git fix/collab-gated-optimizer
    git merge --ff-only FETCH_HEAD
    conda activate minigpt4
    export CUDA_VISIBLE_DEVICES=0
    set -o pipefail
    bash experiments/collab_injection_v1/run_r3_paper_trials.sh 2>&1 | tee logs/collab_v1/R3_COMPARE_MASTER.log

Don't force reset when local changes conflict. The runner executes CPU tests, matched P2B control-source check, 32-example zero-start S2 parity checks, then two independent 24-step smokes and one-epoch full 5200-Valid training runs. All serial and foreground. Neither method inherits weights from the other.

**Original S2 checkpoint**: /home/bbx/user_lx/GraphLoRA/logs/S2_STRICT_20261017_143611/L31_output/20261017143/checkpoint_best.pth

**P2B full-run source YAML**: logs/collab_v1/S2_L30_AUX_P0_20261021_165700/P2B_S2_CACHE_FIXED_FULL.yaml

Original S2 Valid Small: AUC 0.729675260; UAUC 0.684167021. Matched reference P2B after one epoch: AUC 0.721492462; UAUC 0.664777242. Compare R3 one-epoch results primarily against P2B, not original S2 directly. Same original S2 init, batch=3, lr=2e-5, seed=42, scheduler horizon and fix_autocast_cache=True.

To run one method at a time:

    python -u experiments/collab_injection_v1/test_r3_cpu.py
    python -u experiments/collab_injection_v1/r3_train.py --method cflora --check-source-only
    python -u experiments/collab_injection_v1/r3_probe.py --method cflora
    python -u experiments/collab_injection_v1/r3_train.py --method cflora
    python -u experiments/collab_injection_v1/r3_probe.py --method cora
    python -u experiments/collab_injection_v1/r3_train.py --method cora

Results: logs/collab_v1/R3_CFLORA_ANCHOR.json and R3_CORA_ANCHOR.json; training output/manifests under logs/collab_v1/R3_COMPARE_20261022/R3_CFLORA and R3_CORA; checkpoint_best.pth under output/R3_COMPARE_20261022/R3_{CFLORA,CORA}/FULL_E1/.

Important: evaluating R3-trained checkpoints with unmodified train_collm_mf_din.py is INVALID: it will not construct or load the R3 adapters. Always use r3_entry.py and the method-specific YAML. No Test until method selection.

R3 module changes are anchored at step zero, and smoke must prove shallow S2 LoRA tensor updates and new R3 parameter updates. New module sizes are not parameter-matched between the two papers; results are controlled transfer feasibility tests rather than a parameter-budget superiority claim.

These scripts have NOT been executed on the user's 3090 server; any real API/AMP/shapes error must stop before an experimental result is claimed.