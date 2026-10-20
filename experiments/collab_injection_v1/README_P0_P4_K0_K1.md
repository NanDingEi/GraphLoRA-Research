# Two-round candidate-conditioned collaboration experiments (2026-10-20)

Branch: `fix/collab-gated-optimizer`. **GPU0 foreground**, no Test evaluations.

## Round 1: P0-P4 valid-only (no training)

Run CPU test first:
```bash
python experiments/collab_injection_v1/test_target_k_cpu.py
python -m py_compile minigpt4/models/minigpt4rec_v2.py \
  experiments/collab_injection_v1/eval_vg_token_roles.py \
  experiments/collab_injection_v1/run_k0_k1_2ep.py
```

Run:
```bash
export CUDA_VISIBLE_DEVICES=0
set -o pipefail
python experiments/collab_injection_v1/eval_vg_token_roles.py \
  --config logs/collab_v1/S2_VG_2EP_20261020_174659/VG1_CANDIDATE_FULL.yaml \
  --checkpoint output/S2_VG_2EP_20261020_174659/VG1_CANDIDATE/checkpoint_merged_best.pth \
  --factor 50 \
  --result logs/collab_v1/P0_P4_token_roles.json \
  2>&1 | tee logs/collab_v1/P0_P4_token_roles.log \
  | grep --line-buffered -E 'TOKEN-P|Traceback|RuntimeError'
```

Conditions P0 (VG off), P1 (UserID only), P2 (history ID placeholders only), P3 (TargetItemID placeholder only), P4 (all ID placeholders). **P0_REPEAT** reruns P0 with identical seeds and initial prompt flags; script aborts unless prediction logits, FP32 reconstructed Yes logit, and attention activations all match exactly. Roles come from actual `get_ids_order` segment layout, not item ID equality. This distinguishes a historical item from the candidate even if IDs match. Each condition evaluates 5,200 Valid Small examples, reports AUC/UAUC plus logit/FP32/attention shifts.

Read `P0_P4_token_roles.json` **before** selecting the next round's `--scope`. These are diagnostic counterfactual interventions, not trained methods. Do not select hyperparameters by Test metrics.

## Round 2: K0/K1 Key-only 1-epoch training

Same S2-L31 warm start, same Q/V GNN injection, frozen original LoRA/GNN/projector/recommender. A zero-initialized trainable K projection residual at layer 31 bypasses LoRA-B:
`K_new[b,p] = K_S2[b,p] + Adapter([z_user,z_target,z_node,z_target*z_node])` at selected ID token positions.

K0 masks the candidate blocks to 0. K1 uses them. Both have four identically shaped new trainable tensors. Each independently starts from the same S2 champion.

After reviewing round 1, choose scope (default `history`):
```bash
python experiments/collab_injection_v1/run_k0_k1_2ep.py --scope history
```

Other supported scopes: `user`, `target`, `all`. Use the **same** choice for both arms. The orchestrator serially performs a CPU syntax/unit check, creates isolated configs from the S2 champion, verifies zero-init Valid AUC against `0.7296752596287361`, runs 24 training batches and verifies that ONLY Key adapter weights changed, then trains each arm for exactly one epoch and evaluates Valid Small. Checkpoint drift checker fails if any original model weight moved.

Initial run configuration: seed=42, lr=5e-5, 1 epoch, batch=existing S2 value, gradient accumulation=3, GPU0 only. The 24-batch smoke uses a separate output directory and never contributes to the full epoch. Full training runs independently for K0 and K1; NO optimizer continuation from the smoke or other arm.

Results in `logs/collab_v1/S2_K_2EP_SCOPE_TIMESTAMP/summary.tsv`; new adapter-only and merged best checkpoint in `output/S2_K_2EP_SCOPE_TIMESTAMP/`. The `[K-2EP-COMPLETE]` summary is printed only after both drift checks pass.

## Decision criteria

1. Stage 1: compare P1/P2/P3 effects versus P0 with matched sample order, then justify the chosen Key scope.
2. Stage 2: seek `AUC(K1) > AUC(K0)` AND `AUC(K1) > 0.7296752596`. The suggested exploratory margin is at least `0.001` AUC above S2 before spending compute on multi-seed verification, not an absolute significance rule.
3. If K1 does not exceed K0 or both barely change predictions, stop the adapter line; inspect K perturbation and attention changes rather than blindly increasing epochs or strength.

No training has been executed by the assistant. The GPU scripts and CPU tests must be executed on the user's server.
