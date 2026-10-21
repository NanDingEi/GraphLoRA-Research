# P0 — L30 auxiliary recommendation supervision (S2 warm-start)

This is the next test after D1–D3 showed that L30 has near-final recommendation ability but fixed-layer voting does not improve Valid AUC.

## Hypothesis

Preserve **native S2-L31 Yes Logit** as the ONLY inference prediction and final primary recommendation loss. Add optional small training-time auxiliary BCE at L30, with the same original Yes token readout (pretrained frozen final RMSNorm + Yes LM-head row in FP32). Gradients from this auxiliary BCE flow to LoRA and other already-trainable upstream S2 components; no new trainable head, no GNN-only objective, no new inference parameters. The existing S2 L31 GNN injection is unchanged.

`L_total = L_rec(L31) + lambda_30 * BCE(YesLens(L30), label) + original_S2_semantic_terms`

`lambda_30=0` is the isolated control. `lambda_30=0.02` is the treatment. Both are initialized independently from the SAME S2 champion and trained with the SAME seed and fresh optimizer schedule; 1 complete epoch each, with 3-epoch cosine horizon. The original S2 checkpoint and code path are untouched.

## Execute on server GPU0 foreground only

First save uncommitted work before syncing the branch:

```bash
cd /home/bbx/user_lx/GraphLoRA_collab_v1
git status --short
git fetch https://github.com/NanDingEi/GraphLoRA-Research.git fix/collab-gated-optimizer
git merge --ff-only FETCH_HEAD
conda activate minigpt4
export CUDA_VISIBLE_DEVICES=0
set -o pipefail
python -u experiments/collab_injection_v1/run_l30_aux_p0.py \
  --seed 42 --lr 2e-5 --accum 3 --smoke-steps 24 \
  2>&1 | tee logs/collab_v1/L30_P0_console.log
```

The runner runs syntax checks, a pure CPU gradient + semantic-loss assembly regression test, and configuration generation; it launches two training arms **serially**. CONTROL short smoke → CONTROL full ep0 Valid → auxiliary configuration's S2 Valid anchor (must exactly match AUC 0.729675260 and UAUC 0.684167021) → AUX smoke → AUX full ep0 Valid. Evaluation never applies L30 aux; final score is always native S2 L31 Yes logit.

Required logs: `[L30-CPU] ALL PASS`, `[P0-RESULT] P0_CONTROL`, `[L30-ANCHOR] PASS`, `[L30-AUX] step=`, `[P0-RESULT] P0_L30_AUX002`, `[P0-COMPLETE]`. On failure the runner stops without launching the next arm.

Logs and checkpoints under `logs/collab_v1/S2_L30_AUX_P0_<timestamp>/` and `output/S2_L30_AUX_P0_<timestamp>/`.

`comparison.json` contains AUC/UAUC and **treatment minus retrained control**, plus both differences vs the untouched S2 initial checkpoint. Do not misinterpret the treatment-vs-original difference as effect of supervision; primary controlled comparison is treatment minus retrained control.

To resume after an arm failure (same git commit and hyperparams), use the `[P0-RESUME]` command printed by the runner. Completed arm is only skipped if its checkpoint and full config hash match; partial outputs from a failed arm are discarded.

## Cautions

- This is a **single-seed exploratory experiment**, not a claim of statistically reliable improvement.
- The new L30 auxiliary loss uses the true recommendation labels during training, and the final L31 loss remains primary. No teacher/student distillation assumption.
- Because the S2 checkpoint already has strong Valid performance, one extra epoch can degrade it; CONTROL is indispensable.
- No Test metrics until a treatment beats the matched control on Valid and is independently replicated.
- This is intentionally *not* multi-layer injection and not prediction voting. Test the auxiliary-gradient hypothesis before expanding to L28/L30 or GNN-conditioned cross-layer corrections.

## 2026-10-21: fixed guard and resume of the first actual run

The original P0 run completed CONTROL (`AUC=0.7219307045640613`, `UAUC=0.6659307008748429`) but refused the AUX anchor before training because the model guard compared `int(gnn_scale) != 0.1`. Python truncates `int(0.1)` to zero. We corrected the guard to `abs(float(gnn_scale)-0.1) > 1e-10`.

The P0 runner now performs a **fail-closed migration** only when `state.json` refers to commit `1a41ce8a2fa8fd18bf7a01e410fd62bec7858fa0`, the model source differs from that commit in exactly the guarded expression, the S2 checkpoint hash is unchanged, the completed CONTROL checkpoint/config still match, and the failed AUX log contains the precise former guard error. CONTROL is skipped; AUX anchor/smoke/full start anew.

Resume the original run from the same GraphLoRA working directory after git fast-forward:

```bash
cd /home/bbx/user_lx/GraphLoRA_collab_v1
git status --short
git fetch https://github.com/NanDingEi/GraphLoRA-Research.git fix/collab-gated-optimizer
git merge --ff-only FETCH_HEAD
conda activate minigpt4
export CUDA_VISIBLE_DEVICES=0
set -o pipefail
python -u experiments/collab_injection_v1/run_l30_aux_p0.py \
  --resume-dir /home/bbx/user_lx/GraphLoRA_collab_v1/logs/collab_v1/S2_L30_AUX_P0_20261021_165700 \
  --seed 42 --lr 2e-5 --accum 3 --smoke-steps 24 \
  2>&1 | tee -a logs/collab_v1/L30_P0_console.log
```

Expect `[P0-GUARD-FIX]`, `[P0-SKIP] P0_CONTROL`, `[L30-ANCHOR] PASS`, then the AUX smoke/full evaluation. If any invariant fails, stop and inspect; do **not** bypass the guard or rerun CONTROL unless necessary.
