# R4: Frozen-S2 Parameter Isolation (3 Controlled Trials)

## Research question

R3-CFLoRA one-epoch Valid AUC/UAUC = 0.715277595 / 0.657247085; R3-CoRA = 0.720482649 / 0.663502082, each below the matched P2B one-epoch control 0.721492462 / 0.664777242. However the original S2 before continuation is 0.729675260 / 0.684167021, and one-epoch continuation alone already degrades. R4 separates the effect of *new collaborative adapter* from unwanted drift of S2 pretrained parameters.

## Three independently initialized arms

| Arm | Collaborative module | Trainable | Frozen |
| --- | --- | --- | --- |
| R4-01 | ReLLaX-inspired CFLoRA from R3 | r3_adapter.* only | all original S2, including GNN, MF, Projector, all LoRA, L31 biases |
| R4-02 | CoRA-inspired query & weight generator | r3_adapter.* only | all original S2 |
| R4-03 | CoRA-inspired query & weight generator | r3_adapter.* and exactly L31 v_proj.lora_B.default.weight | all other original S2 weights |

Do NOT set model.freeze_lora=true: we still need the original S2 LoRA forward. After the model is constructed and the S2 checkpoint is loaded, we switch requires_grad flags using an exact name whitelist before optimizer initialization. The original Yes/No BCE, semantic teacher settings, GNN-to-Q/V path and data are unchanged.

R4-03 uses a separate 0.1x optimizer group scaling: adapter base LR=2e-5, L31 V-B LR=2e-6. Other R4 runs have no unfrozen original LoRA. The base scheduler must preserve lr_scale multipliers as in the existing runner. Each method starts from the **original S2 checkpoint**, not from an R3/R4-trained checkpoint. The added adapter starts from a zero effect.

## Execute on GPU0, foreground, never parallel

    cd /home/bbx/user_lx/GraphLoRA_collab_v1
    git status --short
    git fetch https://github.com/NanDingEi/GraphLoRA-Research.git fix/collab-gated-optimizer
    git merge --ff-only FETCH_HEAD
    conda activate minigpt4
    export CUDA_VISIBLE_DEVICES=0
    set -o pipefail
    bash experiments/collab_injection_v1/run_r4_isolation.sh 2>&1 | tee logs/collab_v1/R4_ISOLATION_MASTER.log

Run the CPU-only tests alone if you want a quick preflight:

    python -u experiments/collab_injection_v1/test_r4_cpu.py
    python -u experiments/collab_injection_v1/r4_train.py --round 01 --check-source-only

Or run each arm manually (after respective anchor check):

    python -u experiments/collab_injection_v1/r4_probe.py --round 01
    python -u experiments/collab_injection_v1/r4_train.py --round 01
    python -u experiments/collab_injection_v1/r4_probe.py --round 02
    python -u experiments/collab_injection_v1/r4_train.py --round 02
    python -u experiments/collab_injection_v1/r4_probe.py --round 03
    python -u experiments/collab_injection_v1/r4_train.py --round 03

Optional: if the real P2B YAML is not at the known location and autodiscovery finds multiple candidates, add `--p2b-source /absolute/existing/P2B_FULL_E1.yaml` to the shell driver (do not invent a config).

## Stop conditions

1. CPU tests must pass before loading Vicuna.
2. Actual P2B YAML/S2 sha and 32-example zero-init native Yes logits must match S2/R1 within max absolute tolerance 2e-4.
3. R4 freeze policy must list exactly r3_adapter.* parameters (and only L31 V-B for R4-03).
4. 24-step smoke must show actual CFLoRA matrix / CoRA up-projection update, and L31 V-B update in R4-03.
5. SHA256 checks of all frozen LoRA q/v, GNN, MF, Projector and injected biases must match the pre-optimizer snapshot after 24-step smoke **and** full one-epoch training.
6. Full Valid Small 5,200 AUC and UAUC, and unique best checkpoint required.
7. No Test and no background scheduling. Existing partial output is NEVER overwritten. Completed runs are safely skipped on reruns.

Source checkpoint: /home/bbx/user_lx/GraphLoRA/logs/S2_STRICT_20261017_143611/L31_output/20261017143/checkpoint_best.pth

Default P2B config: logs/collab_v1/S2_L30_AUX_P0_20261021_165700/P2B_S2_CACHE_FIXED_FULL.yaml

## Artifacts

32-sample anchors: `logs/collab_v1/R4_01_FROZEN_ANCHOR.json` etc.
Full manifests: `logs/collab_v1/R4_ISOLATION_20261022/R4_01_CFLORA_ONLY/` etc.
Best checkpoints: `output/R4_ISOLATION_20261022/<ARM>/FULL_E1/<date>/checkpoint_best.pth`.

Expected terminal: `[R4-01-RESULT]`, `[R4-02-RESULT]`, `[R4-03-RESULT]`, `[R4-COMPLETE]`.

**Important**: R4 checkpoints contain partial trainable state. Evaluate them by first loading the same original S2 base model and installing R3 collaborative module + R4 freeze policy; then load the R4 checkpoint with `strict=False`, using the method/round recorded in its manifest. Generic GraphLoRA evaluation alone will silently omit new modules.

All source scripts have been committed and inspected, but the R4 code has NOT been executed on the user's RTX 3090 GPU. Real smoke/anchor is mandatory before claiming results.
