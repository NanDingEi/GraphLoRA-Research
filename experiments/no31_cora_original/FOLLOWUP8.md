# CoRA second-stage: F1–F8 (R2/R7 follow-up)

**Existing checkout only**: /home/bbx/user_lx/GraphLoRA_NO31_CORA
**Existing GitHub branch**: exp/no31-ctrkd-cora-20261023
No new git worktree, no background processes, GPU0 only.
All eight are fresh *one-epoch* experiments; no checkpoint is resumed.

## Motivation: first eight completed (ML1M OOD2, Valid)

| Reference | Method | AUC | UAUC |
|---|---|---:|---:|
| NO31 | Baseline CTR/KD + standard LoRA | 0.719760366 | 0.685106415 |
| R0 | Original CoRA, fixed autocast, alpha32 QKVO | 0.633755947 | 0.593198036 |
| R2 | Original CoRA, alpha32 V/O | 0.725721127 | 0.682916847 |
| R7 | Original CoRA, alpha64 QKVO | 0.728864215 | 0.685534572 |
| R8 | Original CoRA, alpha32 QKVO, P-LR x2 | 0.616277961 | 0.560654411 |

R7's valid AUC beats NO31 by +0.009104, but valid UAUC improves
only +0.000428 (single seed). R2's UAUC is -0.002190 vs NO31.
R7 vs R2 changes two axes: injection set and alpha, so there is
**no evidence which factor explains their small relative difference**.

## Eight controlled follow-ups

| ID | CoRA attention targets | alpha | seed | Question |
|---|---|---:|---:|---|
| F1 | 32 layers V/O | 64 | 42 | Combine R2 projection choice with R7 alpha |
| F2 | 32 layers V/O | 96 | 42 | Is larger alpha better on V/O? |
| F3 | 32 layers Q/K/V/O | 96 | 42 | Is alpha64 near the best for full QKVO? |
| F4 | 32 layers Q/K/V/O | 48 | 42 | Is the threshold between alpha32 and alpha64? |
| F5 | 32 layers V only | 32 | 42 | Marginal contribution of O: compare with R2 |
| F6 | 32 layers O only | 32 | 42 | Marginal contribution of V: compare with R2 |
| F7 | 32 layers V/O | 32 | 43 | Fresh-seed R2 repeat |
| F8 | 32 layers Q/K/V/O | 64 | 43 | Fresh-seed R7 repeat |

Compare F7 and F8 **within seed43**, and R2 and R7 within seed42.
Do not interpret seed43 runs as directly comparable to a seed42
historical NO31 baseline for variance-controlled statistical claims.

All unchanged: rank16, frozen MF user/item CoRA conditioning, 8-layer
QFormer, 4 queries, generator architecture, 0.05 LoRA dropout,
train batch2, eval batch3, accum4, initial LR 1e-4, P LR 1x,
semantic correction 1e-4 (teacher layer8), autocast cache fix ON,
NO31 gnn_scale=0 and no input injection, CTR/KD=0.05,
3-epoch scheduler horizon but **one epoch training only**, Valid only.

## Update existing checkout

    cd /home/bbx/user_lx/GraphLoRA_NO31_CORA
    conda activate minigpt4
    git status --short
    git fetch https://github.com/NanDingEi/GraphLoRA-Research.git \
      exp/no31-ctrkd-cora-20261023
    git merge --ff-only FETCH_HEAD

If merge refuses because of local modifications, preserve them;
do not run git reset --hard or discard the historical checkpoints.

## Preflight (no GPU)

    python experiments/no31_cora_original/test_followup8_cpu.py

    python experiments/no31_cora_original/followup8.py --check-fixed-r0

    python experiments/no31_cora_original/followup8.py --plan

The checkpoint health check uses:
output/NO31_CORA_ORIGINAL_20261023_234005_110337/20261023234/checkpoint_0.pth.
It must find nonzero P weights in **all of L0–L7**. This is the
fixed-autocast R0 checkpoint, not the failed earlier R0.

## Run only the first four F1–F4 (recommended initial batch)

    python experiments/no31_cora_original/followup8.py --plan --first 4

    CUDA_VISIBLE_DEVICES=0 \
    python -u experiments/no31_cora_original/followup8.py --run --first 4

This runs F1 then F2 then F3 then F4, and stops. F5–F8 are *not*
started. All four share a single timestamped FOLLOWUP8 log directory and
summary.csv. To run F5–F8 separately later use:

    CUDA_VISIBLE_DEVICES=0 \
    python -u experiments/no31_cora_original/followup8.py --run --start F5

The second batch gets its own timestamp; it does not overwrite F1–F4.

## Run all eight in the foreground, serial, GPU0

    CUDA_VISIBLE_DEVICES=0 \
    python -u experiments/no31_cora_original/followup8.py --run

Independent configs and full train.log outputs:

    logs/NO31_CORA_FOLLOWUP8_<timestamp>/
      F1/{experiment.json,train.yaml,train.log,result.json}
      ...
      F8/{experiment.json,train.yaml,train.log,result.json}
      plan.json
      results.jsonl
      summary.csv

Separate checkpoints under:
    output/NO31_CORA_FOLLOWUP8_<timestamp>/F1/.../checkpoint_0.pth
    ...
    output/NO31_CORA_FOLLOWUP8_<timestamp>/F8/.../checkpoint_0.pth

Any failed trial stops subsequent ones, preserving completed results.
Re-run from a desired trial if necessary:

    python -u experiments/no31_cora_original/followup8.py --run --only F1
    python -u experiments/no31_cora_original/followup8.py --run --start F5

These selective retries create a different timestamp prefix and do not
merge into the previous summary automatically; combine results by ID.

## Decision rule after eight results

1. For AUC, first compare F1/F2 with R2 and F3/F4 with R7/R0.
2. Test additive value: F1 against R2(alpha32,V/O) and R7(alpha64,QKVO).
3. Use F5/F6 to determine if both V and O are needed; do not infer
   "attention shift" from an R1 vs R2 comparison alone.
4. F7/F8 indicate whether relative R2-vs-R7 performance survives
   a seed change; two seeds are **not** robust significance testing.
5. UAUC matters alongside AUC, especially because historical NO31
   UAUC=0.685106 and early CoRA gains are marginal.
6. Take the top 1–2 variants to 3 epochs with proper per-epoch Valid
   checkpoint selection and then a final Test. Do not select on Test.

Actual GPU training and outcome verification have not been performed
by the assistant.
