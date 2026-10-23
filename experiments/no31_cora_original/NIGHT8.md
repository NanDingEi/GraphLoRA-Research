# NO31 + CoRA — 8 sequential one-epoch ablations

These trials continue in the **existing** working directory:

    /home/bbx/user_lx/GraphLoRA_NO31_CORA

The target branch is always:

    exp/no31-ctrkd-cora-20261023

No new Git worktree, no background training. Every run is isolated into
timestamped subdirectories under the **existing** worktree's logs/ and output/
folders. Previously trained NO31 and CoRA checkpoints are preserved.

## Baselines (ML-1M OOD2, Valid)

| Control | AUC | UAUC |
|---|---:|---:|
| NO31 + CTR=.05/KD=.05 standard LoRA | 0.719760366 | 0.685106415 |
| original CoRA, broken L0–L7 autocast | 0.594506910 | 0.574051567 |
| original CoRA + autocast fix (R0) | 0.633755947 | 0.593198036 |

R0 is **historical**, not rerun in night8. All night8 trials start from the
same original pretrained Vicuna-7B and MF checkpoint, not from R0 checkpoint.

## Eight mechanisms

| ID | Intervention | LoRA-P targets |
|---|---|---:|
| R1 | CoRA on Q/V, all 32 layers | 64 |
| R2 | CoRA on V/O, all 32 layers | 64 |
| R3 | CoRA on Q/K/V/O, layers 16–31 only | 64 |
| R4 | CoRA on Q/V, layers 16–31 only (interaction R1×R3) | 32 |
| R5 | 2 CF encoder tokens, rather than 1 concatenated token | 128 |
| R6 | original Q/K/V/O, alpha=16 instead of 32 | 128 |
| R7 | original Q/K/V/O, alpha=64 instead of 32 | 128 |
| R8 | original Q/K/V/O, LoRA-P-only learning rate 2× | 128 |

R5 splits [u;i] into [u;0] and [0;i], both 512-dimension encoder
tokens; it introduces **no additional input projection parameters**.
The 8-layer, 8-head QFormer, four query tokens, normalized full
dynamic A of shape [batch,4096,16], shared generator, and static trainable
zero-initialized P matrices remain as in the original CoRA implementation.
There is **no** old rank gate, token-only injection, or L31 GNN injection.

R1–R4 leave inactive modules at frozen pretrained projection W (with
the unused conventional LoRA branch frozen and zero-initialized).

R8 changes only the optimizer group named cora_p_wd, not QFormer, MF,
GNN, projection MLP or other learning rates.

## All common controls

- GraphLoRA NO31: gnn_scale=0, input alpha=0, CTR=0.05, KD=0.05
- Semantic correction weight 1e-4, autocast fix **ON**
- Original projector regularization 0.0005
- Train batch 2; Valid batch 3; gradient accumulation 4
- Seed 42; init_lr 1e-4; 3-epoch scheduler horizon but run **only epoch 0**
- Complete ML1M OOD2 Valid; **never run Test**
- All experiments train from scratch, independently, not a joint 8-epoch run

## Update existing directory once

Run with a clean Git index. A local change to an unrelated file such as
the user's already-edited run.py need not be lost, but do not overwrite
local modifications if Git reports a conflict.

    cd /home/bbx/user_lx/GraphLoRA_NO31_CORA
    conda activate minigpt4
    git status --short
    git fetch https://github.com/NanDingEi/GraphLoRA-Research.git \
      exp/no31-ctrkd-cora-20261023
    git merge --ff-only FETCH_HEAD

## Preflight

No need to re-run prepare: night8 uses the existing
logs/NO31_CORA_ORIGINAL_READY/train.yaml. It **requires** the already
set fix_autocast_cache=True, train_batch=2, eval_batch=3 and accum=4.

    python experiments/no31_cora_original/test_night8_cpu.py

The output should contain [NIGHT8-CPU-PASS].
Optional, but recommended: verify that the **fixed** R0 checkpoint has
nonzero CoRA-P norms for ALL early layers:

    python experiments/no31_cora_original/night8.py --precheck \
      output/NO31_CORA_ORIGINAL_20261023_234005_110337/20261023234/checkpoint_0.pth

Check the eight variant descriptions without GPU use:

    python experiments/no31_cora_original/night8.py --plan

## Run (single GPU0, serial, foreground)

    CUDA_VISIBLE_DEVICES=0 \
    python -u experiments/no31_cora_original/night8.py --run

If one trial fails, the runner stops instead of silently skipping it.
Previous successful trial results are saved in
logs/NO31_CORA_NIGHT8_<timestamp>/results.jsonl and summary.csv,
alongside each trial's train.yaml, train.log and result.json.
Each trial writes its own checkpoint_0.pth to output/NO31_CORA_NIGHT8_<timestamp>/<Rn>/.

After resolving failure, run the missing experiment(s) explicitly:

    python -u experiments/no31_cora_original/night8.py --run --only R5

Or run the remaining suffix from a given trial (new results folder):

    python -u experiments/no31_cora_original/night8.py --run --start R5

The runner checks GPU0 for another compute process and refuses
overlapping training. It does not launch subprocesses in the background.

## Comparing results

Use the appended summary.csv, rank AUC first, UAUC second, and
compare with R0 AUC 0.633755947 and NO31 0.719760366.

Interpretations:
- R1 vs R0: K/O projections might cause unwanted perturbations.
- R2 vs R0: Query vs value/output pathways.
- R3 vs R0: shallow-layer parameterization vs deeper-only.
- R4 vs R1 and R3: Q/V upper-half interaction, NOT an independent axis.
- R5 vs R0: CF encoder cross-attention granularity.
- R6/R7 vs R0: dynamic rank-16 update scale.
- R8 vs R0: learning rate of the **static P**, not generator.

Eight one-epoch ablations isolate promising designs but do NOT prove an
approach would win after longer training. Re-run promising candidates
with multiple seeds and an identical training budget before making
conclusions.
