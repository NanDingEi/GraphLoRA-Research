# Reproduce the original published CoRA implementation (MF / MovieLens-1M OOD2)

This is **not** the customized GraphLoRA CoRA adapter from NIGHT26.

- Upstream: https://github.com/VanillaCreamer/CoRA
- Pinned source SHA: `4d69c83504259ea8529725f95cfc277ffd1b8077` (2024-12-11).
- Entrypoint **inside official source**: `train_collm_mf_din.py`
- Model: `minigpt4/models/minigpt4rec_vx.py` (`mini_gpt4rec_vx`)
- PLoRA: `minigpt4/models/ppeft/plora.py`, Q-Former dynamic A: `minigpt4/models/ppeft/weight_generator.py`.
- Reference config: `train_configs/plora_pretrain_mf_ood_movie.yaml`.
- We do not copy/edit these official algorithm files, inject online LightGCN, CTR/KD, custom Group8, or preference routers.

## Critical prerequisites

Original official CoRA is two-stage **after** pretrained MF:

1. **MF pretraining:** a matching `MatrixFactorization` `state_dict` with `user_embedding.weight` and `item_embedding.weight`, both embedding dimension 256. Use the same ML-1M OOD2 user/item indexing and data split. Original MF pretraining script seeds 2023; CoRA tuning seeds 42.
2. **Stage-1 CF-to-LLM projection checkpoint:** an original-code compatible checkpoint containing `model` state with trained `llama_proj` and `bias_proj` parameters. The public MF config refers to the authors' machine-local Stage-1 checkpoint, which is **not** supplied by this wrapper.
3. **Vicuna** weights at `/home/bbx/user_lx/models/vicuna-7b-v0`, or override.
4. **Four** original OOD2 files: `train_ood2.pkl`, `valid_ood2.pkl`, `valid_small_ood2.pkl`, `test_ood2.pkl`. Original `train_collm_mf_din.py` loads `valid_ood2.pkl` for ID cardinality; official dataset builder loads `valid_small_ood2.pkl` for actual validation. Do not silently substitute one for the other. Data columns: `uid, iid, label, title, his, his_title`.
5. A compatible legacy Python 3.9 environment. Official `environment.yml` pins PyTorch 1.12.1, Transformers 4.28.0. Prefer `conda env create -f <official_source>/environment.yml` and `conda activate minigpt4`. **Do not change the existing GraphLoRA environment in place.**

The original MF config shown in the repository has `evaluate: True` and a private Stage-1 path. The runner changes only the environment-dependent paths, `run.evaluate=False` for training, GPU0, valid-only testing restriction, and the selected hardware profile. Missing Stage-1 checkpoints **fail closed** rather than initializing random frozen projectors.

The user has already identified the MF state_dict at
`/home/bbx/user_lx/GraphLoRA/checkpoints/0912_ml1m_oodv2_best_model_d256lr-0.001wd0.0001.pth`;
the runner uses this as its **default**, but validates dimensions against the official
dataset rather than assuming equivalence.

The original CoRA MF training profile is `freeze_rec=True`, `freeze_proj=True`, `freeze_lora=False`, `use_lora=True`, rank 16, alpha 32, Q/K/V/O, dropout .05, seed42, cosine-warmup, init LR .001, min LR .00008, warmup LR .00001, 1000 max epochs, 100 iterations/epoch, train batch 8 and eval batch 32. These are **not** the GraphLoRA G2 settings (alpha48, different training schedule/losses).

## Where to run: existing NIGHT26 directory

No new GraphLoRA worktree or project. The official Git source is placed in a *log subdirectory* `logs/CORA_OFFICIAL_UPSTREAM/` and its exact commit is verified. Each run's outputs stay in timestamped `logs/CORA_OFFICIAL_TRAIN_*/` and `output/CORA_OFFICIAL_TRAIN_*/`.

```bash
cd /home/bbx/user_lx/GraphLoRA_NIGHT26
git status --short
git fetch https://github.com/NanDingEi/GraphLoRA-Research.git exp/night26-mf-graph-lambda-adaptive-20261028
git switch --detach FETCH_HEAD
export PYTHONPATH="$PWD"

# View planned source, model architecture, hyperparameters and missing inputs
python experiments/no31_cora_original/cora_official_repro.py --plan

# Download official source only (no training); NEVER modifies official files
python experiments/no31_cora_original/cora_official_repro.py --prepare
```

First, install/use upstream's environment separately. **Do not start training without a trained Stage-1 checkpoint.** Point the two required checkpoint arguments at genuine, corresponding files (example placeholders below):

```bash
MF=/home/bbx/user_lx/GraphLoRA/checkpoints/0912_ml1m_oodv2_best_model_d256lr-0.001wd0.0001.pth
STAGE1=/absolute/path/to/official_compatible_stage1_checkpoint_best.pth
VICUNA=/home/bbx/user_lx/models/vicuna-7b-v0
DATA=/home/bbx/user_lx/GraphLoRA/dataset/ml-1m

# CPU-only prereq/weights/config validation: does NOT load Vicuna
python experiments/no31_cora_original/cora_official_repro.py --check \
  --mf-checkpoint "$MF" --stage1-checkpoint "$STAGE1" \
  --vicuna "$VICUNA" --data-root "$DATA"

# Exact published MF tuning hyperparameters, GPU0 only, foreground:
CUDA_VISIBLE_DEVICES=0 python -u \
  experiments/no31_cora_original/cora_official_repro.py --train \
  --profile official \
  --mf-checkpoint "$MF" --stage1-checkpoint "$STAGE1" \
  --vicuna "$VICUNA" --data-root "$DATA"

# If GPU0 24GB OOMs with batch8, run a clearly labeled *non-exact*
# GPU0 hardware variant. Train batch2/eval batch3; rest unchanged.
CUDA_VISIBLE_DEVICES=0 python -u \
  experiments/no31_cora_original/cora_official_repro.py --train \
  --profile gpu0_24gb \
  --mf-checkpoint "$MF" --stage1-checkpoint "$STAGE1" \
  --vicuna "$VICUNA" --data-root "$DATA"

# Cheap smoke only, NOT an accuracy reproduction: batch2, 10 iters, 1 epoch
CUDA_VISIBLE_DEVICES=0 python -u \
  experiments/no31_cora_original/cora_official_repro.py --train \
  --profile smoke \
  --mf-checkpoint "$MF" --stage1-checkpoint "$STAGE1" \
  --vicuna "$VICUNA" --data-root "$DATA"
```

Run **one** profile at a time; do not launch other GPU0 jobs in parallel. Do not force a one-epoch comparison against G2: published CoRA uses 100 iterations/epoch, unlike prior GraphLoRA experiments.

Evaluation of an official CoRA checkpoint (on **VALID**, not TEST):

```bash
CUDA_VISIBLE_DEVICES=0 python -u \
  experiments/no31_cora_original/cora_official_repro.py --evaluate \
  --mf-checkpoint "$MF" --eval-checkpoint /absolute/path/to/CoRA/checkpoint_best.pth \
  --vicuna "$VICUNA" --data-root "$DATA"
```

The source is kept **byte-for-byte unchanged**. The generated YAML and a manifest document every necessary path and setting deviation. The runner logs stdout and total elapsed seconds. It does not promise AUC or claim experiments were run on the server.

## Scientific limitations

- The authors' exact Stage-1 checkpoint is not distributed through this driver. If absent, one must reproduce Stage-1 separately and document that it is a distinct replication, not the authors' original checkpoint.
- MF embedding IDs and shape must match the same exact dataset split; reusing an unrelated pretrained checkpoint invalidates the experiment.
- Single 24GB GPU may need the hardware profile. That differs from the official batch-size configuration, and its outcome must be labeled separately.
- Original CoRA has no GraphLoRA-specific CTR/KD/semantic losses. AUC/UAUC values from this script can be compared only after confirming the same dataset, validation split, initialization and evaluation protocol.
