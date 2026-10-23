# NO31 + CTR/KD + **original CoRA implementation**

## Purpose

Reproduce the **source-code mechanism** of
[VanillaCreamer/CoRA](https://github.com/VanillaCreamer/CoRA) (AAAI 2025)
on the historical GraphLoRA NO31 + CTR/KD recommendation task.
This is **not** the earlier CoRA-inspired rank-gate experiment.

Original CoRA source:
- `minigpt4/models/ppeft/qformer.py`: vendored unchanged (Salesforce BSD-3)
- `minigpt4/models/ppeft/weight_generator.py`: reproduced in this directory;
  only the learned-query batch expansion is fixed for batch=3
- `minigpt4/models/ppeft/plora.py`, `PLoraLayer.forward` lines 682–709:
  **equivalent dynamic low-rank forward**, implemented by applying a forward
  hook to the existing PEFT/LLaMA modules in `adapter.py`
- `minigpt4/models/pllama.py`: its `w` argument is semantically equivalent
  to the per-sample conditional tensor cached on each projection in this
  worktree (we do not replace GraphLoRA's whole LLaMA implementation)

## Mechanism (matches CoRA, NOT previous CoRA-inspired adapter)

Original CF input: frozen pretrained MF user and target-item vectors,
256 dimensions each, concatenated into `[batch,512]`.

The original 8-layer, 8-head Q-Former with 4 learnable queries consumes that
vector as *one* cross-attention encoder token. A shared projection produces
a whole low-rank matrix for each user-target pair:

```
A_(u,i) = normalize(mean(linear(QFormer([emb_user ; emb_item]))))
shape: [batch_size, 4096, 16]
```

For **each** Transformer attention projection `q_proj,k_proj,v_proj,o_proj`,
for **all 32 layers** and for **all text and ID tokens**:

```python
base = F.linear(x, frozen_weight, bias)
delta = P_layer_projection(torch.bmm(dropout(x), A_user_item)) * (32/16)
return base + delta
```

`P_layer_projection`: static trainable `nn.Linear(16,4096,bias=False)`,
initialized to exactly ZERO. There are 128 such P's.
The **original static LoRA A/B branch is frozen AND BYPASSED** (not added).
The previous experimental CoRA rank gate / ID-only residual / CoRA-Up are
absent from this model path. The dormant previous CoRA hook is removed in
this branch's `minigpt4rec_v2.py`.

### Important distinctions vs literal author setup

This reproduces **CoRA's generation and injection mathematics**, but the
experiment remains the user's **NO31 + CTR/KD** training task.

- Original author sample configuration:
  MF frozen, QKVO full layers, dynamic A rank16, `alpha=32`,
  `init_lr=1e-3`, 1000 max epochs, specific source prompts/dataset.
- Here: **CoRA conditions on a frozen copy** of the pretrained MF user/item
  tables while the underlying NO31 MF/GNN training continues unchanged.
  This keeps CoRA's CF conditioning frozen without deleting the baseline's
  trained MF/semantic/CTR/KD mechanisms.
- The original NO31 schedule, data, prompts, Vicuna-7B model, seed42,
  OOD2 Valid split, batch3, `init_lr=1e-4` / 3-epoch scheduler horizon,
  GNN CTR=0.05, LLM->GNN KD=0.05, GNN semantic alignment=1e-4,
  projector regularization=5e-4, `gnn_scale=0`, input alpha=0 remain.
- No original-CoRA-trained checkpoint is loaded. All 128 P matrices start at
  zero. The ordinary LoRA parameters in the underlying PEFT scaffold are
  not used and have `requires_grad=False`.
- The original author's `query_tokens` tensor is shape `[1,4,512]`
  and is not expanded in the public `weight_generator.py`; we
  expand it to `[batch,4,512]` before Q-Former cross-attention so
  `batch=3` works. This is a **compatibility fix**.

Do NOT call the resulting AUC a literal reproduction of the authors'
published numbers: the task and joint GNN supervision are different.

## Download to a NEW worktree

```bash
cd /home/bbx/user_lx/GraphLoRA
git fetch https://github.com/NanDingEi/GraphLoRA-Research.git \
  exp/no31-ctrkd-cora-original-20261023
git worktree add --detach \
  /home/bbx/user_lx/GraphLoRA_NO31_CORA_ORIGINAL FETCH_HEAD
cd /home/bbx/user_lx/GraphLoRA_NO31_CORA_ORIGINAL
conda activate minigpt4
```

## Validate, prepare, train

```bash
# CPU-only: original QFormer shape, normalized full A, zero-P identity and grads
bash experiments/no31_cora_original/run.sh prepare

# Model-only init probe; loads Vicuna and pretrained MF but does not train
bash experiments/no31_cora_original/run.sh probe

# Run ONE full epoch on CUDA0, foreground, Valid only
bash experiments/no31_cora_original/run.sh run
```

The initial CPU check is run before every command.
The `prepare` command writes exactly one guarded config to
`logs/NO31_CORA_ORIGINAL_READY/train.yaml`.
The source-graph NPZ files are symlinked into the new worktree if needed
and their SHA256 hashes are compared with the old GraphLoRA files.

Each `run` creates distinct timestamped result directories:
`logs/NO31_CORA_ORIGINAL_<stamp>/`, `output/NO31_CORA_ORIGINAL_<stamp>/`.
Original NO31 or previous CoRA checkpoints and logs cannot be overwritten.

Compare **Valid AUC and UAUC** with historical NO31 baseline:
`Valid AUC=0.719760366`, `Valid UAUC=0.685106415`.

### GPU resource caveat

This is **much larger** than the earlier rank-gate:
the shared CoRA final layer alone is 512 x (4096*16) = 33,554,432
weights (plus bias), 8-layer QFormer plus 128 P matrices adds more.
An RTX 3090 24GB might OOM. The experiment does NOT silently fall
back to rank gate, fewer layers, smaller rank or ID-only injection.
If it OOMs, preserve the log and discuss a separately identified
memory-equivalent implementation (activation checkpointing etc.).
