# D1: Decision-Query / Evidence-Value collaborative injection

**Baseline**: exact code snapshot `d818fa2943ede4e9a810b032a3422e94d0c77ed0`, which contains the S2 historical Valid AUC=0.732883563 checkpoint/recovered config. This branch **does not** inherit the G1/G2/G3 graph, gradient, or auxiliary-CTR modifications.

## Purpose

Investigate the causal reachability of NGCF collaborative information injected into the last Vicuna Transformer layer (layer 31).

In original GraphLoRA, NGCF rank-8 vectors are injected into the LoRA bottleneck of both `q_proj` and `v_proj` **at the ID token locations**. At the last causal-attention layer, ID-query changes do not directly affect the final prompt decision logit; ID-value changes can affect it. D1 compares:

* **D1-A**: remove *only* collaborative ID-Q additions; retain original ID-V additions. Ordinary Q/V LoRA updates are untouched.
* **D1-B**: retain D1-A's ID-V additions, plus inject `P([z_user,z_target,z_user*z_target])` into the **last prompt token's Q** bottleneck of layer 31 (the token whose Yes logit is used for binary CTR BCE). P is a trainable zero-initialized 24-to-8 projection; no new GNN parameters, attention head, score or loss.

## What is frozen or retained?

* Load **the same trained S2 best checkpoint** for A and B. Abort if checkpoint lacks `gcn`, `rec_encoder`, `llama_proj`, or LLM LoRA weights.
* Freeze **all MF and NGCF parameters** after loading. Do not train graph edges, message passing weights, or small-model embeddings. Only the existing Vicuna LoRA, LLM-side `llama_proj`, and the 192-weight D1-B pair projector remain trainable. Layer-31 original LoRA attention biases retain their existing optimizer behavior.
* Original S2 three-epoch learning rate / gradient accumulation / seed / split-specific adjacency / Yes-logit BCE / AUC / UAUC. Semantic teacher pass remains configured exactly as S2 to preserve its existing forward computation, but with frozen GNN the semantic auxiliary is constant and cannot optimize the small model.
* No added graph, teacher model, ranking loss, history BCE or new CTR head.
* For the new D1-Q pair vector only, normalize raw GNN output consistently in TRAIN and EVAL before projection. Preserve S2's original ID-V preprocessing (note: historical S2 train vs eval normalization is not the same; changing it here would confound the experiment). No tuning of test data.

## Why is D1 zero-init?

P=0 makes **D1-B mathematically identical to D1-A before fine-tuning**, while removing only last-layer ID-Q injections makes D1-A decision-logit equivalent to historical S2 under causal attention. This is checked by an actual checkpoint **Valid preflight** before costly GPU training. Both branches then fine-tune on original train split for three epochs, select best checkpoint by Valid AUC, evaluate Test once.

The actual user's S2 checkpoint file must exist at:
`output/S2_groups_acc3/20261010223/checkpoint_best.pth`

## Run (ONE GPU 0, foreground and serial)

```bash
cd /home/bbx/user_lx/GraphLoRA
git fetch origin
git switch exp/D1-decision-query-evidence-value-20261015
git pull --ff-only origin exp/D1-decision-query-evidence-value-20261015

# CPU-only tests, then D1-A Valid preflight + 3 ep + Test,
# then D1-B Valid preflight + 3 ep + Test; failure stops the sequence.
GPU=0 bash run_D1.sh both

# Or only one condition:
GPU=0 bash run_D1.sh A
GPU=0 bash run_D1.sh B
```

The preflight requires Valid AUC within 0.003 of S2=0.732883563; otherwise the script aborts **before training** to avoid repeating failures caused by non-comparable checkpoints. Both experiments share exactly the same S2 initialization; contrast **D1-B vs D1-A** is the primary causal test. S2 checkpoint (~0.733 valid, ~0.74 test historically) is a reference, not a matched continued-fine-tuning control. If both A/B degrade similarly, the relevant limitation may be warm-start overfitting; do not ascribe this to D1 alone.

Logs and checkpoints:
* `logs/D1_A_ID_V_ONLY_preflight_*.log`, `logs/D1_B_DECISION_Q_ID_V_preflight_*.log`
* `logs/D1_*_train_*.log`, `logs/D1_*_test_*.log`
* `output/D1_*_<stamp>/<job>/checkpoint_best.pth`
* `out/D1_*_<stamp>.txt`

Notes: 3 epochs are full *training* epochs. Only the final best-validation checkpoint is evaluated on Test. Nothing runs asynchronously or in the background. No GPU training has occurred simply because the branch was committed.
