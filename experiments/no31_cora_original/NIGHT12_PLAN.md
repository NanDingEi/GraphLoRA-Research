# NIGHT12: four structural CoRA improvements (no hyperparameter search)

NIGHT12 follows the best current CoRA result G2 (Valid AUC 0.730121895,
UAUC 0.688041011; alpha=48, rank=16, Q/K/V/O, all 32 layers). The purpose is
to improve how collaborative information generates and enters the LLM, not to
search numeric hyperparameters.

The experiments remain in the existing `experiments/no31_cora_original/` directory
and use the existing branch. All runs use seed=42, fresh initialization,
alpha=48, rank=16, all 32 Q/K/V/O projections, CoRA-P LR multiplier=1.0,
no residual clipping/scaling, the original concatenated MF pair as the
condition source, and the unchanged NO31 MF/GNN/Projector, CTR/KD losses,
semantic alignment, projector regularization and data split. Each arm trains
one epoch and evaluates Valid only. No Test set is used.

## Why investigate architecture first?

Current CoRA creates one dynamic matrix `W_(u,i) in [B,4096,16]` from a frozen
MF user/item pair, then shares it across every selected layer and Q/K/V/O
projection. For each token, the residual is `P_(layer,projection)(X @ W_(u,i))`.
This is a powerful conditioning mechanism, but it has three structural limits:
the QFormer receives the pair as one encoder token; the same generated
low-rank basis is reused across every layer; and every token receives the
residual without a learned token-specific gate.

## Four experiments

| ID | Architecture change | Hypothesis | What remains fixed |
|---|---|---|---|
| M1_TYPED_TOKENS | Encode MF user and candidate item as two separate 512-d QFormer encoder tokens using a shared 256→512 projection and learned role embeddings. | QFormer cross-attention can distinguish user evidence from candidate-item evidence rather than compressing both into one concatenated token. | Same frozen MF embeddings, QFormer depth/query count, final matrix generator, P modules and training settings. |
| M2_INTERACTION_TOKEN | Use the same user/item tokens as M1 and add a third token from elementwise user-item interaction `u * i`. | The generator can directly attend to pair compatibility in addition to individual user/item identity. | Same token projection for user/item, same rank/alpha/targets and losses. Only the interaction token is added relative to M1. |
| M3_LAYER_CONDITIONED | Generate the normal user-item-conditioned basis once, then modulate its rank channels using the QFormer pair summary and a learned Transformer-layer embedding. | Different Transformer layers can use different user-item-conditioned low-rank directions instead of receiving the same basis everywhere. | Same QFormer pair input and per-layer/per-projection P modules; modulation starts near identity. |
| M4_TOKEN_GATE | Apply a shared token-wise gate `2 * sigmoid(g(x_token))` to the CoRA residual before addition. The gate is zero-initialized, so its initial value is exactly 1. | The model can inject collaborative evidence more strongly into useful token states and suppress it for less relevant states. | Same dynamic basis, QKVO targets, P matrices and residual equation apart from the learned token gate. |

M1 and M2 directly change how collaborative signals are represented before
LoRA-A generation. M3 changes how the generated basis is specialized to
different LLM layers. M4 changes how the resulting residual is applied to
different tokens. These test four different architectural hypotheses rather
than four scalar settings.

## Commands on the existing checkout

First preserve local work and fast-forward from GitHub; do not reset the repo:

    cd /home/bbx/user_lx/GraphLoRA_NO31_CORA
    conda activate minigpt4
    git status --short
    git fetch https://github.com/NanDingEi/GraphLoRA-Research.git \
      exp/no31-ctrkd-cora-20261023
    git merge --ff-only FETCH_HEAD

Review the plan and run the CPU-only checks before starting GPU work:

    python experiments/no31_cora_original/night12.py --plan
    python experiments/no31_cora_original/test_night12_cpu.py
    python experiments/no31_cora_original/test_night8_cpu.py
    python experiments/no31_cora_original/test_night10_cpu.py

Run all four serially in the foreground on GPU0:

    CUDA_VISIBLE_DEVICES=0 \
    python -u experiments/no31_cora_original/night12.py --run

To start with the first two structural changes only:

    CUDA_VISIBLE_DEVICES=0 \
    python -u experiments/no31_cora_original/night12.py --run --first 2

After inspecting those results, continue with a new timestamped run directory:

    CUDA_VISIBLE_DEVICES=0 \
    python -u experiments/no31_cora_original/night12.py --run --start M3_LAYER_CONDITIONED

One architecture only can be selected using `--only M4_TOKEN_GATE`. Each arm is a
fresh one-epoch run, not checkpoint continuation. GPU overlap is rejected by
the shared runner.

## Logs and interpretation

The shared runner saves the complete stdout/stderr stream to a per-arm
`train.log`, a `result.json` beside it, and a `summary.csv` at the sweep root.
Checkpoints are saved to the corresponding `output/NO31_CORA_NIGHT12_<timestamp>/` tree.
The existing NIGHT10/NIGHT11 results are not overwritten.

Compare both AUC and UAUC against G2. Because this is an architecture study,
do not tune alpha, LR, residual multipliers or random seed based on these
first-pass results. If a mode improves both metrics or gives a meaningful
trade-off, repeat that architecture for two epochs before any broader claim.
If a mode collapses, inspect its generated-W norms, layer modulation norms or
token-gate distribution before trying unrelated parameter changes.
