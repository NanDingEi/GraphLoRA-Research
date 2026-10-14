# M1: Yes-No logit-margin binary CTR (GraphLoRA/S2)

## Motivation and scope

GraphLoRA/S2 predicts binary labels (click/like = 1, otherwise 0) and is
evaluated by AUC and UAUC. The original code uses the raw \`Yes\` token logit
as the scalar for BCE and AUC. M1 replaces the scalar with:

\`s(u,i) = logit_Yes(u,i) - logit_No(u,i)\`

This is the two-answer conditional log-odds:
\`P(Yes | {Yes, No}) = sigmoid(s)\`.

Training loss remains \`BCEWithLogits(s, label)\`; AUC and UAUC are calculated
from the same score \`s\`. This is **not** a ranking task and does not
introduce a pairwise sampling change, an auxiliary loss, or trainable weights.

The default \`model.answer_score_mode: yes\` retains the original behavior.
\`margin\` enables M1 in both \`forward_v2\` and \`generate_for_samples_v2\`.
For FP16 LLM logits, subtract in FP32. M1 configuration retains the
reconstructed S2 batch size, gradient accumulation, learning rates, optimizer
groups, seed, prompt, graph, and semantic/projector losses. The A1 candidate
module is disabled and no A2/A3 gate is included.

## Foreground command

\`\`\`bash
cd /home/bbx/user_lx/GraphLoRA
git fetch origin
git switch exp/M1-yes-no-margin-20261014
git pull --ff-only
python -u run_M1_answer_margin.py --gpu 0
\`\`\`

The runner checks four CPU tests, trains 3 epochs from scratch, selects the
checkpoint with highest VALID AUC, then evaluates this checkpoint once on Test.
Use \`--skip-test\` to stop after Valid results. Nothing runs in the background.

- Training log: \`log/M1_YESNO_MARGIN_train_<timestamp>.log\`
- Checkpoints: \`output/M1_YESNO_MARGIN_<timestamp>/<run-id>/\`
- Compact report: \`out/M1_yesno_margin_<timestamp>.txt\` and \`.csv\`

The report includes best Valid AUC, Valid UAUC, Test AUC, Test UAUC, training
losses, and the per-epoch Valid AUC. Training has not been executed on the GPU
machine merely by committing this branch.

## About M2

AUC evaluates pairwise discrimination, but CTR remains a binary classification
task. Same-user positive-negative ranking is, at most, an optional auxiliary
training objective. With S2's random batch size of 3, naive same-user
batch-local pairing almost always yields no usable pairs. M2 is deliberately
not enabled or trained in this branch. Evaluate the isolated M1 hypothesis
first before deciding whether a correctly paired auxiliary loss is worthwhile.
