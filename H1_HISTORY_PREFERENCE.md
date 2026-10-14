# H1: History–Target Collaborative Preference Auxiliary (binary CTR)

## Research goal

Improve the final LLM **binary** preference prediction (Valid/Test AUC, UAUC)
while retaining the proven GraphLoRA S2 node-level layer-31 injection.

Earlier work in this repository already has (a) item–item semantic relation
alignment and (b) GNN user–item BCE plus optional candidate-group distillation.
H1 is distinct: it supervises **candidate vs. the current user's positively
rated history** in the existing NGCF representation, rather than building a
new relation vector or gate at inference.

For each training sample (u, i, H_u), use the 256-dimensional NGCF
penultimate embeddings from the existing semantic alignment computation:

  c(u,i) = (1/|H_u|) sum_{j in H_u} cosine(z_i, z_j)

  L_H1 = BCEWithLogits(4 * c(u,i), y_ui)

  L_total = L_CTR_S2 + 0.0001 * L_sem + 0.0005 * L_proj + 0.01 * L_H1

The last two components are those already present in S2. The CTR loss
remains **BCE on the original Yes logit**. H1 is a training-only auxiliary
objective; no LLM classifier modification, inference gate, pairwise
ranking examples, additional neural layers, or stop-gradient teacher.

If a sample has no history ID tokens, exclude it from the H1 auxiliary
loss. Identify the candidate by its actual TargetItemID, not by assuming
the last item token belongs to the candidate (prompt order can vary).
Do not use the validity/test graph during training.

The auxiliary H1 loss updates the GNN/MF representations. Through the
unchanged GraphLoRA injection and L_CTR, the LLM LoRA continues to train
end-to-end to optimize its final binary recommendation prediction.

## Single foreground experiment

```bash
cd /home/bbx/user_lx/GraphLoRA
git fetch origin
git switch exp/H1-history-preference-aux-20261014
git pull --ff-only
python -u run_H1_history_preference.py --gpu 0
```

Before using GPU, the runner runs:
`python -m unittest tests/test_history_preference_aux.py -v`.

S2-effective 3-epoch recipe: train size 33,891; Valid size 5,200,
same MF/NGCF/Vicuna settings, batch 3, accumulation 3, seed 42.
The runner selects checkpoint using **Valid AUC** then evaluates Test.
To save GPU time, if Epoch 0 Valid AUC is below 0.70, stop training,
retain epoch-0 checkpoint, and do not spend two further epochs.

Files:
- `log/H1_HISTORY_PREFERENCE_train_<timestamp>.log`
- `output/H1_HISTORY_PREFERENCE_<timestamp>/...`
- `out/H1_history_preference_<timestamp>.txt` and `.csv`

The report contains per-epoch training losses and Valid AUC/UAUC,
best-valid checkpoint, and Test AUC/UAUC. No full Test console log is
saved. No GPU training was executed by committing this branch.

## Interpretation

- If H1 improves Valid AUC/UAUC over S2 (0.73288356/0.68795261),
  it suggests history–candidate collaborative preference supervision
  helps the unchanged LLM injection pathway.
- If the Valid score deteriorates, do not keep tuning H1 variations.
  The method would not have demonstrated improvement.
- Avoid comparing H1 against M1 using the M1 Yes-minus-No score;
  H1 matches the original S2 Yes-only metric protocol.
