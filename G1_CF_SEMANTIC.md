# G1: Train-only CF + movie content graph for GraphLoRA

## Single-variable hypothesis

G1 retains S2's MF+NGCF+Vicuna-7B, layer-31 GraphLoRA injection (scale 0.1), Yes-logit binary CTR BCE, semantic loss (1e-4), projector regularizer (5e-4), LoRA, optimizer, seed, datasets, AUC and UAUC protocol. No new neural branch, gate, ranking loss, auxiliary CTR loss, or inference classifier.

Only the offline TRAIN graph changes:

1. Read train_ood2.pkl, existing movie_graph_train.npz, and optionally the MovieLens movies.dat genres (by normalized title rather than original MovieID). Never read validation/test labels.
2. Assert the original S2 graph is symmetric, consistent with D^-1/2 support(A) D^-1/2, and its user-item edges exactly match the positive TRAIN samples. Abort on mismatch before GPU use.
3. Consider two items only if >=2 distinct training users gave both positive interactions. Compute title character ngram TF-IDF cosine and, when metadata exists for both, 0.75 genre Jaccard + 0.25 title similarity. Keep similarity >=0.30 and at most 5 undirected content edges per item.
4. Weighted item-item edge = 0.25 * content_similarity * min(1, log(1+common_train_users)/log(6)); fused RAW graph = original positive CF support + weighted new item-item edges; recompute symmetric D^-1/2 W D^-1/2 normalization.
5. TRAIN, VALID and TEST reuse the exact same generated TRAIN graph. Save graph, config and input SHA256 provenance for reproducibility.

This is metadata/lexical graph fusion, not semantic embeddings from the LLM. All classification paths remain original S2.

## Foreground command

```bash
cd /home/bbx/user_lx/GraphLoRA
git fetch origin
git switch exp/G1-cf-semantic-graph-20261014
git pull --ff-only
python -u run_G1_cf_semantic_graph.py --gpu 0
```

The runner first performs CPU unit tests, then builds and validates the G1 graph before starting CUDA training. Trains up to 3 epochs, early-stopping after Epoch 0 if Valid AUC <0.72, choosing the best checkpoint by Valid AUC. If best Valid AUC <0.72, Test is skipped; otherwise Test uses the same G1 graph. To skip Test independently add --skip-test. No nohup, background job or full Test console log.

Graph and provenance: out/G1_graph_<timestamp>/movie_graph_train_G1.npz and .json
Train log: log/G1_CF_SEMANTIC_train_<timestamp>.log
Checkpoint: output/G1_CF_SEMANTIC_<timestamp>/<run-id>/checkpoint_best.pth
Compact per-epoch report: out/G1_cf_semantic_graph_<timestamp>.txt and .csv

Decision: S2 Valid AUC=0.732883563, UAUC=0.687952611. G1 must improve Valid AUC without hurting UAUC; ~0.735 AUC is an objective, not a claimed outcome. Test results must not be used to select method hyperparameters. No GPU training has been run merely by committing this branch.
