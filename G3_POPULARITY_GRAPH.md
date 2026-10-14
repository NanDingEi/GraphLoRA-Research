# G3 — Popularity-debiased CF graph, original GraphLoRA S2 CTR

The ONLY intervention relative to reconstructed S2 is train-only user-item graph edge reweighting, not new relations or losses. Keep original Yes-token BCE, same LoRA/NGCF architecture, semantic and projector supervision, 3 full epochs, best-Valid checkpoint followed by Test.

For a positive TRAIN interaction (u,i), estimate item positive frequency d_i from train_ood2.pkl. Let d_med be the median of strictly positive item degrees. Set W_ui = clip((d_med/d_i)^0.5, 0.5, 2.0), symmetric W_iu=W_ui, and apply D^-1/2 W D^-1/2. No item-item edges are added. Graph size and support stay identical to S2. The constructor aborts if original CF support differs from positive TRAIN interactions or has non-CF edges, and verifies original normalized graph equivalence.

Foreground run:

```bash
cd /home/bbx/user_lx/GraphLoRA
git fetch origin
git switch exp/G3-popularity-debiased-graph-20261015
git pull --ff-only origin exp/G3-popularity-debiased-graph-20261015
python -u run_G3_popularity_graph.py --gpu 1
```

CPU unit tests run before GPU initialization; full three epochs and best-checkpoint Test execute serially. Training logs in log/G3_POP_DEBIAS_train_<timestamp>.log; graph/provenance in out/G3_graph_<timestamp>/; metrics in out/G3_popularity_graph_<timestamp>.txt and .csv.

G3 tests the hypothesis that suppressing popular items in collaborative message propagation will improve long-tail discrimination and hence OOD CTR AUC/UAUC. This is a hypothesis, not a result. Use S2 Valid AUC 0.732883563 and UAUC 0.687952611 to judge improvement.
