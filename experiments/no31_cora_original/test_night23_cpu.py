#!/usr/bin/env python3
"""CPU preflight for N23 graph features and graph pair.
No private data, Vicuna weights, or GPU required.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.no31_cora_original.graph_cf_pretrain import (
    train_rows, normalized_adj, propagate, negative_samples, make_prototypes,
)
from experiments.no31_cora_original.night23_graph_features import (
    install_graph_history, graph_pair, make_graph_features,
)

class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cora_fixed_user", torch.ones(3, 256) * 0.1)
        self.register_buffer("cora_fixed_item", torch.ones(7, 256) * 0.1)

def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        frame = pd.DataFrame(dict(uid=[0, 0, 0, 1, 1, 2, 2],
                                  iid=[1, 2, 3, 2, 4, 5, 6],
                                  label=[1, 1, 0, 1, 0, 1, 0]))
        train = root / "train.pkl"
        frame.to_pickle(train)
        u, i, keys = train_rows(train, 3, 7)
        assert len(keys) == 4
        neg = negative_samples(np.asarray([0, 1, 2]), keys, 7,
                               np.random.default_rng(42))
        assert not np.isin(np.asarray([0, 1, 2]) * 7 + neg, keys).any()
        adj, _ = normalized_adj(u, i, 3, 7, torch.device("cpu"))
        assert adj.shape == (10, 10)
        z = propagate(torch.randn(10, 256), adj, 2)
        assert z.shape == (10, 256) and torch.isfinite(z).all()
        user = np.random.default_rng(42).normal(size=(3, 256)).astype(np.float32)
        item = np.random.default_rng(43).normal(size=(7, 256)).astype(np.float32)
        item[0] = 0
        groups, proto, residual = make_prototypes(item, i, 2, 42)
        # Feature wrapper requires K16, production graph; build a synthetic
        # 16-community table with nonempty clusters to verify shapes.
        # Large enough for 16 active communities, plus padding.
        rng = np.random.default_rng(10)
        item = rng.normal(size=(18, 256)).astype(np.float32)
        item[0] = 0
        groups = np.full(18, -1, dtype=np.int64)
        groups[1:] = np.arange(17) % 16
        proto = np.stack([item[1:][groups[1:] == k].mean(axis=0)
                          for k in range(16)]).astype(np.float32)
        residual = np.zeros_like(item)
        residual[1:] = item[1:] - proto[groups[1:]]
        graph_path = root / "graph.npz"
        np.savez(graph_path, user_graph=user, item_graph=item,
                 item_to_cluster=groups, prototypes=proto, item_residual=residual)
        model = Dummy()
        model.register_buffer("cora_fixed_item_replacement", torch.empty(0))
        model.cora_fixed_item = torch.ones(18, 256) * 0.1
        install_graph_history(model, graph_path, train)
        ids_u = torch.tensor([0, 1, 2])
        ids_i = torch.tensor([1, 2, 5])
        pair = graph_pair(model, ids_u, ids_i)
        coarse, fine = make_graph_features(model, ids_u, ids_i)
        assert pair.shape == (3, 512)
        assert coarse.shape == (3, 514)
        assert fine.shape == (3, 513)
        assert torch.isfinite(coarse).all() and torch.isfinite(fine).all()
        # Positive item 5 is the only positive for user 2; it must be
        # removed from user/community history when evaluated as target.
        assert torch.count_nonzero(fine[2, 256:512]) == 0
    print("[N23-CPU-PASS] graph adjacency/propagation, BPR negatives, "
          "prototype decomposition, graph artifact, features, leave-target-out", flush=True)

if __name__ == "__main__":
    main()
