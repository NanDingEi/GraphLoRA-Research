#!/usr/bin/env python3
"""Synthetic CPU end-to-end smoke test, never reads user's datasets/checkpoints."""
import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from experiments.cf_prototype_probe import probe


def main():
    rng = np.random.default_rng(42)
    n_users, n_items, dim = 16, 24, 8
    user_embeddings = rng.normal(size=(n_users, dim)).astype(np.float32)
    item_embeddings = rng.normal(size=(n_items, dim)).astype(np.float32)
    item_embeddings[0] = 0.0

    training = []
    for uid in range(1, n_users):
        positive = {((uid * 3 + p * 2) % (n_items - 1)) + 1 for p in range(8)}
        for iid in range(1, n_items):
            training.append((uid, iid, int(iid in positive)))
    valid = []
    for uid in range(1, n_users):
        scores = user_embeddings[uid] @ item_embeddings[1:].T
        median = float(np.median(scores))
        for iid in range(1, n_items):
            valid.append((uid, iid, int(scores[iid - 1] > median)))

    with tempfile.TemporaryDirectory(prefix="cf_proto_probe_") as tmp:
        tmp = Path(tmp)
        data = tmp / "dataset"
        data.mkdir()
        pd.DataFrame(training, columns=["uid", "iid", "label"]).to_pickle(
            data / "train_ood2.pkl"
        )
        pd.DataFrame(valid, columns=["uid", "iid", "label"]).to_pickle(
            data / "valid_ood2.pkl"
        )
        ckpt = tmp / "mf.pth"
        torch.save({
            "user_embedding.weight": torch.tensor(user_embeddings),
            "item_embedding.weight": torch.tensor(item_embeddings),
        }, ckpt)

        out = tmp / "out"
        args = argparse.Namespace(
            data_dir=data, train_file="train_ood2.pkl",
            valid_file="valid_ood2.pkl", mf_checkpoint=ckpt,
            k_values=[2, 3], seed=42, output_dir=out,
        )
        probe.run(args)
        summary = list(csv.DictReader((out / "summary.csv").open()))
        assert len(summary) == 2
        for k in (2, 3):
            payload = json.loads((out / f"metrics_k{k}.json").read_text())
            p3 = payload["p3_decomposition"]
            p2 = payload["p2_within_group"]
            assert p3["max_absolute_score_error"] < 1e-8
            assert payload["metrics"]["coarse_plus_fine"]["auc"] == (
                payload["metrics"]["original_mf"]["auc"]
            )
            assert p2["max_auc_difference"] is None or p2["max_auc_difference"] < 1e-10
            assert "mean_item_popularity_auc" in p2
            mat = np.load(out / f"prototypes_k{k}.npz")
            clusters, proto, delta = (
                mat["item_to_cluster"], mat["prototypes"], mat["mf_residual"],
            )
            assert clusters[0] == -1 and np.all(clusters[1:] >= 0)
            assert np.allclose(delta[0], 0)
            assert np.allclose(
                proto[clusters[1:]] + delta[1:], item_embeddings[1:], atol=1e-5
            )
        print("[CF-PROTO-CPU-PASS] synthetic MF checkpoint, train-only graph, "
              "K=2/3 spectral prototypes, exact score reconstruction, "
              "within-community consistency, CSV and NPZ exports")


if __name__ == "__main__":
    main()
