#!/usr/bin/env python3
"""G3: train-only popularity-debiased bipartite CF adjacency.

No title/genre edges. Only reweight existing S2 positive TRAIN user-item
edges inversely to positive item frequency, then symmetric-normalize once.
This is an independent S2 binary-CTR experiment, not a modification of G1.
"""
from pathlib import Path
import json
from hashlib import sha256
import numpy as np
import pandas as pd
import scipy.sparse as sp

from build_G1_cf_semantic_graph import graph_normalize


def digest(path):
    h = sha256()
    with open(path, "rb") as f:
        for data in iter(lambda: f.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()


def build_graph(data_dir, out_path, num_users, num_items, eta=0.5,
                min_weight=0.5, max_weight=2.0):
    if not 0 <= eta <= 1 or not 0 < min_weight <= max_weight:
        raise ValueError("G3 weighting parameters out of range")
    data_dir = Path(data_dir).resolve()
    out_path = Path(out_path).resolve()
    train_path = data_dir / "train_ood2.pkl"
    base_path = data_dir / "movie_graph_train.npz"
    df = pd.read_pickle(train_path)
    n = num_users + num_items
    baseline = sp.load_npz(base_path).tocsr().astype(np.float32)
    if baseline.shape != (n, n):
        raise ValueError("G3 graph shape mismatch")
    sym = baseline - baseline.T
    if sym.nnz and np.max(np.abs(sym.data)) > 1e-5:
        raise ValueError("G3 baseline not symmetric")
    support = baseline.copy().astype(np.float64)
    support.data[:] = 1.
    support.sum_duplicates()
    support.eliminate_zeros()
    delta = (baseline - graph_normalize(support)).tocsr()
    error = float(np.max(np.abs(delta.data))) if delta.nnz else 0.
    if error > 3e-4:
        raise ValueError("G3 original S2 graph cannot be reconstructed from binary support")
    positives = df[df.label > 0][["uid", "iid"]].drop_duplicates()
    if positives.empty:
        raise ValueError("G3 training positives missing")
    if (positives.uid.min() < 0 or positives.uid.max() >= num_users
            or positives.iid.min() < 0 or positives.iid.max() >= num_items):
        raise ValueError("G3 invalid internal IDs")
    expected = set(zip(positives.uid.astype(int), positives.iid.astype(int)))
    bip = support[:num_users, num_users:].tocoo()
    actual = set(zip(bip.row.tolist(), bip.col.tolist()))
    if (actual != expected or support[:num_users, :num_users].nnz or
            support[num_users:, num_users:].nnz):
        raise ValueError("G3 original graph has edges not matching positive TRAIN labels")
    degree = np.bincount(positives.iid.to_numpy(dtype=np.int64), minlength=num_items)
    common_items = degree[degree > 0]
    ref_degree = float(np.median(common_items))
    raw = support.tocoo()
    mask = (raw.row < num_users) & (raw.col >= num_users)
    um = raw.row[mask]
    im = raw.col[mask] - num_users
    if len(um) != len(expected):
        raise ValueError("G3 unexpected edge multiplicity")
    weights = np.clip((ref_degree / degree[im]) ** eta, min_weight, max_weight)
    rows = np.concatenate((um, im + num_users))
    cols = np.concatenate((im + num_users, um))
    vals = np.concatenate((weights, weights))
    weighted = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
    fused = graph_normalize(weighted)
    fused.eliminate_zeros()
    if fused.nnz != baseline.nnz:
        raise ValueError("G3 added or deleted edges unexpectedly")
    if not np.isfinite(fused.data).all():
        raise ValueError("G3 produced nonfinite graph")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out_path, fused)
    report = {
        "train_only": True,
        "method": "positive_item_frequency_debias",
        "eta": float(eta), "min_weight": float(min_weight),
        "max_weight": float(max_weight),
        "reference_median_item_degree": ref_degree,
        "train_positive_edges": len(expected), "edge_count": int(fused.nnz),
        "min_actual_weight": float(weights.min()),
        "max_actual_weight": float(weights.max()),
        "original_support_normalization_error": error,
        "train_sha256": digest(train_path),
        "original_graph_sha256": digest(base_path),
        "result_graph_sha256": digest(out_path),
        "graph_path": str(out_path),
    }
    out_path.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
