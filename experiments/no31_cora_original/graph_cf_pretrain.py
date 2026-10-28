#!/usr/bin/env python3
"""Train-only graph collaborative encoder for CoRA NIGHT23.

LightGCN is trained from random user/item IDs using positive TRAIN edges
and BPR; it does not initialize its embeddings from MF.  The saved NPZ is
frozen for CoRA training.  This is an offline graph stage, NOT a test run.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans


def train_rows(path, n_users, n_items):
    p = Path(path)
    if "valid" in p.name.lower() or "test" in p.name.lower():
        raise ValueError("Graph edges must be built from TRAIN, never Valid/Test")
    df = pd.read_pickle(p)
    if not {"uid", "iid", "label"}.issubset(df.columns):
        raise ValueError("TRAIN requires uid/iid/label")
    x = df[["uid", "iid", "label"]]
    if x.isnull().values.any():
        raise ValueError("TRAIN has missing data")
    u = x.uid.to_numpy(np.int64)
    i = x.iid.to_numpy(np.int64)
    y = x.label.to_numpy(np.int64)
    if (len(u) == 0 or (u < 0).any() or (u >= n_users).any()
            or (i <= 0).any() or (i >= n_items).any()
            or not np.isin(y, [0, 1]).all()):
        raise ValueError("Invalid TRAIN IDs/labels for graph embedding sizes")
    keys = np.unique(u[y == 1] * n_items + i[y == 1])
    if len(keys) < 2:
        raise ValueError("Insufficient positive interactions")
    return keys // n_items, keys % n_items, keys


def normalized_adj(u, i, n_users, n_items, device):
    n = n_users + n_items
    row = np.concatenate([u, i + n_users])
    col = np.concatenate([i + n_users, u])
    data = np.ones(len(row), dtype=np.float32)
    matrix = sp.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
    matrix.data[:] = 1.0
    deg = np.asarray(matrix.sum(axis=1)).reshape(-1)
    inv = np.zeros_like(deg)
    np.divide(1.0, np.sqrt(deg), out=inv, where=deg > 0)
    normalized = (sp.diags(inv) @ matrix @ sp.diags(inv)).tocoo()
    indices = torch.as_tensor(
        np.vstack([normalized.row, normalized.col]), dtype=torch.long, device=device
    )
    values = torch.as_tensor(normalized.data, dtype=torch.float32, device=device)
    adj = torch.sparse_coo_tensor(indices, values, (n, n), device=device).coalesce()
    return adj, deg


def propagate(table, adj, layers):
    z = table
    states = [z]
    for _ in range(layers):
        z = torch.sparse.mm(adj, z)
        states.append(z)
    return torch.stack(states, dim=0).mean(dim=0)


def negative_samples(users, positive_keys, n_items, rng):
    """Sample negatives not present as TRAIN-positive edges for that user."""
    negatives = rng.integers(1, n_items, size=len(users), dtype=np.int64)
    for _ in range(128):
        bad = np.isin(users * n_items + negatives, positive_keys, assume_unique=False)
        if not bad.any():
            return negatives
        negatives[bad] = rng.integers(1, n_items, size=int(bad.sum()), dtype=np.int64)
    raise RuntimeError("Negative sampling failed; too few non-positive items")


def graph_train(positive_u, positive_i, positive_keys, n_users, n_items, dim,
                layers, epochs, max_pairs, lr, reg, seed, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    adj, degree = normalized_adj(positive_u, positive_i, n_users, n_items, device)
    model = torch.nn.Embedding(n_users + n_items, dim, device=device)
    torch.nn.init.normal_(model.weight, mean=0.0, std=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        which = rng.choice(len(positive_u), size=min(max_pairs, len(positive_u)),
                           replace=False)
        u_np, i_np = positive_u[which], positive_i[which]
        j_np = negative_samples(u_np, positive_keys, n_items, rng)
        u = torch.as_tensor(u_np, dtype=torch.long, device=device)
        i = torch.as_tensor(i_np + n_users, dtype=torch.long, device=device)
        j = torch.as_tensor(j_np + n_users, dtype=torch.long, device=device)
        opt.zero_grad(set_to_none=True)
        z = propagate(model.weight, adj, layers)
        pos = (z[u] * z[i]).sum(dim=-1)
        neg = (z[u] * z[j]).sum(dim=-1)
        bpr = torch.nn.functional.softplus(neg - pos).mean()
        regularizer = (model.weight[u].square().sum(dim=-1)
                       + model.weight[i].square().sum(dim=-1)
                       + model.weight[j].square().sum(dim=-1)).mean()
        loss = bpr + reg * regularizer / (2.0 * dim)
        if not torch.isfinite(loss):
            raise FloatingPointError("Graph BPR loss is nonfinite")
        loss.backward()
        opt.step()
        print(f"[N23-GRAPH-EPOCH] {epoch+1}/{epochs} bpr={float(bpr):.6f} "
              f"loss={float(loss):.6f}", flush=True)
    with torch.no_grad():
        embeddings = propagate(model.weight, adj, layers).detach().float().cpu().numpy()
    if not np.isfinite(embeddings).all():
        raise FloatingPointError("Graph representations contain nonfinite numbers")
    raw_id_table = model.weight.detach().float().cpu().numpy()
    return embeddings[:n_users], embeddings[n_users:], degree, raw_id_table


def make_prototypes(items, positive_i, k, seed):
    """Cluster propagated item embeddings, not pretrained MF embeddings."""
    n_items, d = items.shape
    seen = np.unique(positive_i)
    if len(seen) < k:
        raise ValueError("Fewer TRAIN-positive items than clusters")
    active = items[seen]
    norms = np.maximum(np.linalg.norm(active, axis=1, keepdims=True), 1e-8)
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(active / norms)
    groups = np.full(n_items, -1, dtype=np.int64)
    groups[seen] = labels
    centers = np.stack([items[seen[labels == j]].mean(axis=0) for j in range(k)])
    unseen = np.flatnonzero((groups < 0) & (np.arange(n_items) != 0))
    if len(unseen):
        a = items[unseen] / np.maximum(np.linalg.norm(items[unseen], axis=1, keepdims=True), 1e-8)
        c = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)
        groups[unseen] = (a @ c.T).argmax(axis=1)
    groups[0] = -1
    residual = np.zeros((n_items, d), dtype=np.float32)
    residual[1:] = items[1:] - centers[groups[1:]]
    assert np.allclose(centers[groups[1:]] + residual[1:], items[1:], atol=1e-6)
    return groups, centers.astype(np.float32), residual


def shape_from_mf_checkpoint(path):
    # Read only shapes. MF weights are NEVER used to initialize graph embeddings.
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if "model" in state:
        state = state["model"]
    return (int(state["user_embedding.weight"].shape[0]),
            int(state["item_embedding.weight"].shape[0]))


def sha256(path):
    result = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output .npz path")
    parser.add_argument("--shape-checkpoint", type=Path,
                        help="Optional MF checkpoint for table SHAPES ONLY")
    parser.add_argument("--num-users", type=int, default=0)
    parser.add_argument("--num-items", type=int, default=0)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--clusters", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-bpr-pairs", type=int, default=60000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    a = parser.parse_args()
    if a.seed != 42:
        raise ValueError("NIGHT23 fixes seed=42")
    if a.shape_checkpoint:
        nu, ni = shape_from_mf_checkpoint(a.shape_checkpoint)
        if (a.num_users and a.num_users != nu) or (a.num_items and a.num_items != ni):
            raise ValueError("Manual sizes conflict with shape checkpoint")
        a.num_users, a.num_items = nu, ni
    if a.num_users <= 0 or a.num_items <= 1 or a.dim != 256 or a.clusters != 16:
        raise ValueError("Require exact user/item sizes, dim=256, K=16")
    if a.epochs <= 0 or a.layers < 1 or a.max_bpr_pairs <= 0:
        raise ValueError("Invalid graph pretraining configuration")
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required but unavailable")
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if a.device == "cuda":
        torch.cuda.manual_seed_all(a.seed)
    train = a.train_file.expanduser().resolve()
    out = a.output.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"Refusing overwrite: {out}")
    if not train.is_file():
        raise FileNotFoundError(train)
    users, items, keys = train_rows(train, a.num_users, a.num_items)
    print(f"[N23-GRAPH-INPUT] users={a.num_users} items={a.num_items} "
          f"positive_unique_edges={len(keys)} train_only=True mf_init=False", flush=True)
    zuser, zitem, degree, raw_id_table = graph_train(
        users, items, keys, a.num_users, a.num_items,
        a.dim, a.layers, a.epochs, a.max_bpr_pairs, a.lr, a.reg,
        a.seed, torch.device(a.device),
    )
    zitem[0] = 0
    groups, prototypes, residual = make_prototypes(zitem, items, a.clusters, a.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, graph_id_table=raw_id_table.astype(np.float32),
        user_graph=zuser.astype(np.float32),
        item_graph=zitem.astype(np.float32), item_to_cluster=groups,
        prototypes=prototypes, mf_residual=residual,
        item_residual=residual, item_connected_in_graph=(degree[a.num_users:] > 0))
    manifest = {
        "source": "train-only LightGCN ID embeddings with BPR",
        "train_file": str(train), "train_sha256": sha256(train),
        "positive_unique_edges": int(len(keys)), "num_users": a.num_users,
        "num_items": a.num_items, "dim": a.dim, "layers": a.layers,
        "clusters": a.clusters, "epochs": a.epochs, "seed": a.seed,
        "mf_initialization": False, "joint_training_ready": True, "graph_npz": str(out),
        "community_sizes": np.bincount(groups[1:], minlength=a.clusters).tolist(),
        "train_positive_edge_inclusion": True,
        "note": "Train positives are edges in the graph (standard transductive CF); "
                "conditional user-group counts separately exclude the target pair. "
                "No Valid/Test edges or labels are used."
    }
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(f"[N23-GRAPH-SAVED] {out} manifest={out.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    main()
