#!/usr/bin/env python3
"""CPU-only collaborative prototype/residual probe for existing pretrained MF.

NO LLM, NO LoRA training, NO test split. Spectral item communities are computed
from positive TRAIN interactions only. Valid is used strictly for reporting.
P1: prototype coarse-signal quality vs random-size-matched groups.
P2: within-(user,community) ranking, explicitly noting the exact MF tautology.
P3: exact decomposition, coverage, variance and residual-energy diagnostics.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
import torch

DEFAULT_DATA = Path("/home/bbx/user_lx/GraphLoRA/dataset/ml-1m")
DEFAULT_CKPT = Path("/home/bbx/user_lx/GraphLoRA/checkpoints/"
                    "0912_ml1m_oodv2_best_model_d256lr-0.001wd0.0001.pth")


def load_mf(path):
    """The original baseline_train_mf_ood.py saves model.state_dict()."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # compatibility with older PyTorch versions
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise ValueError("Expected MF state_dict or {'model': state_dict}")
    keys = ("user_embedding.weight", "item_embedding.weight")
    if not all(k in state for k in keys):
        raise KeyError("MF checkpoint lacks expected embedding weights: "
                       + ", ".join(keys) + "; found " + repr(list(state)[:8]))
    users = state[keys[0]].detach().float().cpu().numpy().astype(np.float64)
    items = state[keys[1]].detach().float().cpu().numpy().astype(np.float64)
    if users.ndim != 2 or items.ndim != 2 or users.shape[1] != items.shape[1]:
        raise ValueError("Invalid MF user/item embedding shape")
    if not np.isfinite(users).all() or not np.isfinite(items).all():
        raise ValueError("MF embedding contains NaN/Inf")
    return users, items


def load_rows(path, num_users, num_items, name):
    df = pd.read_pickle(path)
    if not {"uid", "iid", "label"}.issubset(df.columns):
        raise ValueError(name + ": expected columns uid, iid, label")
    f = df.loc[:, ["uid", "iid", "label"]].copy()
    if f.isnull().values.any():
        raise ValueError(name + ": null user/item/label")
    u = f.uid.to_numpy(dtype=np.int64)
    i = f.iid.to_numpy(dtype=np.int64)
    y = f.label.to_numpy(dtype=np.float64)
    if (np.any(u < 0) or np.any(i <= 0) or
        np.any(u >= num_users) or np.any(i >= num_items)):
        raise ValueError(name + ": IDs exceed MF checkpoint or use padding item 0")
    if not np.isfinite(y).all() or not np.isin(y, (0, 1)).all():
        raise ValueError(name + ": expected binary labels 0/1")
    return u, i, y.astype(np.int8)


def train_positive_graph(train, n_users, n_items):
    """Construct binary positive user-item graph from TRAIN ONLY."""
    u, i, y = train
    pos = y == 1
    if not pos.any():
        raise RuntimeError("No positive training interactions")
    # Binary CSR automatically collapses repeated positive edges.
    R = sp.coo_matrix(
        (np.ones(int(pos.sum()), dtype=np.float64), (u[pos], i[pos])),
        shape=(n_users, n_items),
    ).tocsr()
    R.data[:] = 1.0
    R.eliminate_zeros()
    counts = np.asarray(R.getnnz(axis=1)).ravel()
    active = np.asarray(R.getnnz(axis=0)).ravel() > 0
    # Weight each user equally rather than weighting highly active users heavily.
    inv_count = np.divide(
        1.0, counts, out=np.zeros(n_users, dtype=np.float64), where=counts > 0,
    )
    cooc = (R.T @ sp.diags(inv_count) @ R).tocsr()
    cooc.setdiag(0)
    cooc.eliminate_zeros()
    cooc = (cooc + cooc.T) * 0.5
    return cooc.tocsr(), active, int(R.nnz)


def _nearest_groups(embeddings, centers):
    # Deterministic cosine fallback for disconnected / unseen-in-train items.
    lhs = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    rhs = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    return np.argmax(lhs @ rhs.T, axis=1).astype(np.int32)


def communities_from_cooc(cooc, item_emb, k, seed):
    """Spectral clusters on positive cooccurrence; MF fallback for isolates.

    Item index 0 is padding and is never assigned a cluster.
    """
    n_items = item_emb.shape[0]
    degrees = np.asarray(cooc.sum(axis=1)).reshape(-1)
    connected = np.flatnonzero((degrees > 0) & (np.arange(n_items) != 0))
    if connected.size <= k + 1:
        raise RuntimeError("Only %d cooccurrence-connected items for k=%d"
                           % (connected.size, k))
    A = cooc[connected, :][:, connected]
    deg = np.asarray(A.sum(axis=1)).reshape(-1)
    inv_sqrt = np.divide(
        1.0, np.sqrt(deg), out=np.zeros_like(deg), where=deg > 0,
    )
    norm_A = sp.diags(inv_sqrt) @ A @ sp.diags(inv_sqrt)
    # Deterministic ARPACK start, fixed KMeans seed.
    v0 = np.random.default_rng(seed).standard_normal(connected.size)
    _, vectors = eigsh(norm_A, k=k, which="LA", v0=v0, tol=1e-4)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    groups_seen = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(vectors)
    groups = np.full(n_items, -1, dtype=np.int32)
    groups[connected] = groups_seen
    tmp_centers = np.stack(
        [item_emb[connected[groups_seen == j]].mean(axis=0) for j in range(k)],
        axis=0,
    )
    missing = np.flatnonzero((groups < 0) & (np.arange(n_items) != 0))
    groups[missing] = _nearest_groups(item_emb[missing], tmp_centers)
    centers = np.stack(
        [item_emb[np.flatnonzero(groups == j)].mean(axis=0) for j in range(k)],
        axis=0,
    )
    assert groups[0] == -1
    assert (groups[1:] >= 0).all()
    return groups, centers, int(connected.size), int(missing.size)


def centers_from_assignment(item_emb, groups, k):
    return np.stack(
        [item_emb[groups == j].mean(axis=0) for j in range(k)],
        axis=0,
    )


def uauc(user, labels, scores):
    out = []
    for uid in np.unique(user):
        ix = user == uid
        if np.unique(labels[ix]).size == 2:
            out.append(float(roc_auc_score(labels[ix], scores[ix])))
    return (float(np.mean(out)) if out else None, len(out))


def global_metrics(user, label, score):
    return {
        "auc": float(roc_auc_score(label, score)) if np.unique(label).size == 2 else None,
        "uauc": uauc(user, label, score)[0],
        "uauc_eligible_users": uauc(user, label, score)[1],
    }


def within_group_pairs(user, group_id, label, original, residual):
    """Within (user, group) the coarse term is constant: AUC identity is expected."""
    # Build a flat key for efficient unique grouping, including no padding cluster.
    keys = np.column_stack((user, group_id))
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    auc_orig, auc_resid, total_rows, pair_count = [], [], 0, 0
    for g in range(unique.shape[0]):
        ix = np.flatnonzero(inverse == g)
        yy = label[ix]
        if np.unique(yy).size != 2:
            continue
        positives = int((yy == 1).sum())
        negatives = int((yy == 0).sum())
        pair_count += positives * negatives
        total_rows += int(ix.size)
        auc_orig.append(float(roc_auc_score(yy, original[ix])))
        auc_resid.append(float(roc_auc_score(yy, residual[ix])))
    return {
        "eligible_user_cluster_groups": len(auc_orig),
        "eligible_rows": total_rows,
        "eligible_rows_fraction": total_rows / max(len(label), 1),
        "comparable_pos_neg_pairs": pair_count,
        "mean_original_mf_auc": float(np.mean(auc_orig)) if auc_orig else None,
        "mean_fine_residual_auc": float(np.mean(auc_resid)) if auc_resid else None,
        "max_auc_difference": float(np.max(np.abs(
            np.array(auc_orig) - np.array(auc_resid)
        ))) if auc_orig else None,
        "note": "Equality is algebraically guaranteed inside each user-cluster group; "
                "do not interpret matching AUC as independent fine-signal evidence.",
    }


def evaluate(users, items, train_active, groups, centers, valid):
    u, i, labels = valid
    base = np.einsum("bd,bd->b", users[u], items[i])
    coarse = np.einsum("bd,bd->b", users[u], centers[groups[i]])
    fine = np.einsum("bd,bd->b", users[u], items[i] - centers[groups[i]])
    error = np.abs(base - (coarse + fine))
    maxerr = float(error.max()) if error.size else 0.0
    energy = np.linalg.norm(items[1:] - centers[groups[1:]], axis=1)
    original_energy = np.linalg.norm(items[1:], axis=1)
    result = {
        "metrics": {
            "original_mf": global_metrics(u, labels, base),
            "prototype_coarse": global_metrics(u, labels, coarse),
            "within_cluster_fine": global_metrics(u, labels, fine),
            "coarse_plus_fine": global_metrics(u, labels, coarse + fine),
        },
        "p2_within_group": within_group_pairs(u, groups[i], labels, base, fine),
        "p3_decomposition": {
            "max_absolute_score_error": maxerr,
            "mean_absolute_score_error": float(error.mean()),
            "relative_embedding_residual_energy":
                float(np.linalg.norm(energy) / max(np.linalg.norm(original_energy), 1e-12)),
            "score_variance_coarse": float(np.var(coarse)),
            "score_variance_fine": float(np.var(fine)),
            "score_correlation": float(np.corrcoef(coarse, fine)[0, 1])
                if np.std(coarse) > 1e-12 and np.std(fine) > 1e-12 else None,
        },
        "coverage": {
            "valid_count": len(i),
            "train_positive_item_coverage_on_valid":
                float(np.mean(train_active[i])),
            "distinct_valid_users": int(np.unique(u).size),
            "distinct_valid_items": int(np.unique(i).size),
            "cluster_sizes": np.bincount(groups[1:], minlength=len(centers)).tolist(),
            "valid_cluster_sizes": np.bincount(groups[i], minlength=len(centers)).tolist(),
        },
    }
    if maxerr > 1e-8:
        raise RuntimeError("P3 decomposition violated: max error = %g" % maxerr)
    combined_auc = result["metrics"]["coarse_plus_fine"]["auc"]
    orig_auc = result["metrics"]["original_mf"]["auc"]
    if abs(combined_auc - orig_auc) > 1e-12:
        raise RuntimeError("P3 MF AUC differs after exact reconstruction")
    return result


def run(args):
    train_path = Path(args.data_dir) / args.train_file
    valid_path = Path(args.data_dir) / args.valid_file
    if "test" in str(valid_path).lower():
        raise ValueError("Do not use Test split for the prototype probe")
    for path in (train_path, valid_path, args.mf_checkpoint):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    user_emb, item_emb = load_mf(args.mf_checkpoint)
    train = load_rows(train_path, len(user_emb), len(item_emb), "train")
    valid = load_rows(valid_path, len(user_emb), len(item_emb), "valid")
    cooc, train_active, unique_pos_edges = train_positive_graph(
        train, len(user_emb), len(item_emb),
    )
    outdir = Path(args.output_dir) if args.output_dir else (
        Path("logs") / ("CF_PROTOTYPE_PROBE_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    outdir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "data_dir": str(args.data_dir), "train_file": str(train_path),
        "valid_file": str(valid_path), "mf_checkpoint": str(args.mf_checkpoint),
        "seed": args.seed, "k_values": args.k_values,
        "graph": "train-only binary-positive cooccurrence / user positive degree",
        "algorithm": "normalized spectral graph embedding + KMeans",
        "unique_train_positive_edges": unique_pos_edges,
        "num_train_rows": len(train[0]), "num_valid_rows": len(valid[0]),
        "num_users": len(user_emb), "num_items": len(item_emb),
        "label_distribution_valid": {str(v): int((valid[2] == v).sum()) for v in (0, 1)},
        "note": "No LLM/CoRA training; validation results are diagnostics, not Test.",
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summary_rows = []
    for k in args.k_values:
        print(f"[PROTO-START] K={k}, train positive edges={unique_pos_edges}", flush=True)
        groups, centers, nconnected, nfallback = communities_from_cooc(
            cooc, item_emb, k, args.seed,
        )
        result = evaluate(user_emb, item_emb, train_active, groups, centers, valid)
        # Null control: random item community IDs with identical group sizes.
        rng = np.random.default_rng(args.seed + 1000 + k)
        shuffled = groups.copy()
        shuffled[1:] = rng.permutation(shuffled[1:])
        null_centers = centers_from_assignment(item_emb, shuffled, k)
        random_result = evaluate(
            user_emb, item_emb, train_active, shuffled, null_centers, valid,
        )
        result["p1_null_control"] = {
            "random_size_matched_coarse": random_result["metrics"]["prototype_coarse"],
            "random_size_matched_residual_energy":
                random_result["p3_decomposition"]["relative_embedding_residual_energy"],
        }
        result["k"] = k
        result["connected_items"] = nconnected
        result["mf_assigned_disconnected_items"] = nfallback
        (outdir / f"metrics_k{k}.json").write_text(json.dumps(result, indent=2))
        np.savez_compressed(
            outdir / f"prototypes_k{k}.npz",
            item_to_cluster=groups, prototypes=centers.astype(np.float32),
            mf_residual=(item_emb -
                         np.where(np.arange(len(groups))[:, None] == 0, 0,
                                  centers[np.maximum(groups, 0)])).astype(np.float32),
            item_connected_in_graph=(np.asarray(cooc.sum(axis=1)).ravel() > 0),
        )
        m = result["metrics"]
        p2 = result["p2_within_group"]
        p3 = result["p3_decomposition"]
        summary_rows.append({
            "K": k,
            "MF_AUC": m["original_mf"]["auc"],
            "MF_UAUC": m["original_mf"]["uauc"],
            "coarse_AUC": m["prototype_coarse"]["auc"],
            "coarse_UAUC": m["prototype_coarse"]["uauc"],
            "fine_global_AUC": m["within_cluster_fine"]["auc"],
            "fine_global_UAUC": m["within_cluster_fine"]["uauc"],
            "random_group_coarse_AUC":
                result["p1_null_control"]["random_size_matched_coarse"]["auc"],
            "random_group_coarse_UAUC":
                result["p1_null_control"]["random_size_matched_coarse"]["uauc"],
            "residual_energy_ratio": p3["relative_embedding_residual_energy"],
            "random_residual_energy_ratio":
                result["p1_null_control"]["random_size_matched_residual_energy"],
            "within_group_AUC": p2["mean_fine_residual_auc"],
            "within_group_rows": p2["eligible_rows"],
            "within_group_pairs": p2["comparable_pos_neg_pairs"],
            "decomposition_maxerr": p3["max_absolute_score_error"],
            "disconnected_items_assigned_by_MF": nfallback,
        })
        print("[PROTO-RESULT] K=%d MF=%.6f coarse=%.6f null_coarse=%.6f "
              "within_pairs=%d residual_energy=%.4f" % (
                  k, m["original_mf"]["auc"], m["prototype_coarse"]["auc"],
                  result["p1_null_control"]["random_size_matched_coarse"]["auc"],
                  p2["comparable_pos_neg_pairs"],
                  p3["relative_embedding_residual_energy"],
              ), flush=True)
    with (outdir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print("[PROTO-COMPLETE] results in", outdir, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--train-file", default="train_ood2.pkl")
    parser.add_argument("--valid-file", default="valid_ood2.pkl")
    parser.add_argument("--mf-checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--k-values", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.k_values or min(args.k_values) < 2:
        parser.error("K must be at least 2")
    if len(set(args.k_values)) != len(args.k_values):
        parser.error("K values must be unique")
    run(args)


if __name__ == "__main__":
    main()
