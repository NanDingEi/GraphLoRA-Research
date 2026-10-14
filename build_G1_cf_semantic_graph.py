#!/usr/bin/env python3
"""G1 graph construction: train-only collaborative support + movie metadata.

Inputs are ONLY movie_graph_train.npz, train_ood2.pkl and optional movies.dat.
Validation/test files are not read. Internal item IDs come from the train PKL,
never from MovieLens's original movies.dat ID numbers.

Build raw W = support(original TRAIN graph) + alpha * A_sem,
then D^-1/2 W D^-1/2. The original normalized adjacency is checked against
normalizing its own unweighted support; abort if it does not match instead of
silently changing the S2 control graph's edge weighting.
"""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


def graph_normalize(raw):
    raw = raw.tocsr().astype(np.float64)
    raw.sum_duplicates()
    raw.eliminate_zeros()
    deg = np.asarray(raw.sum(axis=1)).reshape(-1)
    inv = np.zeros_like(deg, dtype=np.float64)
    nonzero = deg > 0
    inv[nonzero] = deg[nonzero] ** -0.5
    result = sp.diags(inv) @ raw @ sp.diags(inv)
    return result.tocsr().astype(np.float32)


def normalized_movie_title(value):
    s = str(value).strip().strip('"').lower()
    s = re.sub(r"\s+", " ", s)
    return s


def without_year(value):
    return re.sub(r"\s*\(\d{4}\)\s*$", "", normalized_movie_title(value)).strip()


def sha(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def titles_from_train(df, num_items):
    # Training target ID is the authoritative index; history IDs and titles
    # from the training prompt provide extra metadata coverage if available.
    counts = defaultdict(Counter)
    positives = defaultdict(set)
    for row in df.itertuples(index=False):
        iid = int(row.iid)
        if 0 <= iid < num_items and pd.notna(row.title):
            counts[iid][normalized_movie_title(row.title)] += 1
        if int(row.label) == 1 and 0 <= iid < num_items:
            positives[int(row.uid)].add(iid)
        ids = getattr(row, "his", None)
        titles = getattr(row, "his_title", None)
        if isinstance(ids, (list, tuple, np.ndarray)) and isinstance(
            titles, (list, tuple, np.ndarray)
        ):
            for hist_id, hist_title in zip(ids, titles):
                if hist_id is None or hist_title is None:
                    continue
                try:
                    h = int(hist_id)
                except (ValueError, TypeError):
                    continue
                # 0 denotes history padding in the OOD loader.
                if 0 < h < num_items and isinstance(hist_title, str):
                    counts[h][normalized_movie_title(hist_title)] += 1

    titles = {
        i: sorted(v.items(), key=lambda pair: (-pair[1], pair[0]))[0][0]
        for i, v in counts.items() if v
    }
    return titles, positives


def genres_from_movies_dat(path):
    if path is None or not Path(path).is_file():
        return {}, False
    mapping = {}
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("::", 2)
            if len(parts) != 3:
                continue
            title, genres = normalized_movie_title(parts[1]), frozenset(
                x.strip().lower() for x in parts[2].split("|") if x.strip()
            )
            if title in mapping and mapping[title] != genres:
                # Ambiguous title: do not map a wrong original MovieID.
                mapping[title] = frozenset()
            else:
                mapping[title] = genres
    return mapping, True


def supported_pairs(positives):
    shared_users = Counter()
    for items in positives.values():
        for pair in combinations(sorted(items), 2):
            shared_users[pair] += 1
    return shared_users


def select_semantic_edges(titles, positives, genres_by_title, num_users,
                          num_items, alpha=0.25, min_shared_users=2,
                          min_semantic=0.30, max_item_degree=5):
    if not (0 < alpha <= 1 and 0 < min_semantic <= 1):
        raise ValueError("alpha and minimum similarity must be within (0,1]")
    if min_shared_users < 1 or max_item_degree < 1:
        raise ValueError("min_shared_users and max_item_degree must be positive")

    available = sorted(titles.keys())
    if len(available) < 2:
        raise ValueError("Not enough train-known items to build a content graph")
    # Title character ngrams are lexical, not pretrained language embeddings.
    # Genres (when available) provide stronger metadata-based similarity.
    docs = [without_year(titles[i]) for i in available]
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5),
                          lowercase=True, dtype=np.float32)
    X = vec.fit_transform(docs)
    title_sim = (X @ X.T).toarray()  # 3256x3256 ~ 42 MB
    col = {iid: j for j, iid in enumerate(available)}
    genres = {iid: genres_by_title.get(titles[iid], frozenset()) for iid in available}
    genre_coverage = sum(bool(v) for v in genres.values())
    common = supported_pairs(positives)

    candidates = []
    n_after_support = 0
    for (i, j), n_shared in common.items():
        if n_shared < min_shared_users or i not in col or j not in col:
            continue
        n_after_support += 1
        gi, gj = genres[i], genres[j]
        title_score = float(title_sim[col[i], col[j]])
        if gi and gj:
            sem = 0.75 * len(gi & gj) / len(gi | gj) + 0.25 * title_score
        else:
            sem = title_score
        if sem < min_semantic:
            continue
        # Count support influences weight but not the semantic criterion.
        weight = sem * min(1., np.log1p(n_shared) / np.log1p(5.))
        candidates.append((weight, n_shared, i, j, sem))

    # Greedy symmetric degree cap, deterministic across repeated runs.
    candidates.sort(key=lambda t: (-t[0], -t[1], t[2], t[3]))
    degree = np.zeros(num_items, dtype=np.int32)
    edges = []
    for weight, n_shared, i, j, sem in candidates:
        if degree[i] >= max_item_degree or degree[j] >= max_item_degree:
            continue
        edges.append((i, j, alpha * weight))
        degree[i] += 1
        degree[j] += 1

    if not edges:
        raise RuntimeError(
            "G1 produced zero edges with collaborative+semantic agreement. "
            "Inspect train metadata/genre coverage; refuse a no-op GPU run."
        )
    rr, cc, vv = [], [], []
    for i, j, w in edges:
        a, b = num_users + i, num_users + j
        rr.extend((a, b))
        cc.extend((b, a))
        vv.extend((w, w))
    add = sp.csr_matrix((np.array(vv, dtype=np.float64), (rr, cc)),
                        shape=(num_users + num_items, num_users + num_items))
    report = {
        "train_known_titles": len(titles), "genres_mapped": genre_coverage,
        "positive_train_users": len(positives),
        "cooccurrence_pairs": len(common),
        "pairs_meeting_collaborative_support": n_after_support,
        "pairs_meeting_both_criteria": len(candidates),
        "undirected_new_item_edges": len(edges),
        "added_directed_entries": int(add.nnz),
        "new_item_max_degree": int(degree.max()),
        "new_item_mean_degree": float(degree.mean()),
        "alpha": float(alpha),
        "min_shared_users": int(min_shared_users),
        "min_semantic": float(min_semantic),
        "max_item_degree": int(max_item_degree),
    }
    return add, report


def build_graph(data_dir, out_path, num_users, num_items,
                alpha=0.25, min_shared_users=2, min_semantic=0.30,
                max_item_degree=5):
    data_dir = Path(data_dir).resolve()
    out_path = Path(out_path).resolve()
    base_path = data_dir / "movie_graph_train.npz"
    train_path = data_dir / "train_ood2.pkl"
    movies_path = data_dir / "movies.dat"
    for p in (base_path, train_path):
        if not p.is_file():
            raise FileNotFoundError(str(p))
    baseline = sp.load_npz(base_path).tocsr().astype(np.float32)
    n = num_users + num_items
    if baseline.shape != (n, n):
        raise ValueError("S2 graph size {} != configured ({} users + {} items)".format(
            baseline.shape, num_users, num_items))
    if baseline.nnz == 0:
        raise ValueError("S2 train graph empty")
    sym = baseline - baseline.T
    sym_err = float(np.max(np.abs(sym.data))) if sym.nnz else 0.
    if sym_err > 1.e-5:
        raise ValueError("S2 input adjacency not symmetric")
    support = baseline.copy().astype(np.float64)
    support.data[:] = 1.
    support.sum_duplicates()
    support.eliminate_zeros()
    reconstructed = graph_normalize(support)
    diff = (baseline - reconstructed).tocsr()
    base_err = float(np.max(np.abs(diff.data))) if diff.nnz else 0.
    if base_err > 3.e-4:
        raise ValueError(
            "S2 adjacency is NOT D^-1/2 binary-support D^-1/2 "
            "(max error {:.7f}). G1 would change the CF baseline too; "
            "aborting before GPU training.".format(base_err)
        )

    df = pd.read_pickle(train_path)
    needed = {"uid", "iid", "title", "label"}
    if not needed.issubset(df.columns):
        raise ValueError("Expected TRAIN columns {}; got {}".format(
            needed, set(df.columns)))
    if (df.uid.min() < 0 or df.uid.max() >= num_users or
            df.iid.min() < 0 or df.iid.max() >= num_items):
        raise ValueError("Training ID range exceeds configured CF node space")
    # Confirm original S2 bipartite graph itself contains ONLY positive
    # TRAIN interactions; no hidden valid/test edges or cached mixed graph.
    expected_edges = set(
        zip(df.loc[df.label > 0, "uid"].astype(int),
            df.loc[df.label > 0, "iid"].astype(int))
    )
    cf = support[:num_users, num_users:].tocoo()
    actual_edges = set(zip(cf.row.tolist(), cf.col.tolist()))
    if (support[:num_users, :num_users].nnz != 0 or
            support[num_users:, num_users:].nnz != 0 or
            actual_edges != expected_edges):
        raise ValueError(
            "S2 graph must contain only positive TRAIN user-item edges. "
            "CF graph edges={}, train-positive edges={}, differences "
            "missing={} extra={}; abort to prevent data leakage".format(
                len(actual_edges), len(expected_edges),
                len(expected_edges - actual_edges),
                len(actual_edges - expected_edges),
            )
        )
    titles, positives = titles_from_train(df, num_items)
    genres, has_movies = genres_from_movies_dat(
        movies_path if movies_path.is_file() else None
    )
    sem_raw, report = select_semantic_edges(
        titles, positives, genres, num_users, num_items,
        alpha, min_shared_users, min_semantic, max_item_degree,
    )
    fused = graph_normalize(support + sem_raw)
    fused.eliminate_zeros()
    if not np.isfinite(fused.data).all() or np.min(fused.data) < 0:
        raise ValueError("G1 fused graph has invalid weights")
    if fused.shape != baseline.shape:
        raise ValueError("G1 changed graph size")
    if (fused - fused.T).nnz and np.max(
        np.abs((fused - fused.T).data)
    ) > 1e-5:
        raise ValueError("G1 fused graph not symmetric")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out_path, fused)
    report.update({
        "baseline_nnz": int(baseline.nnz),
        "verified_CF_edges_match_positive_train": True,
        "G1_nnz": int(fused.nnz),
        "baseline_max_reconstruction_err": base_err,
        "uses_movies_dat_genres": has_movies,
        "train_rows": int(len(df)),
        "graph_path": str(out_path),
        "input_sha256": {
            "train_ood2.pkl": sha(train_path),
            "movie_graph_train.npz": sha(base_path),
            "movies.dat": sha(movies_path) if has_movies else None,
        },
        "graph_sha256": sha(out_path),
        "construction": "binary original CF support + weighted title/genre "
                        "agreement restricted to train positive co-occurrences; "
                        "recompute symmetric normalization",
        "data_leakage_policy": "no validation/test interactions or labels used",
    })
    out_path.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="dataset/ml-1m")
    p.add_argument("--out", required=True)
    p.add_argument("--num-users", type=int, default=839)
    p.add_argument("--num-items", type=int, default=3256)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--min-shared-users", type=int, default=2)
    p.add_argument("--min-semantic", type=float, default=0.30)
    p.add_argument("--max-item-degree", type=int, default=5)
    args = p.parse_args()
    print(json.dumps(build_graph(
        args.data_dir, args.out, args.num_users, args.num_items,
        args.alpha, args.min_shared_users, args.min_semantic,
        args.max_item_degree,
    ), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
