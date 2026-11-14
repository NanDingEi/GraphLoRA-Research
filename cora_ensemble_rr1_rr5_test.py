#!/usr/bin/env python3
"""Parameter-free logit ensembles of RR1 and RR5 TEST logits (CPU only).

Both NPZs come from the same deterministic test dataloader; sample identity is
verified via (label, user, item) alignment before any blending.

Methods (no free parameter fitted anywhere; no Valid-based weight search):
  E0_RR1 / E0_RR5        single-model z32 controls
  E1_ZMEAN               mean of per-model z-scored z32 logits
  E2_RANKMEAN            mean of per-model percentile ranks
  E3_ZMEAN_L31BLEND      both models use s=z32+1.6*(z31-z32) (the valid-
                         selected alpha from the post-hoc study), then z-mean
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments" / "no31_cora_original"
RR1_NPZ = EXP / "out" / "CORA_RR1_TEST_LOGITS_20261112" / "layer_logits_test.npz"
RR5_NPZ = EXP / "out" / "CORA_RR5_TEST_LOGITS_20261112" / "layer_logits_test.npz"
OUT_DIR = EXP / "out" / "CORA_ENSEMBLE_RR1_RR5_TEST_20261112"


def _auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos, neg = labels == 1, labels == 0
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sv = scores[order]
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sv[j] == sv[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2.0)
                 / (pos.sum() * neg.sum()))


def _uauc(labels, users, scores):
    users = np.asarray(users)
    values = []
    for uid in np.unique(users):
        mask = users == uid
        y = labels[mask]
        if np.any(y == 1) and np.any(y == 0):
            values.append(_auc(y, scores[mask]))
    return float(np.mean(values)) if values else float("nan")


def _load(path):
    with np.load(path, allow_pickle=False) as data:
        return (np.asarray(data["labels"], dtype=np.int64),
                np.asarray(data["users"], dtype=np.int64),
                np.asarray(data["items"], dtype=np.int64),
                np.asarray(data["yes_logits"], dtype=np.float64))


def _z(x):
    return (x - x.mean()) / max(x.std(), 1e-12)


def _rank(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    return ranks / len(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rr1", default=str(RR1_NPZ))
    parser.add_argument("--rr5", default=str(RR5_NPZ))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    y1, u1, i1, z1 = _load(Path(args.rr1))
    y5, u5, i5, z5 = _load(Path(args.rr5))
    identity = (np.array_equal(y1, y5) and np.array_equal(u1, u5)
                and np.array_equal(i1, i5))
    if not identity:
        raise RuntimeError("RR1/RR5 test NPZ sample identity mismatch; "
                           "refusing to ensemble misaligned samples")
    y, u = y1, u1

    rows = []

    def record(method, scores, note):
        rows.append(dict(method=method, test_auc=f"{_auc(y, scores):.9f}",
                         test_uauc=f"{_uauc(y, u, scores):.9f}", note=note))
        print(f"[{method}] test AUC={rows[-1]['test_auc']} "
              f"UAUC={rows[-1]['test_uauc']}  ({note})")

    record("E0_RR1", z1[3], "z32 control, official 0.739652")
    record("E0_RR5", z5[3], "z32 control, official 0.739113")
    record("E1_ZMEAN", 0.5 * (_z(z1[3]) + _z(z5[3])), "mean of z-scored z32")
    record("E2_RANKMEAN", 0.5 * (_rank(z1[3]) + _rank(z5[3])),
           "mean of percentile ranks")
    a = 1.6
    b1 = z1[3] + a * (z1[2] - z1[3])
    b5 = z5[3] + a * (z5[2] - z5[3])
    record("E3_ZMEAN_L31BLEND", 0.5 * (_z(b1) + _z(b5)),
           f"both s=z32+{a}*(z31-z32) then z-mean")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "ensemble_test.csv").open("w", newline="",
                                              encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "ensemble_test.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ENSEMBLE SAVED] {out_dir}")


if __name__ == "__main__":
    main()
