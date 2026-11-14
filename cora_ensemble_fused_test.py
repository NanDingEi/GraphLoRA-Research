#!/usr/bin/env python3
"""Rebuild RR5's FUSED test logits offline and ensemble with RR1 (CPU only).

RR5 fusion: u = normalize(user_mf); g = MLP(u); s = sigmoid(g);
fused = z32 + s * (z31 - z32).  All parameters are in the RR5 checkpoint
(cora_residual_fusion.user_gate.*) plus the MF user table.

Ensembles (no fitted parameter; fixed 0.5 mixing):
  F1_ZMEAN_FUSED     0.5*(z(RR1 z32) + z(RR5 fused))
  F2_RANKMEAN_FUSED  0.5*(rank(RR1 z32) + rank(RR5 fused))
  F3_RANK3           mean of ranks: RR1 z32, RR5 fused, RR1 L30
  F4_RANK3B          mean of ranks: RR1 z32, RR1 L31blend(a=1.6), RR5 fused
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments" / "no31_cora_original"
RR1_NPZ = EXP / "out" / "CORA_RR1_TEST_LOGITS_20261112" / "layer_logits_test.npz"
RR5_NPZ = EXP / "out" / "CORA_RR5_TEST_LOGITS_20261112" / "layer_logits_test.npz"
RR5_CKPT = (EXP / "output" / "CORA_RESIDUAL_ROUTING6_20261103" /
            "RR5_USER_GATE_R31" / "20261103153" / "checkpoint_best.pth")
OUT_DIR = EXP / "out" / "CORA_ENSEMBLE_FUSED_TEST_20261112"


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


def rebuild_rr5_fused(rr5_npz, checkpoint):
    y, u, i, z = _load(rr5_npz)
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = payload["model"]
    user_table = state["rec_encoder.user_embedding.weight"].float()
    gate_keys = {k: v for k, v in state.items()
                 if k.startswith("cora_residual_fusion.user_gate.")}
    if len(gate_keys) != 6:
        raise RuntimeError(f"Expected 6 user_gate tensors, got {sorted(gate_keys)}")

    def lin(k, x):
        return F.linear(x, gate_keys[f"cora_residual_fusion.user_gate.{k}.weight"].float(),
                        gate_keys[f"cora_residual_fusion.user_gate.{k}.bias"].float())

    user = F.normalize(user_table[torch.as_tensor(u, dtype=torch.long)], dim=-1)
    affine_w = gate_keys["cora_residual_fusion.user_gate.0.weight"].float()
    affine_b = gate_keys["cora_residual_fusion.user_gate.0.bias"].float()
    if affine_w.numel() != user.shape[1] or affine_b.numel() != user.shape[1]:
        raise RuntimeError("user_gate LayerNorm affine shape mismatch")
    h = F.layer_norm(user, (user.shape[1],), affine_w, affine_b, 1e-5)
    h = F.gelu(lin("1", h))
    gate_logit = lin("3", h).reshape(-1)
    strength = torch.sigmoid(gate_logit).numpy()
    fused = z[3] + strength * (z[2] - z[3])
    return y, u, i, z, fused, float(strength.mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    y1, u1, i1, z1 = _load(RR1_NPZ)
    y5, u5, i5, z5, fused5, gate_mean = rebuild_rr5_fused(RR5_NPZ, RR5_CKPT)
    if not (np.array_equal(y1, y5) and np.array_equal(u1, u5)
            and np.array_equal(i1, i5)):
        raise RuntimeError("Sample identity mismatch between RR1 and RR5 NPZs")
    y, u = y1, u1
    print(f"[FUSED REBUILD] gate_mean={gate_mean:.4f} "
          f"fused AUC={_auc(y, fused5):.6f} (official 0.739113) "
          f"UAUC={_uauc(y, u, fused5):.6f} (official 0.705807)")

    rows = []

    def record(method, scores, note):
        rows.append(dict(method=method, test_auc=f"{_auc(y, scores):.9f}",
                         test_uauc=f"{_uauc(y, u, scores):.9f}", note=note))
        print(f"[{method}] test AUC={rows[-1]['test_auc']} "
              f"UAUC={rows[-1]['test_uauc']}  ({note})")

    record("F0_RR1", z1[3], "control 0.739652")
    record("F0_RR5_FUSED", fused5, "control 0.739113")
    record("F1_ZMEAN_FUSED", 0.5 * (_z(z1[3]) + _z(fused5)),
           "0.5*(z(RR1 z32)+z(RR5 fused))")
    record("F2_RANKMEAN_FUSED", 0.5 * (_rank(z1[3]) + _rank(fused5)),
           "0.5*(rank(RR1 z32)+rank(RR5 fused))")
    record("F3_RANK3", (_rank(z1[3]) + _rank(fused5) + _rank(z1[1])) / 3.0,
           "mean rank: RR1 z32, RR5 fused, RR1 L30")
    blend31 = z1[3] + 1.6 * (z1[2] - z1[3])
    record("F4_RANK3B", (_rank(z1[3]) + _rank(blend31) + _rank(fused5)) / 3.0,
           "mean rank: RR1 z32, RR1 a1.6-L31, RR5 fused")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "ensemble_fused_test.csv").open("w", newline="",
                                                    encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "ensemble_fused_test.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[FUSED ENSEMBLE SAVED] {OUT_DIR}")


if __name__ == "__main__":
    main()
