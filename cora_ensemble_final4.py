#!/usr/bin/env python3
"""Final 4-model ensembles: RR1, RR5(fused), G3(fused), G5(fused).

Weights selected on VALID only (grid search); TEST scored once per
pre-registered selection rule. Fixed equal-weight baselines included.

  A = RR1 z32                                     (official model)
  B = RR5 fused  = z32 + sigmoid(g(u))*(z31-z32)  [RR5 ckpt gate]
  C = G3  fused  = z32 + 0.25*(z31-z32)           [fixed alpha]
  D = G5  fused  = z32 + sigmoid(g'(u))*(z31-z32) [G5 ckpt gate]
"""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments" / "no31_cora_original"
CAP = EXP / "out" / "CORA_G6_CAPTURE_20261112"
RR1_VALID = ROOT / "out" / "LATE_LAYER_DIAG_D4_C6_FIX_20261102_165625" / "layer_logits_valid.npz"
RR1_TEST = EXP / "out" / "CORA_RR1_TEST_LOGITS_20261112" / "layer_logits_test.npz"
RR5_VALID = CAP / "RR5_USER_GATE_R31_valid" / "layer_logits_valid.npz"
RR5_TEST = EXP / "out" / "CORA_RR5_TEST_LOGITS_20261112" / "layer_logits_test.npz"
G3_VALID = CAP / "G3_FIXED_A025_valid" / "layer_logits_valid.npz"
G3_TEST = CAP / "G3_FIXED_A025_test" / "layer_logits_test.npz"
G5_VALID = CAP / "G5_UGATE_B20_valid" / "layer_logits_valid.npz"
G5_TEST = CAP / "G5_UGATE_B20_test" / "layer_logits_test.npz"
RR5_CKPT = (EXP / "output" / "CORA_RESIDUAL_ROUTING6_20261103" /
            "RR5_USER_GATE_R31" / "20261103153" / "checkpoint_best.pth")
G5_CKPT_DIR = (EXP / "output" / "CORA_GATE_TUNE_20261112" / "G5_UGATE_B20")
OUT_DIR = EXP / "out" / "CORA_ENSEMBLE_FINAL4_20261112"


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


def _gate_fused(npz_path, ckpt_path, users):
    y, u, i, z = _load(npz_path)
    if not np.array_equal(u, users):
        raise RuntimeError("user identity mismatch")
    state = torch.load(str(ckpt_path), map_location="cpu",
                       weights_only=False)["model"]
    table = state["rec_encoder.user_embedding.weight"].float()
    gk = {k: v for k, v in state.items()
          if k.startswith("cora_residual_fusion.user_gate.")}
    user = F.normalize(table[torch.as_tensor(u, dtype=torch.long)], dim=-1)
    h = F.layer_norm(user, (user.shape[1],),
                     gk["cora_residual_fusion.user_gate.0.weight"].float(),
                     gk["cora_residual_fusion.user_gate.0.bias"].float(), 1e-5)
    h = F.gelu(F.linear(h, gk["cora_residual_fusion.user_gate.1.weight"].float(),
                        gk["cora_residual_fusion.user_gate.1.bias"].float()))
    gate = torch.sigmoid(F.linear(
        h, gk["cora_residual_fusion.user_gate.3.weight"].float(),
        gk["cora_residual_fusion.user_gate.3.bias"].float()).reshape(-1)).numpy()
    return z[3] + gate * (z[2] - z[3])


def _g3_fused(npz_path, users):
    y, u, i, z = _load(npz_path)
    if not np.array_equal(u, users):
        raise RuntimeError("user identity mismatch")
    return z[3] + 0.25 * (z[2] - z[3])


def _resolve_g5_ckpt():
    matches = sorted(G5_CKPT_DIR.glob("*/checkpoint_best.pth"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one G5 checkpoint: {matches}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    yv, uv, iv, zv = _load(RR1_VALID)
    yt, ut, it, zt = _load(RR1_TEST)
    g5_ckpt = _resolve_g5_ckpt()
    Bv = _z(_gate_fused(RR5_VALID, RR5_CKPT, uv))
    Bt = _z(_gate_fused(RR5_TEST, RR5_CKPT, ut))
    Cv = _z(_g3_fused(G3_VALID, uv))
    Ct = _z(_g3_fused(G3_TEST, ut))
    Dv = _z(_gate_fused(G5_VALID, g5_ckpt, uv))
    Dt = _z(_gate_fused(G5_TEST, g5_ckpt, ut))
    Av, At = _z(zv[3]), _z(zt[3])
    names = "ABCD"
    Vm = [Av, Bv, Cv, Dv]
    Tm = [At, Bt, Ct, Dt]

    print("[SINGLE valid]", {n: f"{_auc(yv, m):.6f}" for n, m in zip(names, Vm)})
    print("[SINGLE test ]", {n: f"{_auc(yt, m):.6f}" for n, m in zip(names, Tm)})

    rows = []

    def record(tag, va, vu, ta, tu, note):
        rows.append(dict(config=tag, valid_auc=f"{va:.9f}",
                         valid_uauc=f"{vu:.9f}", test_auc=f"{ta:.9f}",
                         test_uauc=f"{tu:.9f}", note=note))
        print(f"[{tag}] valid {va:.6f}/{vu:.6f} -> test {ta:.6f}/{tu:.6f}  ({note})")

    record("S_A_RR1", _auc(yv, Av), _uauc(yv, uv, Av),
           _auc(yt, At), _uauc(yt, ut, At), "single-model control")

    def search(subset, step):
        k = len(subset)
        best_auc = best_comp = None
        if k == 2:
            combos = [(w, 1 - w) for w in np.round(np.arange(0, 1.0001, step), 4)]
        else:
            combos = []
            for weights in itertools.product(np.round(np.arange(0, 1.0001, step), 4),
                                             repeat=k):
                if abs(sum(weights) - 1.0) > 1e-6:
                    continue
                combos.append(weights)
        for weights in combos:
            s = sum(w * Vm[i] for w, i in zip(weights, subset))
            a, u_ = _auc(yv, s), _uauc(yv, uv, s)
            key = (a, u_, weights)
            if best_auc is None or (a, u_) > (best_auc[0], best_auc[1]):
                best_auc = key
            if best_comp is None or (a + 0.25 * u_, a) > (best_comp[0] + 0.25 * best_comp[1], best_comp[0]):
                best_comp = key
        results = []
        for tag, sel in (("AUC", best_auc), ("COMP", best_comp)):
            weights = sel[2]
            sv = sum(w * Vm[i] for w, i in zip(weights, subset))
            st = sum(w * Tm[i] for w, i in zip(weights, subset))
            name = "".join(names[i] for i in subset)
            results.append((f"E_{name}_{tag}",
                            _auc(yv, sv), _uauc(yv, uv, sv),
                            _auc(yt, st), _uauc(yt, ut, st),
                            f"w={dict(zip(name, weights))}"))
        # equal-weight baseline
        w0 = 1.0 / k
        sv = sum(w0 * Vm[i] for i in subset)
        st = sum(w0 * Tm[i] for i in subset)
        name = "".join(names[i] for i in subset)
        results.append((f"E_{name}_EQ", _auc(yv, sv), _uauc(yv, uv, sv),
                        _auc(yt, st), _uauc(yt, ut, st), f"equal 1/{k}"))
        return results

    for subset, step in (((0, 1), 0.05), ((0, 1, 2), 0.05),
                         ((0, 1, 3), 0.05), ((0, 1, 2, 3), 0.1)):
        for r in search(subset, step):
            record(*r)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "ensemble_final4.csv").open("w", newline="",
                                                encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "ensemble_final4.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ENSEMBLE FINAL4 SAVED] {OUT_DIR}")


if __name__ == "__main__":
    main()
