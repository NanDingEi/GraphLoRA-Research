#!/usr/bin/env python3
"""Three-family test ensemble: RR1 (residual baseline) + RR5 (user-gate
residual) + HP12 (hierarchical cluster routing). Weights searched on VALID
only; TEST scored once per selected configuration.

  A = RR1 z32                          (official test 0.739654/0.706352)
  B = RR5 fused = z32+sigmoid(g(u))(z31-z32)  (official test 0.739113/0.705807)
  C = HP12 final-layer yes logit       (official test 0.737195/0.711361)
"""
import csv
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
HP12_VALID = EXP / "out" / "CORA_HP_VALID_LOGITS_20261115V" / "HP12_FINE_B07_valid" / "layer_logits_valid.npz"
HP12_TEST = EXP / "out" / "CORA_HP_TEST_LOGITS_20261115" / "HP12_FINE_B07" / "layer_logits_test.npz"
RR5_CKPT = (EXP / "output" / "CORA_RESIDUAL_ROUTING6_20261103" /
            "RR5_USER_GATE_R31" / "20261103153" / "checkpoint_best.pth")
OUT_DIR = EXP / "out" / "CORA_ENSEMBLE_THREE_FAMILY_20261115"


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


def _load(path, split):
    with np.load(path, allow_pickle=False) as data:
        return (np.asarray(data["labels"], dtype=np.int64),
                np.asarray(data["users"], dtype=np.int64),
                np.asarray(data["yes_logits"], dtype=np.float64))


def _z(x):
    return (x - x.mean()) / max(x.std(), 1e-12)


def _gate_fused(npz_path, ckpt_path, users):
    y, u, z = _load(npz_path, "fused-source")
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


def main():
    yv, uv, zv = _load(RR1_VALID, "valid")
    yt, ut, zt = _load(RR1_TEST, "test")
    Bv = _z(_gate_fused(RR5_VALID, RR5_CKPT, uv))
    Bt = _z(_gate_fused(RR5_TEST, RR5_CKPT, ut))
    yv2, uv2, cv_raw = _load(HP12_VALID, "valid")
    yt2, ut2, ct_raw = _load(HP12_TEST, "test")
    if not (np.array_equal(yv, yv2) and np.array_equal(uv, uv2)):
        raise RuntimeError("Valid identity mismatch RR1 vs HP12")
    if not (np.array_equal(yt, yt2) and np.array_equal(ut, ut2)):
        raise RuntimeError("Test identity mismatch RR1 vs HP12")
    Av, At = _z(zv[3]), _z(zt[3])
    Cv, Ct = _z(cv_raw[3]), _z(ct_raw[3])
    Vm, Tm, names = [Av, Bv, Cv], [At, Bt, Ct], "ABC"

    print("[SINGLE valid]", {n: f"{_auc(yv, m):.6f}" for n, m in zip(names, Vm)})
    print("[SINGLE test ]", {n: f"{_auc(yt, m):.6f}/{_uauc(yt, ut, m):.6f}"
                             for n, m in zip(names, Tm)})

    rows = []

    def record(tag, va, vu, ta, tu, note):
        rows.append(dict(config=tag, valid_auc=f"{va:.9f}",
                         valid_uauc=f"{vu:.9f}", test_auc=f"{ta:.9f}",
                         test_uauc=f"{tu:.9f}", note=note))
        print(f"[{tag}] valid {va:.6f}/{vu:.6f} -> test {ta:.6f}/{tu:.6f}  ({note})")

    record("S_A_RR1", _auc(yv, Av), _uauc(yv, uv, Av),
           _auc(yt, At), _uauc(yt, ut, At), "single control")
    record("S_C_HP12", _auc(yv, Cv), _uauc(yv, uv, Cv),
           _auc(yt, Ct), _uauc(yt, ut, Ct), "single control")

    import itertools
    grid = np.round(np.arange(0.0, 1.0001, 0.05), 4)
    for subset in ((0, 2), (1, 2), (0, 1, 2)):
        combos = []
        if len(subset) == 2:
            combos = [(w, 1 - w) for w in grid]
        else:
            combos = [w for w in itertools.product(grid, repeat=3)
                      if abs(sum(w) - 1.0) < 1e-6]
        best_auc = best_comp = None
        for weights in combos:
            s = sum(w * Vm[i] for w, i in zip(weights, subset))
            a, u_ = _auc(yv, s), _uauc(yv, uv, s)
            if best_auc is None or (a, u_) > best_auc[:2]:
                best_auc = (a, u_, weights)
            if best_comp is None or (a + 0.25 * u_, a) > \
                    (best_comp[0] + 0.25 * best_comp[1], best_comp[0]):
                best_comp = (a, u_, weights)
        name = "".join(names[i] for i in subset)
        for tag, sel in (("AUC", best_auc), ("COMP", best_comp)):
            weights = sel[2]
            sv = sum(w * Vm[i] for w, i in zip(weights, subset))
            st = sum(w * Tm[i] for w, i in zip(weights, subset))
            record(f"E_{name}_{tag}", _auc(yv, sv), _uauc(yv, uv, sv),
                   _auc(yt, st), _uauc(yt, ut, st),
                   f"w={dict(zip(name, weights))} on valid")
        w0 = 1.0 / len(subset)
        sv = sum(w0 * Vm[i] for i in subset)
        st = sum(w0 * Tm[i] for i in subset)
        record(f"E_{name}_EQ", _auc(yv, sv), _uauc(yv, uv, sv),
               _auc(yt, st), _uauc(yt, ut, st), f"equal 1/{len(subset)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "ensemble_three_family.csv").open("w", newline="",
                                                      encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "ensemble_three_family.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[THREE FAMILY SAVED] {OUT_DIR}")


if __name__ == "__main__":
    main()
