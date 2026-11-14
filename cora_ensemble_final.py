#!/usr/bin/env python3
"""Valid-weight-searched ensembles of RR1, RR5(fused), G3(fused) -> TEST.

All weight selection happens on VALID; TEST is scored once per selected
configuration. Model scores:
  A = RR1 z32 (raw final layer)
  B = RR5 fused = z32 + sigmoid(g(u)) * (z31 - z32)   [gate rebuilt offline]
  C = G3 fused = z32 + 0.25 * (z31 - z32)             [fixed alpha]

Selection rules reported side by side:
  R_AUC    maximize valid AUC
  R_COMP   maximize valid AUC + 0.25 * valid UAUC
Each rule's winner is applied to TEST once (2-model and 3-model variants),
plus parameter-free 1/2 and 1/3 z-mean baselines.
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
RR5_CKPT = (EXP / "output" / "CORA_RESIDUAL_ROUTING6_20261103" /
            "RR5_USER_GATE_R31" / "20261103153" / "checkpoint_best.pth")
OUT_DIR = EXP / "out" / "CORA_ENSEMBLE_FINAL_20261112"


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


def _rr5_fused(npz_path, users):
    y, u, i, z = _load(npz_path)
    if not np.array_equal(u, users):
        raise RuntimeError("RR5 user identity mismatch")
    state = torch.load(str(RR5_CKPT), map_location="cpu", weights_only=False)["model"]
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
    return z[3] + gate * (z[2] - z[3]), y, u, i


def _g3_fused(npz_path, users):
    y, u, i, z = _load(npz_path)
    if not np.array_equal(u, users):
        raise RuntimeError("G3 user identity mismatch")
    return z[3] + 0.25 * (z[2] - z[3]), y, u, i


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    yv, uv, iv, zv = _load(RR1_VALID)
    yt, ut, it, zt = _load(RR1_TEST)
    Bv, yv5, uv5, _ = _rr5_fused(RR5_VALID, uv)
    Bt, _, _, _ = _rr5_fused(RR5_TEST, ut)
    Cv, yg3, ug3, _ = _g3_fused(G3_VALID, uv)
    Ct, _, _, _ = _g3_fused(G3_TEST, ut)
    if not (np.array_equal(yv, yv5) and np.array_equal(uv, uv5)
            and np.array_equal(yv, yg3)):
        raise RuntimeError("Valid sample identity mismatch across models")

    Av, At = _z(zv[3]), _z(zt[3])
    Bv, Bt = _z(Bv), _z(Bt)
    Cv, Ct = _z(Cv), _z(Ct)

    print(f"[SINGLE valid] A(RR1)={_auc(yv, Av):.6f} B(RR5)={_auc(yv, Bv):.6f} "
          f"C(G3)={_auc(yv, Cv):.6f}")
    print(f"[SINGLE test ] A(RR1)={_auc(yt, At):.6f} B(RR5)={_auc(yt, Bt):.6f} "
          f"C(G3)={_auc(yt, Ct):.6f}")

    rows = []

    def record(tag, va, vu, ta, tu, note):
        rows.append(dict(config=tag, valid_auc=f"{va:.9f}", valid_uauc=f"{vu:.9f}",
                         test_auc=f"{ta:.9f}", test_uauc=f"{tu:.9f}", note=note))
        print(f"[{tag}] valid {va:.6f}/{vu:.6f} -> test {ta:.6f}/{tu:.6f}  ({note})")

    record("S_A_RR1", _auc(yv, Av), _uauc(yv, uv, Av),
           _auc(yt, At), _uauc(yt, ut, At), "single model control")
    record("S_B_RR5", _auc(yv, Bv), _uauc(yv, uv, Bv),
           _auc(yt, Bt), _uauc(yt, ut, Bt), "single model control")

    # ---- 2-model A+B weight search on valid ----
    grid = np.round(np.arange(0.0, 1.0001, 0.05), 3)
    stats2 = []
    for w in grid:
        s = w * Av + (1 - w) * Bv
        stats2.append((_auc(yv, s), _uauc(yv, uv, s), float(w)))
    sel_auc = max(stats2, key=lambda t: (t[0], t[1]))
    sel_comp = max(stats2, key=lambda t: (t[0] + 0.25 * t[1], t[0]))
    for tag, sel in (("E2_AB_AUC", sel_auc), ("E2_AB_COMP", sel_comp)):
        w = sel[2]
        st = w * At + (1 - w) * Bt
        sv = w * Av + (1 - w) * Bv
        record(tag, _auc(yv, sv), _uauc(yv, uv, sv),
               _auc(yt, st), _uauc(yt, ut, st), f"w_A={w:.2f} on valid")
    sv = 0.5 * Av + 0.5 * Bv
    st = 0.5 * At + 0.5 * Bt
    record("E2_AB_HALF", _auc(yv, sv), _uauc(yv, uv, sv),
           _auc(yt, st), _uauc(yt, ut, st), "fixed 0.5/0.5")

    # ---- 3-model A+B+C simplex search on valid ----
    stats3 = []
    for wa in grid:
        for wb in np.round(np.arange(0.0, 1.0001 - wa + 1e-9, 0.05), 3):
            wc = round(1.0 - wa - wb, 3)
            if wc < -1e-9:
                continue
            wc = max(wc, 0.0)
            s = wa * Av + wb * Bv + wc * Cv
            stats3.append((_auc(yv, s), _uauc(yv, uv, s), float(wa), float(wb), float(wc)))
    sel3_auc = max(stats3, key=lambda t: (t[0], t[1]))
    sel3_comp = max(stats3, key=lambda t: (t[0] + 0.25 * t[1], t[0]))
    for tag, sel in (("E3_ABC_AUC", sel3_auc), ("E3_ABC_COMP", sel3_comp)):
        wa, wb, wc = sel[2], sel[3], sel[4]
        svt = wa * Av + wb * Bv + wc * Cv
        stt = wa * At + wb * Bt + wc * Ct
        record(tag, _auc(yv, svt), _uauc(yv, uv, svt),
               _auc(yt, stt), _uauc(yt, ut, stt),
               f"w=({wa:.2f},{wb:.2f},{wc:.2f}) on valid")
    svt = (Av + Bv + Cv) / 3.0
    stt = (At + Bt + Ct) / 3.0
    record("E3_ABC_THIRD", _auc(yv, svt), _uauc(yv, uv, svt),
           _auc(yt, stt), _uauc(yt, ut, stt), "fixed 1/3 each")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "ensemble_final.csv").open("w", newline="",
                                               encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "ensemble_final.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ENSEMBLE FINAL SAVED] {OUT_DIR}")


if __name__ == "__main__":
    main()
