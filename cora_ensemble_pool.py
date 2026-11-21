#!/usr/bin/env python3
"""CPU-only ensemble pool expansion: add HP11/HP15/HP27/MC to the search.

Requires valid+test captures for each candidate. GPU captures are queued
separately (capture entries run when GPU is free); this script consumes
whatever NPZs already exist and reports which candidates are usable.
Weights searched on VALID only; TEST scored once per configuration.
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
HP_TEST_LOGITS = EXP / "out" / "CORA_HP_TEST_LOGITS_20261115"
RR1_VALID = ROOT / "out" / "LATE_LAYER_DIAG_D4_C6_FIX_20261102_165625" / "layer_logits_valid.npz"
RR1_TEST = EXP / "out" / "CORA_RR1_TEST_LOGITS_20261112" / "layer_logits_test.npz"
RR5_VALID = CAP / "RR5_USER_GATE_R31_valid" / "layer_logits_valid.npz"
RR5_TEST = EXP / "out" / "CORA_RR5_TEST_LOGITS_20261112" / "layer_logits_test.npz"
HP12_VALID = EXP / "out" / "CORA_HP_VALID_LOGITS_20261115V" / "HP12_FINE_B07_valid" / "layer_logits_valid.npz"
HP12_TEST = EXP / "out" / "CORA_HP_TEST_LOGITS_20261115" / "HP12_FINE_B07" / "layer_logits_test.npz"
HP24_VALID = EXP / "out" / "CORA_HP_TEST_LOGITS_20261115" / "HP24_B07_T05_valid" / "layer_logits_valid.npz"
HP24_TEST = EXP / "out" / "CORA_HP_TEST_LOGITS_20261115" / "HP24_B07_T05_test" / "layer_logits_test.npz"
RR5_CKPT = (EXP / "output" / "CORA_RESIDUAL_ROUTING6_20261103" /
            "RR5_USER_GATE_R31" / "20261103153" / "checkpoint_best.pth")
CANDIDATES = ["HP11_FINE_B03", "HP15_FINE_T05", "HP27_B07_P50", "MC_ITEMROUTE"]
OUT_DIR = EXP / "out" / "CORA_ENSEMBLE_POOL_20261117"


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
                np.asarray(data["yes_logits"], dtype=np.float64))


def _z(x):
    return (x - x.mean()) / max(x.std(), 1e-12)


def _gate_fused(npz_path, ckpt_path, users):
    y, u, z = _load(npz_path)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=8,
                        help="how many candidates to include beyond the 4 bases")
    args = parser.parse_args()

    yv, uv, zv = _load(RR1_VALID)
    yt, ut, zt = _load(RR1_TEST)
    Bv = _z(_gate_fused(RR5_VALID, RR5_CKPT, uv))
    Bt = _z(_gate_fused(RR5_TEST, RR5_CKPT, ut))
    yv2, uv2, cv_raw = _load(HP12_VALID)
    yt2, ut2, ct_raw = _load(HP12_TEST)
    yv3, uv3, dv_raw = _load(HP24_VALID)
    yt3, ut3, dt_raw = _load(HP24_TEST)
    y, u = yv, uv
    for (a, b) in ((yv, yv2), (uv, uv2), (yv, yv3), (uv, uv3),
                   (yt, yt2), (ut, ut2), (yt, yt3), (ut, ut3)):
        if not np.array_equal(a, b):
            raise RuntimeError("Sample identity mismatch across NPZs")
    Vm = [_z(zv[3]), Bv, _z(cv_raw[3]), _z(dv_raw[3])]
    Tm = [_z(zt[3]), Bt, _z(ct_raw[3]), _z(dt_raw[3])]
    names = ["RR1", "RR5", "HP12", "HP24"]

    # Add available candidates (need valid+test NPZs both present)
    added = []
    for cand in CANDIDATES[:args.top_k]:
        vpath = HP_TEST_LOGITS / f"{cand}_valid" / "layer_logits_valid.npz"
        tpath = HP_TEST_LOGITS / f"{cand}_test" / "layer_logits_test.npz"
        if vpath.is_file() and tpath.is_file():
            yv4, uv4, cv4 = _load(vpath)
            yt4, ut4, ct4 = _load(tpath)
            if not (np.array_equal(yv, yv4) and np.array_equal(uv, uv4)
                    and np.array_equal(yt, yt4) and np.array_equal(ut, ut4)):
                print(f"[POOL] {cand}: identity mismatch; skipped")
                continue
            Vm.append(_z(cv4[3]))
            Tm.append(_z(ct4[3]))
            names.append(cand.split("_")[0])
            added.append(cand)
        else:
            print(f"[POOL] {cand}: missing capture ({'valid' if not vpath.is_file() else 'test'}); skipped")
    print(f"[POOL] bases=4 added={len(added)} {added}")
    n = len(Vm)
    print("[SINGLE test]", {names[i]: f"{_auc(yt, Tm[i]):.6f}/{_uauc(yt, ut, Tm[i]):.6f}"
                            for i in range(n)})

    rows = []

    def record(tag, va, vu, ta, tu, note):
        rows.append(dict(config=tag, valid_auc=f"{va:.9f}",
                         valid_uauc=f"{vu:.9f}", test_auc=f"{ta:.9f}",
                         test_uauc=f"{tu:.9f}", note=note))
        print(f"[{tag}] valid {va:.6f}/{vu:.6f} -> test {ta:.6f}/{tu:.6f}  ({note})")

    import itertools
    step = 0.1 if n > 4 else 0.05
    grid = np.round(np.arange(0.0, 1.0001, step), 4)
    combos = [w for w in itertools.product(grid, repeat=n) if abs(sum(w) - 1.0) < 1e-6]
    best_auc = best_comp = best_eq = None
    eq_w = tuple([1.0 / n] * n)
    for weights in combos + [eq_w]:
        s = sum(w * Vm[i] for w, i in zip(weights, range(n)))
        a, u_ = _auc(y, s), _uauc(y, uv, s)
        tag = "EQ" if weights == eq_w else None
        if best_auc is None or (a, u_) > best_auc[:2]:
            best_auc = (a, u_, weights, tag == "EQ")
        if best_comp is None or (a + 0.25 * u_, a) > \
                (best_comp[0] + 0.25 * best_comp[1], best_comp[0]):
            best_comp = (a, u_, weights, tag == "EQ")
        if weights == eq_w:
            best_eq = (a, u_, weights)
    for label, sel in (("AUC", best_auc), ("COMP", best_comp)):
        weights = sel[2]
        sv = sum(w * Vm[i] for w, i in zip(weights, range(n)))
        st = sum(w * Tm[i] for w, i in zip(weights, range(n)))
        note = "equal weights" if sel[3] else f"w={dict(zip(names, weights))} on valid"
        record(f"E_ALL_{label}", _auc(y, sv), _uauc(y, uv, sv),
               _auc(yt, st), _uauc(yt, ut, st), note)
    sv = sum(w * Vm[i] for w, i in zip(best_eq[2], range(n)))
    st = sum(w * Tm[i] for w, i in zip(best_eq[2], range(n)))
    record("E_ALL_EQ", _auc(y, sv), _uauc(y, uv, sv),
           _auc(yt, st), _uauc(yt, ut, st), f"equal 1/{n}")
    # Greedy forward selection on valid (AUC), then one-shot test
    chosen, cur = [], None
    remaining = list(range(n))
    while remaining and len(chosen) < 6:
        best_i, best_v = None, None
        for i in remaining:
            cand_idx = chosen + [i]
            s = sum(Vm[i2] / (len(cand_idx)) for i2 in cand_idx)
            v = _auc(y, s)
            if best_v is None or v > best_v:
                best_v, best_i = v, i
        if cur is not None and best_v <= cur + 1e-6:
            break
        chosen.append(best_i)
        remaining.remove(best_i)
        cur = best_v
    k = len(chosen)
    sv = sum(Vm[i] / k for i in chosen)
    st = sum(Tm[i] / k for i in chosen)
    record("E_GREEDY", _auc(y, sv), _uauc(y, uv, sv),
           _auc(yt, st), _uauc(yt, ut, st), f"greedy {k}-model: "
           f"{[names[i] for i in chosen]}")

    # ---- Per-user mixture (S5 style) with shrinkage, fit on valid ----
    gw = np.array([_auc(y, Vm[i]) for i in range(n)])
    gw = gw / gw.sum()
    lam = 40.0
    W_by_uid = {}
    for uid in np.unique(uv):
        mask = uv == uid
        n_u = int(mask.sum())
        per = np.array([_auc(y[mask], Vm[i][mask]) for i in range(n)])
        per = np.nan_to_num(per, nan=0.5)
        w = (lam * gw + n_u * per ** 4) / (lam + n_u)
        W_by_uid[int(uid)] = w / w.sum()
    sv = np.einsum("ij,ij->i", np.stack(Vm, axis=1),
                   np.stack([W_by_uid[int(x)] for x in uv]))
    Wt = np.stack([W_by_uid.get(int(x), gw) for x in ut])
    st = np.einsum("ij,ij->i", np.stack(Tm, axis=1), Wt)
    record("S_USERMIX", _auc(y, sv), _uauc(y, uv, sv),
           _auc(yt, st), _uauc(yt, ut, st), f"per-user mixture, lambda={lam}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "ensemble_pool.csv").open("w", newline="",
                                              encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "ensemble_pool.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[POOL SAVED] {OUT_DIR}")


if __name__ == "__main__":
    main()
