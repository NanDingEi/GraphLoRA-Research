#!/usr/bin/env python3
"""CPU-only experiment studio: five score-space studies on captured logits.

All experiments use ONLY the captured valid/test NPZs already on disk —
no GPU, no model loading. Every fit happens on VALID; TEST is scored once
per configuration for transparency.

  S1_SCORE_INTERP   continuous interpolation between HP12 (T=0.25) and
                    HP24 (T=0.5) scores — samples the routing-temperature
                    axis without GPU training.
  S2_RANK_ENSEMBLE  scale-free rank averaging (parameter-free) over all
                    captured models, incl. mixed AUC-weights.
  S3_CALIBRATION    per-model Platt scaling fit on VALID, applied to TEST,
                    then z-mean ensemble — does calibration improve the mix?
  S4_DIVERSITY      error-correlation matrix over models on VALID — which
                    models are complementary (guides future GPU work).
  S5_USER_MIXTURE   per-user model mixture weights fit on VALID with strong
                    shrinkage toward the global optimum (leakage-safe).
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
HP24_VALID = EXP / "out" / "CORA_HP_TEST_LOGITS_20261115" / "HP24_B07_T05_valid" / "layer_logits_valid.npz"
HP24_TEST = EXP / "out" / "CORA_HP_TEST_LOGITS_20261115" / "HP24_B07_T05_test" / "layer_logits_test.npz"
RR5_CKPT = (EXP / "output" / "CORA_RESIDUAL_ROUTING6_20261103" /
            "RR5_USER_GATE_R31" / "20261103153" / "checkpoint_best.pth")
OUT_DIR = EXP / "out" / "CORA_CPU_STUDIO_20261120"


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
    yv, uv, zv = _load(RR1_VALID)
    yt, ut, zt = _load(RR1_TEST)
    Bv = _z(_gate_fused(RR5_VALID, RR5_CKPT, uv))
    Bt = _z(_gate_fused(RR5_TEST, RR5_CKPT, ut))
    yv2, uv2, cv_raw = _load(HP12_VALID)
    yt2, ut2, ct_raw = _load(HP12_TEST)
    yv3, uv3, dv_raw = _load(HP24_VALID)
    yt3, ut3, dt_raw = _load(HP24_TEST)
    for (a, b) in ((yv, yv2), (uv, uv2), (yv, yv3), (uv, uv3),
                   (yt, yt2), (ut, ut2), (yt, yt3), (ut, ut3)):
        if not np.array_equal(a, b):
            raise RuntimeError("Sample identity mismatch")
    V = {"RR1": _z(zv[3]), "RR5": Bv, "HP12": _z(cv_raw[3]), "HP24": _z(dv_raw[3])}
    T = {"RR1": _z(zt[3]), "RR5": Bt, "HP12": _z(ct_raw[3]), "HP24": _z(dt_raw[3])}
    rows = []

    def record(tag, va, vu, ta, tu, note):
        rows.append(dict(config=tag, valid_auc=f"{va:.9f}",
                         valid_uauc=f"{vu:.9f}", test_auc=f"{ta:.9f}",
                         test_uauc=f"{tu:.9f}", note=note))
        print(f"[{tag}] valid {va:.6f}/{vu:.6f} -> test {ta:.6f}/{tu:.6f}  ({note})")

    # ---- S1: score interpolation HP12 <-> HP24 ----
    print("== S1 SCORE INTERPOLATION ==")
    best = None
    for a in np.round(np.arange(0.0, 1.0001, 0.05), 3):
        sv = a * V["HP12"] + (1 - a) * V["HP24"]
        v = _auc(yv, sv)
        if best is None or v > best[0]:
            best = (v, float(a))
    a = best[1]
    st = a * T["HP12"] + (1 - a) * T["HP24"]
    sv = a * V["HP12"] + (1 - a) * V["HP24"]
    record("S1_INTERP", _auc(yv, sv), _uauc(yv, uv, sv),
           _auc(yt, st), _uauc(yt, ut, st), f"alpha_HP12={a:.2f} on valid")
    sv, st = 0.5 * (V["HP12"] + V["HP24"]), 0.5 * (T["HP12"] + T["HP24"])
    record("S1_INTERP_EQ", _auc(yv, sv), _uauc(yv, uv, sv),
           _auc(yt, st), _uauc(yt, ut, st), "alpha=0.5 fixed")

    # ---- S2: rank ensemble ----
    print("== S2 RANK ENSEMBLE ==")
    RV = {k: _rank_of(v) for k, v in V.items()} if (_rank_of := None) else None
    # (rank defined inline below to keep the helper local)

    def _rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(1, len(x) + 1)
        return r / len(x)

    RV = {k: _rank(v) for k, v in V.items()}
    RT = {k: _rank(v) for k, v in T.items()}
    sv = np.mean([RV[k] for k in V], axis=0)
    st = np.mean([RT[k] for k in T], axis=0)
    record("S2_RANK4", _auc(yv, sv), _uauc(yv, uv, sv),
           _auc(yt, st), _uauc(yt, ut, st), "rank mean of 4")

    # ---- S3: per-model Platt calibration ----
    print("== S3 CALIBRATION ==")

    def platt(train_scores, train_y, apply_scores):
        a, b = 1.0, 0.0
        lr = 0.5
        for _ in range(3000):
            p = 1.0 / (1.0 + np.exp(-(a * train_scores + b)))
            g_a = float(np.mean((p - train_y) * train_scores))
            g_b = float(np.mean(p - train_y))
            a -= lr * g_a
            b -= lr * g_b
        return a * apply_scores + b

    cal_V, cal_T = {}, {}
    for k in V:
        cal_V[k] = platt(V[k], yv.astype(np.float64), V[k])
        cal_T[k] = platt(V[k], yv.astype(np.float64), T[k])
    sv = np.mean([cal_V[k] for k in V], axis=0)
    st = np.mean([cal_T[k] for k in T], axis=0)
    record("S3_CAL_MEAN", _auc(yv, sv), _uauc(yv, uv, sv),
           _auc(yt, st), _uauc(yt, ut, st), "Platt per model, mean")
    zv_cal = np.mean([_z(cal_V[k]) for k in V], axis=0)
    zt_cal = np.mean([_z(cal_T[k]) for k in V], axis=0)
    record("S3_CAL_ZMEAN", _auc(yv, zv_cal), _uauc(yv, uv, zv_cal),
           _auc(yt, zt_cal), _uauc(yt, ut, zt_cal), "Platt then z-mean")

    # ---- S4: error-correlation diversity matrix (valid) ----
    print("== S4 DIVERSITY ==")
    errs = {k: (V[k] >= 0).astype(np.int64) != yv for k in V}
    div = {}
    for a_k in V:
        for b_k in V:
            if a_k < b_k:
                both = float(np.mean(errs[a_k] & errs[b_k]))
                corr = float(np.corrcoef(errs[a_k].astype(float),
                                         errs[b_k].astype(float))[0, 1])
                div[f"{a_k}|{b_k}"] = dict(both_wrong=round(both, 4),
                                           err_corr=round(corr, 4))
    for k, v in sorted(div.items(), key=lambda t: t[1]["err_corr"]):
        print(f"  {k}: err_corr={v['err_corr']} both_wrong={v['both_wrong']}")

    # ---- S5: per-user mixture with shrinkage ----
    print("== S5 USER MIXTURE ==")
    users_all = np.unique(uv)
    global_w = np.array([_auc(yv, V[k]) for k in V])
    global_w = global_w / global_w.sum()
    lam = 40.0  # shrinkage: n_u pseudo-counts
    W = np.zeros((len(uv), len(V)))
    for i, uid in enumerate(users_all):
        mask = uv == uid
        n_u = int(mask.sum())
        per = np.array([_auc(yv[mask], V[k][mask]) for k in V])
        per = np.nan_to_num(per, nan=0.5)
        w_u = (lam * global_w + n_u * (per ** 4)) / (lam + n_u)
        W[i] = w_u / w_u.sum()
    W_by_uid = {int(uid): W[i] for i, uid in enumerate(users_all)}
    Wv = np.stack([W_by_uid[int(x)] for x in uv])                # [Nv, M]
    Wt = np.stack([W_by_uid.get(int(x), global_w) for x in ut])  # [Nt, M]
    sv = np.einsum("ij,ij->i", np.stack([V[k] for k in V], axis=1), Wv)
    st = np.einsum("ij,ij->i", np.stack([T[k] for k in T], axis=1), Wt)
    record("S5_USERMIX", _auc(yv, sv), _uauc(yv, uv, sv),
           _auc(yt, st), _uauc(yt, ut, st), f"per-user weights, lambda={lam}")

    # ---- S5b: mixture over the complementary subset only ----
    # S4 diversity says RR1 is the most redundant member (err_corr ~0.89
    # with HP24); drop it and keep the three most orthogonal models.
    sub = ["RR5", "HP12", "HP24"]
    gw2 = np.array([_auc(yv, V[k]) for k in sub])
    gw2 = gw2 / gw2.sum()
    W2 = {}
    for uid in users_all:
        mask = uv == uid
        n_u = int(mask.sum())
        per = np.array([_auc(yv[mask], V[k][mask]) for k in sub])
        per = np.nan_to_num(per, nan=0.5)
        w = (lam * gw2 + n_u * per ** 4) / (lam + n_u)
        W2[int(uid)] = w / w.sum()
    sv2 = np.einsum("ij,ij->i", np.stack([V[k] for k in sub], axis=1),
                    np.stack([W2[int(x)] for x in uv]))
    Wt2 = np.stack([W2.get(int(x), gw2) for x in ut])
    st2 = np.einsum("ij,ij->i", np.stack([T[k] for k in sub], axis=1), Wt2)
    record("S5B_USERMIX3", _auc(yv, sv2), _uauc(yv, uv, sv2),
           _auc(yt, st2), _uauc(yt, ut, st2),
           "per-user over complementary subset (no RR1)")

    # ---- S5C: user-mixture over any additional captured models ----
    # Reads extra valid+test NPZ pairs from the capture directory when both
    # exist; runs the same shrunk per-user mixture over each augmentation.
    extra_pool = {}
    for cand in ("HP11_FINE_B03", "HP15_FINE_T05", "HP27_B07_P50",
                 "MC_ITEMROUTE", "L2_DUALGATE", "L5_DUALGATE_T05"):
        vp = EXP / "out" / "CORA_HP_TEST_LOGITS_20261115" / \
            f"{cand}_valid" / "layer_logits_valid.npz"
        tp = EXP / "out" / "CORA_HP_TEST_LOGITS_20261115" / \
            f"{cand}_test" / "layer_logits_test.npz"
        if vp.is_file() and tp.is_file():
            with np.load(vp, allow_pickle=False) as d1:
                cv4 = (d1["fused_scores"].astype(np.float64)
                       if "fused_scores" in d1.files else d1["yes_logits"][3])
                good_v = (np.array_equal(d1["labels"], yv)
                          and np.array_equal(d1["users"], uv))
            with np.load(tp, allow_pickle=False) as d2:
                ct4 = (d2["fused_scores"].astype(np.float64)
                       if "fused_scores" in d2.files else d2["yes_logits"][3])
                good_t = (np.array_equal(d2["labels"], yt)
                          and np.array_equal(d2["users"], ut))
            if good_v and good_t:
                extra_pool[cand] = (_z(cv4), _z(ct4))
    score_v = dict(V)
    score_t = dict(T)
    for cand, (cv, ct) in extra_pool.items():
        score_v[cand] = cv
        score_t[cand] = ct

    def user_mix_over(member_names):
        gw_m = np.array([_auc(yv, score_v[k]) for k in member_names])
        gw_m = gw_m / gw_m.sum()
        Wm = {}
        for uid in users_all:
            mask = uv == uid
            n_u = int(mask.sum())
            per = np.array([_auc(yv[mask], score_v[k][mask])
                            for k in member_names])
            per = np.nan_to_num(per, nan=0.5)
            w = (lam * gw_m + n_u * per ** 4) / (lam + n_u)
            Wm[int(uid)] = w / w.sum()
        sv_m = np.einsum("ij,ij->i",
                         np.stack([score_v[k] for k in member_names], axis=1),
                         np.stack([Wm[int(x)] for x in uv]))
        Wt_m = np.stack([Wm.get(int(x), gw_m) for x in ut])
        st_m = np.einsum("ij,ij->i",
                         np.stack([score_t[k] for k in member_names], axis=1),
                         Wt_m)
        return _auc(yv, sv_m), _uauc(yv, uv, sv_m), _auc(yt, st_m), \
            _uauc(yt, ut, st_m)

    for cand in extra_pool:
        va, vu_, ta, tu = user_mix_over(["RR5", "HP12", "HP24", cand])
        record(f"S5C_{cand}", va, vu_, ta, tu,
               f"user-mix over RR5+HP12+HP24+{cand}")
    if len(extra_pool) >= 2:
        all_names = ["RR5", "HP12", "HP24"] + list(extra_pool)
        va, vu_, ta, tu = user_mix_over(all_names)
        record("S5C_ALL", va, vu_, ta, tu,
               f"user-mix over all {len(all_names)} captured")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "cpu_studio.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "cpu_studio.json").write_text(
        json.dumps(dict(results=rows, diversity=div), ensure_ascii=False,
                   indent=2, default=str), encoding="utf-8")
    print(f"[CPU STUDIO SAVED] {OUT_DIR}")


if __name__ == "__main__":
    main()
