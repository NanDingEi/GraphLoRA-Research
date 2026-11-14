#!/usr/bin/env python3
"""Post-hoc late-layer residual gate: fit on VALID, apply once to TEST.

Inputs:
  - VALID NPZ  out/LATE_LAYER_DIAG_D4_C6_FIX_20261102_165625/layer_logits_valid.npz
    (same model: RR1 deterministically reproduced D4_TOPK_RES025)
  - TEST NPZ   experiments/no31_cora_original/out/CORA_RR1_TEST_LOGITS_20261112/
               layer_logits_test.npz  (captured by cora_rr1_test_logits.py)
  - RR1 checkpoint for the trained MF user table (user-gate features).

All fitting/selection uses VALID only; TEST is scored once per method for
transparency. Official score mode is yes_raw_logit (verified in diagnostics).

Methods:
  M0_BASE        s = z32                          (control)
  M1_ALPHA31     s = z32 + a*(z31 - z32), a from a grid selected on Valid
                 (reported for both AUC-selected and AUC+UAUC-selected a)
  M2_LOGIT_ZR    logistic on [z32, r31] fit on Valid
  M3_GATE_USER   logistic on [z32, r31, user-MF mean/std stats] fit on Valid
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise SystemExit("Run inside the minigpt4 conda env: " + str(exc))

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments" / "no31_cora_original"
VALID_NPZ = ROOT / "out" / "LATE_LAYER_DIAG_D4_C6_FIX_20261102_165625" / "layer_logits_valid.npz"
TEST_NPZ = EXP / "out" / "CORA_RR1_TEST_LOGITS_20261112" / "layer_logits_test.npz"
CHECKPOINT = (EXP / "output" / "CORA_RESIDUAL_ROUTING6_20261103" /
              "RR1_L32_BASE" / "20261103092" / "checkpoint_best.pth")
OUT_DIR = EXP / "out" / "CORA_POSTHOC_GATE_TEST_20261112"


def _auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _uauc(labels, users, scores):
    users = np.asarray(users)
    values = []
    for uid in np.unique(users):
        mask = users == uid
        y = labels[mask]
        if np.any(y == 1) and np.any(y == 0):
            values.append(_auc(y, scores[mask]))
    return float(np.mean(values)) if values else float("nan")


def _load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        for key in ("labels", "users", "items", "yes_logits"):
            if key not in data.files:
                raise RuntimeError(f"{path} missing key {key}")
        labels = np.asarray(data["labels"], dtype=np.int64)
        users = np.asarray(data["users"], dtype=np.int64)
        items = np.asarray(data["items"], dtype=np.int64)
        scores = np.asarray(data["yes_logits"], dtype=np.float64)
    if scores.shape != (4, len(labels)):
        raise RuntimeError(f"{path}: expected yes_logits (4,N), got {scores.shape}")
    if not np.isfinite(scores).all():
        raise RuntimeError(f"{path}: nonfinite logits")
    return labels, users, items, scores


def _user_features(users, table):
    emb = table[np.asarray(users, dtype=np.int64)]
    mean = np.zeros((len(users), emb.shape[1]))
    std = np.zeros((len(users), emb.shape[1]))
    for uid in np.unique(users):
        mask = users == uid
        mean[mask] = emb[mask].mean(axis=0)
        std[mask] = emb[mask].std(axis=0)
    feats = np.concatenate([mean, std], axis=1)
    return (feats - feats.mean(axis=0)) / np.where(
        feats.std(axis=0) < 1e-8, 1.0, feats.std(axis=0))


def _fit_logit(x_train, y_train, x_apply):
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=0.5, max_iter=4000))
    model.fit(x_train, y_train)
    return model.decision_function(x_apply), model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid-npz", default=str(VALID_NPZ))
    parser.add_argument("--test-npz", default=str(TEST_NPZ))
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    y_v, u_v, i_v, z_v = _load_npz(Path(args.valid_npz))
    y_t, u_t, i_t, z_t = _load_npz(Path(args.test_npz))
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    table = payload["model"]["rec_encoder.user_embedding.weight"].numpy()

    base_valid = dict(auc=_auc(y_v, z_v[3]), uauc=_uauc(y_v, u_v, z_v[3]))
    base_test = dict(auc=_auc(y_t, z_t[3]), uauc=_uauc(y_t, u_t, z_t[3]))
    print(f"[BASE z32] valid AUC={base_valid['auc']:.6f} UAUC={base_valid['uauc']:.6f} "
          f"| test AUC={base_test['auc']:.6f} UAUC={base_test['uauc']:.6f}")

    rows = []
    results = {}

    def record(method, detail, v_auc, v_uauc, t_auc, t_uauc):
        rows.append(dict(method=method, detail=detail,
                         valid_auc=f"{v_auc:.9f}", valid_uauc=f"{v_uauc:.9f}",
                         test_auc=f"{t_auc:.9f}", test_uauc=f"{t_uauc:.9f}",
                         delta_test_auc=f"{t_auc - base_test['auc']:+.9f}",
                         delta_test_uauc=f"{t_uauc - base_test['uauc']:+.9f}"))
        results[method] = dict(detail=detail, valid_auc=v_auc, valid_uauc=v_uauc,
                               test_auc=t_auc, test_uauc=t_uauc)
        print(f"[{method}] {detail}\n  valid AUC={v_auc:.6f} UAUC={v_uauc:.6f} "
              f"-> test AUC={t_auc:.6f} UAUC={t_uauc:.6f} "
              f"(dAUC={t_auc - base_test['auc']:+.6f} "
              f"dUAUC={t_uauc - base_test['uauc']:+.6f})")

    record("M0_BASE", "z32", base_valid["auc"], base_valid["uauc"],
           base_test["auc"], base_test["uauc"])

    # ---- M1: alpha grid on z31 residual ----
    r_v = z_v[2] - z_v[3]
    r_t = z_t[2] - z_t[3]
    grid = np.round(np.arange(-0.5, 2.01, 0.05), 3)
    stats = []
    for a in grid:
        s = z_v[3] + a * r_v
        stats.append((_auc(y_v, s), _uauc(y_v, u_v, s), float(a)))
    auc_sel = max(stats, key=lambda t: (t[0], t[1]))
    comp_sel = max(stats, key=lambda t: (t[0] + 0.25 * t[1], t[0]))
    uauc_sel = max(stats, key=lambda t: (t[1], t[0]))
    for tag, sel in (("M1_ALPHA31_AUC", auc_sel), ("M1_ALPHA31_COMP", comp_sel),
                     ("M1_ALPHA31_UAUC", uauc_sel)):
        a = sel[2]
        s_t = z_t[3] + a * r_t
        record(tag, f"alpha={a:.2f} (valid AUC={sel[0]:.6f} UAUC={sel[1]:.6f})",
               sel[0], sel[1], _auc(y_t, s_t), _uauc(y_t, u_t, s_t))

    # ---- M2: logistic on [z32, r31] ----
    x_v = np.column_stack([z_v[3], r_v])
    x_t = np.column_stack([z_t[3], r_t])
    pred_v, _ = _fit_logit(x_v, y_v, x_v)
    pred_t, _ = _fit_logit(x_v, y_v, x_t)
    record("M2_LOGIT_ZR", "logistic([z32, r31]) C=0.5 fit on valid",
           _auc(y_v, pred_v), _uauc(y_v, u_v, pred_v),
           _auc(y_t, pred_t), _uauc(y_t, u_t, pred_t))

    # ---- M3: + user MF features ----
    f_v = _user_features(u_v, table)
    f_t = _user_features(u_t, table)
    x_v3 = np.column_stack([z_v[3], r_v, f_v])
    x_t3 = np.column_stack([z_t[3], r_t, f_t])
    pred_v3, _ = _fit_logit(x_v3, y_v, x_v3)
    pred_t3, _ = _fit_logit(x_v3, y_v, x_t3)
    record("M3_GATE_USER", "logistic([z32, r31, userMF-mean/std]) fit on valid",
           _auc(y_v, pred_v3), _uauc(y_v, u_v, pred_v3),
           _auc(y_t, pred_t3), _uauc(y_t, u_t, pred_t3))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "posthoc_gate_test.csv").open("w", newline="",
                                                  encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "posthoc_gate_test.json").write_text(
        json.dumps(dict(base_valid=base_valid, base_test=base_test,
                        results=results,
                        selection_rule="All fitting on VALID only; "
                                       "TEST scored once per method"),
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[POSTHOC SAVED] {out_dir}")


if __name__ == "__main__":
    main()
