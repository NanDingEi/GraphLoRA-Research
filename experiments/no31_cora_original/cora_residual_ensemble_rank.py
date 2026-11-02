#!/usr/bin/env python3
"""User-grouped out-of-fold evaluation of late-layer residual ensembles.

This is post-processing only: it loads the captured Valid logits NPZ, does not
load the LLM, does not run CoRA training, and does not evaluate Test.
The only fitted parameters are small calibration/gating heads, fitted strictly
inside each training fold. All four layers from a user remain in one fold.
"""
import argparse
import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import minimize
    from scipy.special import expit, softmax
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise SystemExit(
        "Missing analysis dependencies. In the minigpt4 environment run: "
        "python -m pip install scikit-learn scipy"
    ) from exc

ROOT = Path(__file__).resolve().parent
LAYER_NAMES = ("L29", "L30", "L31", "L32")
METHODS = (
    "L1_L32_BASE",
    "L2_GLOBAL_R31_RESIDUAL",
    "L3_USER_GATE_R31",
    "L4_USER_GATE_MULTI",
    "L5_PAIRWISE_USER_R31",
    "L6_BCE_PAIRWISE_R31",
    "L7_USER_SCALAR_GATE_R31",
)


def _sigmoid(x):
    return expit(np.clip(x, -30.0, 30.0))


def _logloss(logits, labels):
    logits = np.clip(np.asarray(logits, dtype=np.float64), -40.0, 40.0)
    labels = np.asarray(labels, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))


def _auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) != 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _uauc(labels, scores, users):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    users = np.asarray(users)
    values = []
    for uid in np.unique(users):
        mask = users == uid
        if len(np.unique(labels[mask])) == 2:
            values.append(_auc(labels[mask], scores[mask]))
    return (float(np.mean(values)) if values else float("nan"), len(values))


def _brier(labels, scores):
    p = _sigmoid(scores)
    return float(np.mean((p - np.asarray(labels, dtype=np.float64)) ** 2))


def _find_input(explicit):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if path.is_dir():
            path = path / "layer_logits_valid.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = list((ROOT / "out").glob(
        "LATE_LAYER_DIAG_D4_C6*/layer_logits_valid.npz"
    ))
    if not candidates:
        raise FileNotFoundError(
            "No late-layer NPZ found. Expected out/LATE_LAYER_DIAG_D4_C6*/"
            "layer_logits_valid.npz; pass --input explicitly if needed."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_data(path):
    diagnostics_path = path.parent / "diagnostics.json"
    if not diagnostics_path.is_file():
        raise FileNotFoundError(
            f"Missing {diagnostics_path}; use the NPZ produced together with diagnostics.json."
        )
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    if diagnostics.get("official_score_mode_match") != "yes_raw_logit":
        raise RuntimeError(
            "The official scoring mode is not confirmed as yes_raw_logit. "
            "Do not run residual experiments until score semantics are resolved."
        )
    expected_layers = [29, 30, 31, 32]
    if diagnostics.get("layers_one_based") != expected_layers:
        raise RuntimeError(
            f"Expected layers {expected_layers}, got {diagnostics.get('layers_one_based')}"
        )
    with np.load(path, allow_pickle=False) as data:
        required = ("labels", "users", "items", "yes_logits")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise RuntimeError(f"NPZ missing keys: {missing}")
        labels = np.asarray(data["labels"], dtype=np.int64)
        users = np.asarray(data["users"], dtype=np.int64)
        items = np.asarray(data["items"], dtype=np.int64)
        scores = np.asarray(data["yes_logits"], dtype=np.float64)
    if scores.shape != (4, len(labels)):
        raise RuntimeError(f"Expected yes_logits shape (4,{len(labels)}), got {scores.shape}")
    if users.shape != labels.shape or items.shape != labels.shape:
        raise RuntimeError("labels/users/items length mismatch")
    if not np.isin(labels, [0, 1]).all():
        raise RuntimeError("Labels must be binary 0/1")
    if not np.isfinite(scores).all():
        raise RuntimeError("Found non-finite layer logits")
    official_auc = float(diagnostics["official_aggregate_auc"])
    layer32_auc = _auc(labels, scores[3])
    delta = abs(layer32_auc - official_auc)
    if delta > 0.005:
        raise RuntimeError(
            f"L32 Yes-logit AUC {layer32_auc:.8f} differs from official "
            f"{official_auc:.8f} by {delta:.6f}; score semantics mismatch."
        )
    return labels, users, items, scores, diagnostics, official_auc


def _residualize(train_idx, test_idx, layer_idx, scores):
    base_train = scores[3, train_idx]
    layer_train = scores[layer_idx, train_idx]
    base_mean = float(base_train.mean())
    layer_mean = float(layer_train.mean())
    denom = float(np.sum((base_train - base_mean) ** 2))
    coef = (
        float(np.sum((base_train - base_mean) * (layer_train - layer_mean)) / denom)
        if denom > 1e-12 else 0.0
    )
    intercept = layer_mean - coef * base_mean
    residual_train = layer_train - (coef * base_train + intercept)
    residual_test = (
        scores[layer_idx, test_idx]
        - (coef * scores[3, test_idx] + intercept)
    )
    return residual_train, residual_test, {
        "slope": coef,
        "intercept": intercept,
    }


def _user_mean_std(values, users):
    values = np.asarray(values, dtype=np.float64)
    users = np.asarray(users)
    mean = np.zeros_like(values)
    std = np.zeros_like(values)
    for uid in np.unique(users):
        mask = users == uid
        mean[mask] = float(values[mask].mean())
        std[mask] = float(values[mask].std())
    return np.column_stack((mean, std))


def _standardize(train_features, test_features):
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (
        (train_features - mean) / std,
        (test_features - mean) / std,
        mean,
        std,
    )


def _fit_l2(train_idx, test_idx, labels, scores):
    r_train, r_test, residual_fit = _residualize(
        train_idx, test_idx, 2, scores
    )
    x_train = np.column_stack((scores[3, train_idx], r_train))
    x_test = np.column_stack((scores[3, test_idx], r_test))
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs"),
    )
    model.fit(x_train, labels[train_idx])
    prediction = model.decision_function(x_test)
    params = {
        "residual_fit_L31_on_L32": residual_fit,
        "standardized_logistic_coef": model[-1].coef_.reshape(-1).tolist(),
        "logistic_intercept": float(model[-1].intercept_[0]),
        "C": 1.0,
    }
    return prediction, params


def _fit_l3(train_idx, test_idx, labels, users, scores, reg=0.05):
    """L32 + L31 residual with a sample- and user-context-conditioned gate."""
    r_train, r_test, residual_fit = _residualize(
        train_idx, test_idx, 2, scores
    )
    ctx_train = _user_mean_std(r_train, users[train_idx])
    ctx_test = _user_mean_std(r_test, users[test_idx])
    ctx_base_train = _user_mean_std(scores[3, train_idx], users[train_idx])
    ctx_base_test = _user_mean_std(scores[3, test_idx], users[test_idx])

    f_train = np.column_stack((
        scores[3, train_idx], np.abs(r_train), ctx_train, ctx_base_train
    ))
    f_test = np.column_stack((
        scores[3, test_idx], np.abs(r_test), ctx_test, ctx_base_test
    ))
    f_train, f_test, f_mean, f_std = _standardize(f_train, f_test)
    x_train = np.column_stack((np.ones(len(train_idx)), f_train))
    x_test = np.column_stack((np.ones(len(test_idx)), f_test))
    d = x_train.shape[1]

    # Parameters: gate weights, log temperature, and bias.
    initial = np.zeros(d + 2, dtype=np.float64)
    initial[0] = -1.5  # initial residual strength ~= 0.18

    def objective(params):
        gate_w = params[:d]
        log_scale = params[d]
        bias = params[d + 1]
        scale = np.exp(log_scale)
        gate = _sigmoid(x_train @ gate_w)
        final_score = scale * (
            scores[3, train_idx] + gate * r_train
        ) + bias
        penalty = reg * (
            np.sum(gate_w[1:] ** 2) + log_scale ** 2
        )
        return _logloss(final_score, labels[train_idx]) + float(penalty)

    bounds = [(None, None)] * d + [(-2.0, 2.0), (None, None)]
    result = minimize(
        objective, initial, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-9},
    )
    gate_w = result.x[:d]
    scale = float(np.exp(result.x[d]))
    bias = float(result.x[d + 1])
    gate_test = _sigmoid(x_test @ gate_w)
    prediction = scale * (
        scores[3, test_idx] + gate_test * r_test
    ) + bias
    params = {
        "residual_fit_L31_on_L32": residual_fit,
        "gate_weights": gate_w.tolist(),
        "feature_mean": f_mean.tolist(),
        "feature_std": f_std.tolist(),
        "scale": scale,
        "bias": bias,
        "regularization": reg,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "iterations": int(result.nit),
        "test_gate_mean": float(gate_test.mean()),
        "test_gate_std": float(gate_test.std()),
    }
    return prediction, params


def _fit_l4(train_idx, test_idx, labels, users, scores, reg=0.10):
    """L32 + user-context softmax mixture of L29/L30/L31 residuals."""
    residual_train = []
    residual_test = []
    residual_fits = []
    for layer_idx in (0, 1, 2):
        rtr, rte, fit = _residualize(
            train_idx, test_idx, layer_idx, scores
        )
        residual_train.append(rtr)
        residual_test.append(rte)
        residual_fits.append(fit)
    residual_train = np.column_stack(residual_train)
    residual_test = np.column_stack(residual_test)

    context_train = []
    context_test = []
    for j in range(3):
        c_train = _user_mean_std(
            residual_train[:, j], users[train_idx]
        )
        c_test = _user_mean_std(
            residual_test[:, j], users[test_idx]
        )
        context_train.append(c_train)
        context_test.append(c_test)
    context_train = np.column_stack(context_train)
    context_test = np.column_stack(context_test)

    f_train = np.column_stack((
        scores[3, train_idx],
        np.abs(residual_train),
        context_train,
    ))
    f_test = np.column_stack((
        scores[3, test_idx],
        np.abs(residual_test),
        context_test,
    ))
    f_train, f_test, f_mean, f_std = _standardize(f_train, f_test)
    x_train = np.column_stack((np.ones(len(train_idx)), f_train))
    x_test = np.column_stack((np.ones(len(test_idx)), f_test))
    d = x_train.shape[1]

    # Softmax selects among three residual channels; a separate sigmoid
    # controls total residual strength, allowing the model to ignore all.
    initial = np.zeros(3 * d + d + 2, dtype=np.float64)
    initial[3 * d] = -1.5
    initial_w = initial[:3 * d].reshape(3, d)

    def objective(params):
        gate_w = params[:3 * d].reshape(3, d)
        strength_w = params[3 * d:4 * d]
        log_scale = params[-2]
        bias = params[-1]
        scale = np.exp(log_scale)
        weights = softmax(x_train @ gate_w.T, axis=1)
        strength = _sigmoid(x_train @ strength_w)
        residual = np.sum(weights * residual_train, axis=1) * strength
        final_score = scale * (
            scores[3, train_idx] + residual
        ) + bias
        penalty = reg * (
            np.sum(gate_w[:, 1:] ** 2)
            + np.sum(strength_w[1:] ** 2)
            + log_scale ** 2
        )
        return _logloss(final_score, labels[train_idx]) + float(penalty)

    bounds = (
        [(None, None)] * (4 * d)
        + [(-2.0, 2.0), (None, None)]
    )
    result = minimize(
        objective, initial, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 700, "ftol": 1e-9},
    )
    gate_w = result.x[:3 * d].reshape(3, d)
    strength_w = result.x[3 * d:4 * d]
    scale = float(np.exp(result.x[-2]))
    bias = float(result.x[-1])
    weights_test = softmax(x_test @ gate_w.T, axis=1)
    strength_test = _sigmoid(x_test @ strength_w)
    residual_test_mix = (
        np.sum(weights_test * residual_test, axis=1) * strength_test
    )
    prediction = scale * (
        scores[3, test_idx] + residual_test_mix
    ) + bias
    params = {
        "residual_fits_L29_L30_L31_on_L32": residual_fits,
        "mixture_gate_weights": gate_w.tolist(),
        "residual_strength_weights": strength_w.tolist(),
        "feature_mean": f_mean.tolist(),
        "feature_std": f_std.tolist(),
        "scale": scale,
        "bias": bias,
        "regularization": reg,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "iterations": int(result.nit),
        "test_mean_layer_weights": weights_test.mean(axis=0).tolist(),
        "test_mean_residual_strength": float(strength_test.mean()),
        "test_std_residual_strength": float(strength_test.std()),
    }
    return prediction, params



def _pairwise_pairs(labels, users):
    """Build all within-user positive/negative pairs with equal user weight."""
    labels = np.asarray(labels, dtype=np.int64)
    users = np.asarray(users)
    positive_indices = []
    negative_indices = []
    pair_weights = []
    valid_user_count = 0
    for uid in np.unique(users):
        local = np.flatnonzero(users == uid)
        pos = local[labels[local] == 1]
        neg = local[labels[local] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        valid_user_count += 1
        p = np.repeat(pos, len(neg))
        n = np.tile(neg, len(pos))
        positive_indices.append(p)
        negative_indices.append(n)
        pair_weights.append(
            np.full(len(p), 1.0 / (len(pos) * len(neg)), dtype=np.float64)
        )
    if valid_user_count == 0:
        raise RuntimeError("No training user has both positive and negative labels.")
    p_idx = np.concatenate(positive_indices)
    n_idx = np.concatenate(negative_indices)
    weights = np.concatenate(pair_weights) / float(valid_user_count)
    return p_idx, n_idx, weights, valid_user_count


def _build_l3_features(train_idx, test_idx, users, scores):
    """Use exactly the L3 feature recipe for pairwise-objective comparisons."""
    r_train, r_test, residual_fit = _residualize(
        train_idx, test_idx, 2, scores
    )
    ctx_train = _user_mean_std(r_train, users[train_idx])
    ctx_test = _user_mean_std(r_test, users[test_idx])
    ctx_base_train = _user_mean_std(scores[3, train_idx], users[train_idx])
    ctx_base_test = _user_mean_std(scores[3, test_idx], users[test_idx])
    f_train = np.column_stack((
        scores[3, train_idx], np.abs(r_train), ctx_train, ctx_base_train
    ))
    f_test = np.column_stack((
        scores[3, test_idx], np.abs(r_test), ctx_test, ctx_base_test
    ))
    f_train, f_test, f_mean, f_std = _standardize(f_train, f_test)
    x_train = np.column_stack((np.ones(len(train_idx)), f_train))
    x_test = np.column_stack((np.ones(len(test_idx)), f_test))
    return r_train, r_test, x_train, x_test, residual_fit, f_mean, f_std


def _fit_l5_pairwise(train_idx, test_idx, labels, users, scores, reg=0.05):
    """L3 gate trained only with equal-user-weighted within-user pairwise loss."""
    r_train, r_test, x_train, x_test, residual_fit, f_mean, f_std = (
        _build_l3_features(train_idx, test_idx, users, scores)
    )
    y_train = labels[train_idx]
    p_idx, n_idx, pair_weights, pair_users = _pairwise_pairs(
        y_train, users[train_idx]
    )
    d = x_train.shape[1]
    initial = np.zeros(d, dtype=np.float64)
    initial[0] = -1.5

    def objective(gate_w):
        gate = _sigmoid(x_train @ gate_w)
        final_score = scores[3, train_idx] + gate * r_train
        pair_loss = np.sum(
            pair_weights * np.logaddexp(
                0.0, -(final_score[p_idx] - final_score[n_idx])
            )
        )
        penalty = reg * np.sum(gate_w[1:] ** 2)
        return float(pair_loss + penalty)

    result = minimize(
        objective, initial, method="L-BFGS-B",
        options={"maxiter": 600, "ftol": 1e-9},
    )
    gate_w = result.x
    gate_test = _sigmoid(x_test @ gate_w)
    prediction = scores[3, test_idx] + gate_test * r_test
    params = {
        "residual_fit_L31_on_L32": residual_fit,
        "gate_weights": gate_w.tolist(),
        "feature_mean": f_mean.tolist(),
        "feature_std": f_std.tolist(),
        "regularization": reg,
        "pairwise_users": int(pair_users),
        "pair_count": int(len(p_idx)),
        "train_gate_mean": float(_sigmoid(x_train @ gate_w).mean()),
        "test_gate_mean": float(gate_test.mean()),
        "test_gate_std": float(gate_test.std()),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "iterations": int(result.nit),
        "objective": float(result.fun),
    }
    return prediction, params


def _fit_l6_bce_pairwise(
    train_idx, test_idx, labels, users, scores, reg=0.05, lambda_pair=0.20
):
    """L3 gate trained with BCE plus equal-user-weighted pairwise ranking loss."""
    r_train, r_test, x_train, x_test, residual_fit, f_mean, f_std = (
        _build_l3_features(train_idx, test_idx, users, scores)
    )
    y_train = labels[train_idx]
    p_idx, n_idx, pair_weights, pair_users = _pairwise_pairs(
        y_train, users[train_idx]
    )
    d = x_train.shape[1]
    initial = np.zeros(d + 2, dtype=np.float64)
    initial[0] = -1.5

    def objective(params):
        gate_w = params[:d]
        log_scale = params[d]
        bias = params[d + 1]
        scale = np.exp(log_scale)
        gate = _sigmoid(x_train @ gate_w)
        final_score = scale * (
            scores[3, train_idx] + gate * r_train
        ) + bias
        bce = _logloss(final_score, y_train)
        pair_loss = np.sum(
            pair_weights * np.logaddexp(
                0.0, -(final_score[p_idx] - final_score[n_idx])
            )
        )
        penalty = reg * (
            np.sum(gate_w[1:] ** 2) + log_scale ** 2
        )
        return float(bce + lambda_pair * pair_loss + penalty)

    bounds = [(None, None)] * d + [(-2.0, 2.0), (None, None)]
    result = minimize(
        objective, initial, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 700, "ftol": 1e-9},
    )
    gate_w = result.x[:d]
    scale = float(np.exp(result.x[d]))
    bias = float(result.x[d + 1])
    gate_test = _sigmoid(x_test @ gate_w)
    prediction = scale * (
        scores[3, test_idx] + gate_test * r_test
    ) + bias
    params = {
        "residual_fit_L31_on_L32": residual_fit,
        "gate_weights": gate_w.tolist(),
        "feature_mean": f_mean.tolist(),
        "feature_std": f_std.tolist(),
        "scale": scale,
        "bias": bias,
        "regularization": reg,
        "lambda_pair": lambda_pair,
        "pairwise_users": int(pair_users),
        "pair_count": int(len(p_idx)),
        "train_gate_mean": float(_sigmoid(x_train @ gate_w).mean()),
        "test_gate_mean": float(gate_test.mean()),
        "test_gate_std": float(gate_test.std()),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "iterations": int(result.nit),
        "objective": float(result.fun),
    }
    return prediction, params


def _user_level_features(residual, base_score, users):
    """Unlabeled user-level statistics; one identical feature vector per user."""
    users = np.asarray(users)
    residual = np.asarray(residual, dtype=np.float64)
    base_score = np.asarray(base_score, dtype=np.float64)
    unique_users = np.unique(users)
    feature_by_user = {}
    user_feature_rows = []
    for uid in unique_users:
        mask = users == uid
        r = residual[mask]
        b = base_score[mask]
        feature = np.asarray([
            r.mean(),
            r.std(),
            np.abs(r).mean(),
            np.abs(r).std(),
            b.mean(),
            b.std(),
            np.abs(b).mean(),
        ], dtype=np.float64)
        feature_by_user[uid] = feature
        user_feature_rows.append(feature)
    sample_features = np.vstack([feature_by_user[uid] for uid in users])
    user_features = np.vstack(user_feature_rows)
    return sample_features, user_features, unique_users


def _fit_l7_user_scalar_gate(
    train_idx, test_idx, labels, users, scores, reg=0.05
):
    """L3-style BCE fit, but one residual gate per user from unlabeled aggregates."""
    r_train, r_test, residual_fit = _residualize(
        train_idx, test_idx, 2, scores
    )
    base_train = scores[3, train_idx]
    base_test = scores[3, test_idx]
    f_train, uf_train, train_user_ids = _user_level_features(
        r_train, base_train, users[train_idx]
    )
    f_test, uf_test, test_user_ids = _user_level_features(
        r_test, base_test, users[test_idx]
    )
    mean = uf_train.mean(axis=0)
    std = uf_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    f_train = (f_train - mean) / std
    f_test = (f_test - mean) / std
    x_train = np.column_stack((np.ones(len(train_idx)), f_train))
    x_test = np.column_stack((np.ones(len(test_idx)), f_test))
    d = x_train.shape[1]
    initial = np.zeros(d + 2, dtype=np.float64)
    initial[0] = -1.5

    def objective(params):
        gate_w = params[:d]
        log_scale = params[d]
        bias = params[d + 1]
        scale = np.exp(log_scale)
        gate = _sigmoid(x_train @ gate_w)
        final_score = scale * (base_train + gate * r_train) + bias
        penalty = reg * (
            np.sum(gate_w[1:] ** 2) + log_scale ** 2
        )
        return _logloss(final_score, labels[train_idx]) + float(penalty)

    bounds = [(None, None)] * d + [(-2.0, 2.0), (None, None)]
    result = minimize(
        objective, initial, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 600, "ftol": 1e-9},
    )
    gate_w = result.x[:d]
    scale = float(np.exp(result.x[d]))
    bias = float(result.x[d + 1])
    gate_train = _sigmoid(x_train @ gate_w)
    gate_test = _sigmoid(x_test @ gate_w)
    prediction = scale * (base_test + gate_test * r_test) + bias
    params = {
        "residual_fit_L31_on_L32": residual_fit,
        "gate_weights": gate_w.tolist(),
        "user_feature_names": [
            "residual_mean", "residual_std", "residual_abs_mean",
            "residual_abs_std", "base_mean", "base_std", "base_abs_mean"
        ],
        "user_feature_mean": mean.tolist(),
        "user_feature_std": std.tolist(),
        "train_users": int(len(train_user_ids)),
        "test_users": int(len(test_user_ids)),
        "scale": scale,
        "bias": bias,
        "regularization": reg,
        "train_gate_mean": float(gate_train.mean()),
        "test_gate_mean": float(gate_test.mean()),
        "test_gate_std": float(gate_test.std()),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "iterations": int(result.nit),
        "objective": float(result.fun),
    }
    return prediction, params

def _evaluate_fold(labels, users, scores):
    auc = _auc(labels, scores)
    uauc, n_users = _uauc(labels, scores, users)
    return {
        "auc": auc,
        "uauc": uauc,
        "uauc_users": int(n_users),
        "brier": _brier(labels, scores),
        "accuracy_at_zero": float(np.mean((scores >= 0) == labels)),
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Four late-layer residual ensemble experiments (Valid-only OOF)."
    )
    parser.add_argument("--input", default=None,
                        help="Path to layer_logits_valid.npz or its containing directory.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    if args.folds < 3:
        raise ValueError("--folds must be >= 3")
    random.seed(args.seed)
    np.random.seed(args.seed)

    input_path = _find_input(args.input)
    labels, users, items, scores, diagnostics, official_auc = _load_data(input_path)
    n = len(labels)
    unique_users = np.unique(users)
    if len(unique_users) < args.folds:
        raise RuntimeError("Number of unique users is smaller than number of folds.")

    print(f"[RESIDUAL INPUT] {input_path}", flush=True)
    print(
        f"[RESIDUAL DATA] n={n} users={len(unique_users)} "
        f"positive={(labels == 1).sum()} negative={(labels == 0).sum()}",
        flush=True,
    )
    print(
        f"[RESIDUAL CHECK] L32_AUC={_auc(labels, scores[3]):.9f} "
        f"official_AUC={official_auc:.9f} "
        f"checkpoint={diagnostics.get('checkpoint')}",
        flush=True,
    )

    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    splits = list(splitter.split(np.zeros(n), labels, groups=users))
    test_count = np.zeros(n, dtype=np.int64)
    for train_idx, test_idx in splits:
        if set(np.unique(users[train_idx])) & set(np.unique(users[test_idx])):
            raise RuntimeError("User leakage detected between train and test fold.")
        test_count[test_idx] += 1
    if not np.all(test_count == 1):
        raise RuntimeError("OOF split does not cover each sample exactly once.")

    oof_scores = {
        name: np.full(n, np.nan, dtype=np.float64) for name in METHODS
    }
    fold_rows = []
    fold_parameters = []
    for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
        print(
            f"[RESIDUAL FOLD START] {fold_id}/{args.folds} "
            f"train={len(train_idx)} test={len(test_idx)} "
            f"train_users={len(np.unique(users[train_idx]))} "
            f"test_users={len(np.unique(users[test_idx]))}",
            flush=True,
        )
        fold_predictions = {
            "L1_L32_BASE": scores[3, test_idx].copy()
        }
        p2, params2 = _fit_l2(train_idx, test_idx, labels, scores)
        p3, params3 = _fit_l3(train_idx, test_idx, labels, users, scores)
        p4, params4 = _fit_l4(train_idx, test_idx, labels, users, scores)
        p5, params5 = _fit_l5_pairwise(train_idx, test_idx, labels, users, scores)
        p6, params6 = _fit_l6_bce_pairwise(train_idx, test_idx, labels, users, scores)
        p7, params7 = _fit_l7_user_scalar_gate(train_idx, test_idx, labels, users, scores)
        fold_predictions["L2_GLOBAL_R31_RESIDUAL"] = p2
        fold_predictions["L3_USER_GATE_R31"] = p3
        fold_predictions["L4_USER_GATE_MULTI"] = p4
        fold_predictions["L5_PAIRWISE_USER_R31"] = p5
        fold_predictions["L6_BCE_PAIRWISE_R31"] = p6
        fold_predictions["L7_USER_SCALAR_GATE_R31"] = p7

        row_params = {
            "fold": fold_id,
            "train_users": int(len(np.unique(users[train_idx]))),
            "test_users": int(len(np.unique(users[test_idx]))),
            "L2": params2,
            "L3": params3,
            "L4": params4,
            "L5": params5,
            "L6": params6,
            "L7": params7,
        }
        fold_parameters.append(row_params)
        for method, prediction in fold_predictions.items():
            if not np.isfinite(prediction).all():
                raise RuntimeError(f"Non-finite OOF predictions: {method}, fold={fold_id}")
            oof_scores[method][test_idx] = prediction
            metrics = _evaluate_fold(
                labels[test_idx], users[test_idx], prediction
            )
            fold_rows.append({
                "fold": fold_id,
                "method": method,
                "train_samples": int(len(train_idx)),
                "test_samples": int(len(test_idx)),
                **metrics,
            })
            print(
                f"[RESIDUAL FOLD] {fold_id} {method} "
                f"AUC={metrics['auc']:.6f} UAUC={metrics['uauc']:.6f} "
                f"Brier={metrics['brier']:.6f}",
                flush=True,
            )
        print(
            f"[RESIDUAL FOLD END] {fold_id} "
            f"L3_gate_mean={params3['test_gate_mean']:.4f} "
            f"L4_strength_mean={params4['test_mean_residual_strength']:.4f} "
            f"L4_weights={params4['test_mean_layer_weights']}",
            flush=True,
        )

    if any(not np.isfinite(v).all() for v in oof_scores.values()):
        raise RuntimeError("Some samples have no finite out-of-fold prediction.")

    fold_summary = {}
    for method in METHODS:
        rows = [r for r in fold_rows if r["method"] == method]
        auc_values = np.asarray([r["auc"] for r in rows], dtype=np.float64)
        uauc_values = np.asarray([r["uauc"] for r in rows], dtype=np.float64)
        fold_summary[method] = {
            "fold_auc_mean": float(np.mean(auc_values)),
            "fold_auc_std": float(np.std(auc_values)),
            "fold_uauc_mean": float(np.mean(uauc_values)),
            "fold_uauc_std": float(np.std(uauc_values)),
            "oof_pooled_auc": _auc(labels, oof_scores[method]),
            "oof_user_macro_uauc": _uauc(
                labels, oof_scores[method], users
            )[0],
            "oof_users_with_both_classes": _uauc(
                labels, oof_scores[method], users
            )[1],
            "oof_brier": _brier(labels, oof_scores[method]),
            "oof_accuracy_at_zero": float(
                np.mean((oof_scores[method] >= 0) == labels)
            ),
        }

    baseline = fold_summary["L1_L32_BASE"]
    summary_rows = []
    for method in METHODS:
        m = fold_summary[method]
        summary_rows.append({
            "method": method,
            "fold_auc_mean": m["fold_auc_mean"],
            "fold_auc_std": m["fold_auc_std"],
            "oof_pooled_auc": m["oof_pooled_auc"],
            "delta_fold_auc_vs_L1": m["fold_auc_mean"] - baseline["fold_auc_mean"],
            "fold_uauc_mean": m["fold_uauc_mean"],
            "fold_uauc_std": m["fold_uauc_std"],
            "oof_user_macro_uauc": m["oof_user_macro_uauc"],
            "delta_oof_uauc_vs_L1": (
                m["oof_user_macro_uauc"] - baseline["oof_user_macro_uauc"]
            ),
            "oof_brier": m["oof_brier"],
            "oof_accuracy_at_zero": m["oof_accuracy_at_zero"],
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag or f"CORA_USER_RANKING_ENSEMBLE_{timestamp}"
    out_dir = ROOT / "out" / tag
    out_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "tag": tag,
        "timestamp": timestamp,
        "input_npz": str(input_path),
        "source_checkpoint": diagnostics.get("checkpoint"),
        "source_checkpoint_epoch": diagnostics.get("checkpoint_epoch"),
        "protocol": diagnostics.get("protocol"),
        "official_score_mode": diagnostics.get("official_score_mode_match"),
        "official_valid_auc": official_auc,
        "n_samples": int(n),
        "n_users": int(len(unique_users)),
        "positive_count": int((labels == 1).sum()),
        "negative_count": int((labels == 0).sum()),
        "folds": int(args.folds),
        "splitter": "StratifiedGroupKFold(shuffle=True)",
        "seed": int(args.seed),
        "no_user_overlap_between_train_and_test": True,
        "test_evaluation": False,
        "base_model_training": False,
        "fitted_components": (
            "Fold-local linear residualization, logistic residual calibrators, "
            "pairwise ranking gates, and user-level scalar gates only."
        ),
        "methods": fold_summary,
        "notes": [
            "Primary comparisons use mean fold AUC and user-macro OOF UAUC.",
            "OOF pooled AUC is also reported; fold-specific calibrators can shift score scales.",
            "All samples from one user stay in a single held-out fold.",
            "L3/L4/L7 user context uses only that user's unlabeled layer-score statistics.",
            "L5 uses equal-user-weighted within-user positive/negative pairs.",
            "L6 combines BCE with 0.20 times equal-user-weighted pairwise loss.",
            "No labels from held-out users are used to fit residuals or gates.",
            "This is Valid-only exploratory analysis, not Test performance.",
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(out_dir / "summary.csv", summary_rows)
    _write_csv(out_dir / "fold_metrics.csv", fold_rows)
    (out_dir / "fold_parameters.json").write_text(
        json.dumps(_json_safe(fold_parameters), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    prediction_rows = []
    for i in range(n):
        row = {
            "sample_index": i,
            "user_id": int(users[i]),
            "item_id": int(items[i]),
            "label": int(labels[i]),
        }
        for method in METHODS:
            row[f"oof_score_{method}"] = float(oof_scores[method][i])
            row[f"oof_probability_{method}"] = float(_sigmoid(oof_scores[method][i]))
        prediction_rows.append(row)
    _write_csv(out_dir / "oof_predictions.csv", prediction_rows)

    print("\n[RESIDUAL SUMMARY] VALID-ONLY, USER-GROUPED OOF", flush=True)
    print(
        f"{'Method':28s} {'Fold AUC':>10s} {'OOF UAUC':>10s} "
        f"{'dAUC':>9s} {'dUAUC':>9s} {'Brier':>9s}",
        flush=True,
    )
    for row in summary_rows:
        print(
            f"{row['method']:28s} "
            f"{row['fold_auc_mean']:.6f} "
            f"{row['oof_user_macro_uauc']:.6f} "
            f"{row['delta_fold_auc_vs_L1']:+.6f} "
            f"{row['delta_oof_uauc_vs_L1']:+.6f} "
            f"{row['oof_brier']:.6f}",
            flush=True,
        )
    print(f"[RESIDUAL OUTPUT] {out_dir}", flush=True)
    print("[RESIDUAL DONE] no CoRA training, no checkpoint save, no Test evaluation", flush=True)

    global_log = ROOT / "logs" / "CORA_RESIDUAL_ENSEMBLE_SUMMARY.log"
    global_log.parent.mkdir(parents=True, exist_ok=True)
    with global_log.open("a", encoding="utf-8") as fp:
        fp.write(f"\n[{timestamp}] tag={tag} input={input_path}\n")
        fp.write(
            f"{'Method':28s} {'FoldAUC':>10s} {'OOF_UAUC':>10s} "
            f"{'dAUC':>9s} {'dUAUC':>9s} {'Brier':>9s}\n"
        )
        for row in summary_rows:
            fp.write(
                f"{row['method']:28s} "
                f"{row['fold_auc_mean']:.6f} "
                f"{row['oof_user_macro_uauc']:.6f} "
                f"{row['delta_fold_auc_vs_L1']:+.6f} "
                f"{row['delta_oof_uauc_vs_L1']:+.6f} "
                f"{row['oof_brier']:.6f}\n"
            )
        fp.write(f"output={out_dir}\n")
    print(f"[RESIDUAL GLOBAL SUMMARY] {global_log}", flush=True)


if __name__ == "__main__":
    main()
