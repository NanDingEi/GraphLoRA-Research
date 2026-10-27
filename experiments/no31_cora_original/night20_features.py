"""NIGHT20 frozen MF collaborative-prototype features, leakage-safe.

The prototype file was built from TRAIN positive interactions only. This module
precomputes per-(user,community) observed and positive counts and positive-item
residual sums using TRAIN ONLY. For every target (u,i), any train records of
the exact same (u,i) pair are removed from these features, even in evaluation,
to prevent target-label inclusion while preserving one consistent path.

No validation/test labels or LLM checkpoints are read here.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_prototypes(proto_path, frozen_item):
    path = Path(proto_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"NIGHT20 requires K16 prototype NPZ: {path}")
    with np.load(path, allow_pickle=False) as loaded:
        required = {"item_to_cluster", "prototypes", "mf_residual"}
        if not required.issubset(set(loaded.files)):
            raise KeyError(f"Missing prototype file keys {required - set(loaded.files)}")
        group = loaded["item_to_cluster"].astype(np.int64)
        proto = loaded["prototypes"].astype(np.float32)
        residual = loaded["mf_residual"].astype(np.float32)
    n_items, dim = frozen_item.shape
    if group.shape != (n_items,) or residual.shape != (n_items, dim):
        raise ValueError(
            f"Prototype shapes incompatible with frozen MF items: "
            f"group={group.shape} residual={residual.shape}, item={(n_items, dim)}"
        )
    if proto.shape != (16, dim):
        raise ValueError(f"NIGHT20 requires K=16, MF dim={dim}; found {proto.shape}")
    if group[0] != -1 or (group[1:] < 0).any() or (group[1:] >= 16).any():
        raise ValueError("Padding item 0 must be -1; all other items must be K16")
    mf_items = frozen_item.detach().float().cpu().numpy()
    error = np.max(np.abs(proto[group[1:]] + residual[1:] - mf_items[1:]))
    if not np.isfinite(error) or error > 1e-4:
        raise RuntimeError(
            f"Prototype file does not reconstruct current pretrained MF item table "
            f"(max_abs_err={error:.6g}); stop instead of training invalid conditions"
        )
    if not (np.isfinite(proto).all() and np.isfinite(residual).all()):
        raise ValueError("Non-finite prototype vectors")
    return group, proto, residual, float(error)


def build_history_tables(train_path, user_count, group, residual):
    """Return numpy tables for leave-target-out user/group preference.

    Keys are sorted (uid * n_items + iid); duplicate observed train records
    (if any) are aggregated so ALL observations of current target are removed.
    """
    path = Path(train_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"NIGHT20 requires train-only interactions: {path}")
    frame = pd.read_pickle(path)
    if not {"uid", "iid", "label"}.issubset(frame.columns):
        raise KeyError("Training file must contain uid/iid/label")
    data = frame[["uid", "iid", "label"]]
    if data.isnull().values.any():
        raise ValueError("Training interactions contain missing values")
    u = data["uid"].to_numpy(dtype=np.int64)
    i = data["iid"].to_numpy(dtype=np.int64)
    y = data["label"].to_numpy(dtype=np.int64)
    n_items, dim = residual.shape
    if (np.any(u < 0) or np.any(u >= user_count)
            or np.any(i <= 0) or np.any(i >= n_items)
            or not np.isin(y, (0, 1)).all()):
        raise ValueError("Invalid train user/item IDs, padding or binary labels")
    if len(u) == 0:
        raise ValueError("Empty training split")
    k = group[i]
    n_clusters = 16
    total = np.zeros((user_count, n_clusters), dtype=np.float32)
    pos = np.zeros((user_count, n_clusters), dtype=np.float32)
    pos_sum = np.zeros((user_count, n_clusters, dim), dtype=np.float32)
    np.add.at(total, (u, k), 1)
    pos_ix = y == 1
    np.add.at(pos, (u[pos_ix], k[pos_ix]), 1)
    np.add.at(pos_sum, (u[pos_ix], k[pos_ix]), residual[i[pos_ix]])
    cluster_total = np.bincount(k, minlength=n_clusters).astype(np.float32)
    cluster_pos = np.bincount(k[pos_ix], minlength=n_clusters).astype(np.float32)
    # Laplace-smoothed global prior, computed from TRAIN only.
    prior = (cluster_pos + 1.) / (cluster_total + 2.)
    key = u * n_items + i
    keys, inverse = np.unique(key, return_inverse=True)
    pair_total = np.bincount(inverse).astype(np.float32)
    pair_pos = np.bincount(inverse, weights=y).astype(np.float32)
    return dict(
        user_total=total,
        user_pos=pos,
        user_pos_residual_sum=pos_sum,
        group_prior=prior,
        pair_keys=keys.astype(np.int64),
        pair_total=pair_total,
        pair_pos=pair_pos,
        observed_train_rows=len(u),
        unique_train_pairs=len(keys),
    )


def install_frozen_history(model, proto_path, train_path):
    """Register compact nonpersistent frozen tables, moving with the model."""
    item_table = model.cora_fixed_item
    user_table = model.cora_fixed_user
    group, proto, residual, reconstruction_error = load_prototypes(
        proto_path, item_table,
    )
    tables = build_history_tables(
        train_path, user_table.shape[0], group, residual,
    )
    device = item_table.device
    import torch
    def add(name, data, dtype=None):
        arr = torch.as_tensor(data, dtype=dtype, device=device)
        model.register_buffer("n20_" + name, arr, persistent=False)
    # Padding maps to 0 for safe indexing, but is always forbidden as target.
    group_safe = group.copy()
    group_safe[0] = 0
    add("item_group", group_safe, torch.long)
    add("prototype", proto, torch.float32)
    add("item_residual", residual, torch.float32)
    for name, val in tables.items():
        if isinstance(val, np.ndarray):
            add(name, val, torch.long if name == "pair_keys" else torch.float32)
    print(
        f"[CORA-N20-DATA] K=16 num_users={user_table.shape[0]} "
        f"num_items={item_table.shape[0]} frozen_MF_maxerr={reconstruction_error:.3g} "
        f"train_rows={tables['observed_train_rows']} "
        f"unique_train_pairs={tables['unique_train_pairs']} "
        "train_only=True target_pair_leave_out=True",
        flush=True,
    )


def make_collaborative_features(model, users, targets, tau=5.0):
    """Return coarse [B,514], fine [B,513] features with no label leakage."""
    users = users.to(dtype=torch.long).reshape(-1)
    targets = targets.to(dtype=torch.long).reshape(-1)
    if users.numel() != targets.numel() or not users.numel():
        raise ValueError("Invalid user/target batch shapes")
    if (torch.any(targets <= 0) or
            torch.any(targets >= model.cora_fixed_item.shape[0])):
        raise ValueError("Padding/out-of-range target supplied to NIGHT20")
    group = model.n20_item_group[targets]
    user = model.cora_fixed_user[users]
    proto = model.n20_prototype[group]
    item_delta = model.n20_item_residual[targets]
    # Find ALL TRAIN entries for current (user, target) and remove them.
    key = users * model.cora_fixed_item.shape[0] + targets
    keys = model.n20_pair_keys
    idx = torch.searchsorted(keys, key)
    safe_idx = idx.clamp(max=keys.numel()-1)
    found = (idx < keys.numel()) & (keys[safe_idx] == key)
    remove_total = torch.where(
        found, model.n20_pair_total[safe_idx],
        torch.zeros_like(key, dtype=torch.float32),
    )
    remove_pos = torch.where(
        found, model.n20_pair_pos[safe_idx],
        torch.zeros_like(key, dtype=torch.float32),
    )
    total = (model.n20_user_total[users, group] - remove_total).clamp_min(0)
    pos = (model.n20_user_pos[users, group] - remove_pos).clamp_min(0)
    prior = model.n20_group_prior[group]
    p = ((pos + tau * prior) / (total + tau)).clamp(1e-4, 1.-1e-4)
    prior_logit = torch.logit(p)
    coarse_score = (user * proto).sum(dim=-1, keepdim=True)
    coarse = torch.cat(
        (user, proto, coarse_score, prior_logit.unsqueeze(-1)), dim=-1,
    )
    # User-conditioned within-cluster positive history, excluding target.
    sum_history = model.n20_user_pos_residual_sum[users, group]
    sum_history = sum_history - remove_pos.unsqueeze(-1) * item_delta
    mean_history = sum_history / pos.clamp_min(1).unsqueeze(-1)
    # Exactly zero if no positive items remain in the user's cluster.
    mean_history = torch.where(
        pos.unsqueeze(-1) > 0, mean_history, torch.zeros_like(mean_history),
    )
    fine_score = (user * item_delta).sum(dim=-1, keepdim=True)
    fine = torch.cat((item_delta, mean_history, fine_score), dim=-1)
    if coarse.shape[-1] != 514 or fine.shape[-1] != 513:
        raise RuntimeError(f"Unexpected NIGHT20 feature sizes: {coarse.shape}, {fine.shape}")
    if not torch.isfinite(coarse).all() or not torch.isfinite(fine).all():
        raise FloatingPointError("Nonfinite NIGHT20 coarse/fine features")
    return coarse, fine
