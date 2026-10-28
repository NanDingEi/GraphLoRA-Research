"""NIGHT23 graph coarse/fine feature adapter, preserving NIGHT20 shapes.

Graph CF users/items and graph clusters are TRAIN-derived, frozen for the
LLM stage.  Original MF continues to supply NO31 CTR/KD and input prompts.
"""
import numpy as np
import torch
from experiments.no31_cora_original.night20_features import build_history_tables


def install_graph_history(model, artifact, train_path):
    with np.load(artifact, allow_pickle=False) as f:
        required = ("user_graph", "item_graph", "item_to_cluster", "prototypes", "item_residual")
        if not all(x in f for x in required):
            raise KeyError(f"Missing graph feature keys: {required}")
        users = f["user_graph"].astype(np.float32)
        items = f["item_graph"].astype(np.float32)
        groups = f["item_to_cluster"].astype(np.int64)
        prototypes = f["prototypes"].astype(np.float32)
        residual = f["item_residual"].astype(np.float32)
    nu, d = model.cora_fixed_user.shape
    ni, di = model.cora_fixed_item.shape
    if d != 256 or di != d or users.shape != (nu, d) or items.shape != (ni, d):
        raise ValueError("Graph artifact not aligned with MF IDs and 256-d adapter")
    if (groups.shape != (ni,) or prototypes.shape != (16, d)
            or residual.shape != (ni, d) or groups[0] != -1
            or np.any(groups[1:] < 0) or np.any(groups[1:] >= 16)):
        raise ValueError("Graph K16 community artifact invalid")
    if not all(np.isfinite(x).all() for x in (users, items, prototypes, residual)):
        raise ValueError("Nonfinite graph artifacts")
    if not np.allclose(items[1:], prototypes[groups[1:]] + residual[1:],
                       atol=1e-4, rtol=1e-4):
        raise ValueError("Graph prototype decomposition does not reconstruct item")
    # Match feature magnitudes to existing frozen-MF condition.  The graph
    # embedding content is unchanged up to two global scalar factors.
    mf_u = model.cora_fixed_user.detach().float().cpu()
    mf_i = model.cora_fixed_item.detach().float().cpu()
    rms_u = float(mf_u.square().mean().sqrt())
    rms_i = float(mf_i[1:].square().mean().sqrt())
    g_u = float(np.sqrt(np.mean(users ** 2)))
    g_i = float(np.sqrt(np.mean(items[1:] ** 2)))
    if min(g_u, g_i) < 1e-9:
        raise ValueError("Graph embeddings collapsed")
    su, si = rms_u / g_u, rms_i / g_i
    users *= su
    items *= si
    prototypes *= si
    residual *= si
    tables = build_history_tables(train_path, nu, groups, residual)
    device = model.cora_fixed_user.device
    def store(name, values, dtype):
        model.register_buffer("n20_" + name,
                              torch.as_tensor(values, dtype=dtype, device=device),
                              persistent=False)
    group_safe = groups.copy()
    group_safe[0] = 0
    store("graph_user", users, torch.float32)
    store("graph_item", items, torch.float32)
    store("item_group", group_safe, torch.long)
    store("prototype", prototypes, torch.float32)
    store("item_residual", residual, torch.float32)
    for name, val in tables.items():
        if isinstance(val, np.ndarray):
            store(name, val, torch.long if name == "pair_keys" else torch.float32)
    print(f"[CORA-N23-GRAPH] users={nu} items={ni} K16 "
          f"graph/MF_RMS_user={su:.6g} item={si:.6g} "
          "train_only=True target_pair_leave_out_history=True", flush=True)


def graph_pair(model, users, targets):
    return torch.cat((model.n20_graph_user[users],
                      model.n20_graph_item[targets]), dim=-1)


def make_graph_features(model, users, targets, tau=5.):
    """Return exactly NIGHT20-compatible coarse [B,514] and fine [B,513]."""
    users = users.long().reshape(-1)
    targets = targets.long().reshape(-1)
    group = model.n20_item_group[targets]
    u = model.n20_graph_user[users]
    p = model.n20_prototype[group]
    delta = model.n20_item_residual[targets]
    key = users * model.n20_graph_item.size(0) + targets
    keys = model.n20_pair_keys
    idx = torch.searchsorted(keys, key)
    j = idx.clamp(max=keys.numel()-1)
    found = (idx < keys.numel()) & (keys[j] == key)
    zeros = torch.zeros_like(key, dtype=torch.float32)
    remove_total = torch.where(found, model.n20_pair_total[j], zeros)
    remove_pos = torch.where(found, model.n20_pair_pos[j], zeros)
    total = (model.n20_user_total[users, group] - remove_total).clamp_min(0)
    pos = (model.n20_user_pos[users, group] - remove_pos).clamp_min(0)
    prior = model.n20_group_prior[group]
    prob = ((pos + tau * prior) / (total + tau)).clamp(1e-4, 1 - 1e-4)
    coarse_score = (u * p).sum(dim=-1, keepdim=True)
    coarse = torch.cat((u, p, coarse_score, torch.logit(prob).unsqueeze(-1)), -1)
    history_sum = model.n20_user_pos_residual_sum[users, group]
    history_sum = history_sum - remove_pos.unsqueeze(-1) * delta
    history = torch.where(pos.unsqueeze(-1) > 0,
                          history_sum / pos.clamp_min(1).unsqueeze(-1),
                          torch.zeros_like(delta))
    fine_score = (u * delta).sum(dim=-1, keepdim=True)
    fine = torch.cat((delta, history, fine_score), -1)
    if coarse.shape[1] != 514 or fine.shape[1] != 513:
        raise RuntimeError("Graph feature dimensions invalid")
    if not torch.isfinite(coarse).all() or not torch.isfinite(fine).all():
        raise FloatingPointError("Graph CF feature nonfinite")
    return coarse, fine
