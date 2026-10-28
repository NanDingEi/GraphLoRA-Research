"""Trainable LightGCN + hierarchical graph conditions for NIGHT24.

No hidden-state injection.  Graph representations are recomputed with autograd
each CoRA conditioning forward.  Cluster assignments stay fixed (discrete);
prototypes, item residuals, and positive-history residual pools are live.

Original MF input prompts, CTR/KD and other losses remain independent.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def _sparse_normalize(u, i, n_users, n_items):
    n = n_users + n_items
    rows = np.concatenate([u, i + n_users])
    cols = np.concatenate([i + n_users, u])
    mat = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                        shape=(n, n)).tocsr()
    mat.data[:] = 1.
    degree = np.asarray(mat.sum(axis=1)).ravel()
    inv = np.zeros_like(degree)
    np.divide(1., np.sqrt(degree), out=inv, where=degree > 0)
    mat = (sp.diags(inv) @ mat @ sp.diags(inv)).tocoo()
    indices = torch.from_numpy(np.stack([mat.row, mat.col], axis=0).astype(np.int64))
    values = torch.from_numpy(mat.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()


def _history_incidence(u, i, groups, n_users, n_items, k):
    # Duplicate positive interaction records count as separate observations,
    # matching existing NIGHT20 positive-history count conventions.
    gi = groups[i]
    incidence = sp.coo_matrix(
        (np.ones(len(u), dtype=np.float32), (u * k + gi, i)),
        shape=(n_users * k, n_items),
    ).tocoo()
    indices = torch.from_numpy(np.stack([incidence.row, incidence.col]).astype(np.int64))
    values = torch.from_numpy(incidence.data.astype(np.float32))
    return torch.sparse_coo_tensor(
        indices, values, (n_users * k, n_items)
    ).coalesce()


class JointLightGCN(nn.Module):
    def __init__(self, artifact, train_path, n_users, n_items, mf_user, mf_item,
                 layers=2, k=16):
        super().__init__()
        if layers not in (1, 2, 3) or k != 16:
            raise ValueError("N24 requires 1-3 LightGCN layers and K16")
        with np.load(str(artifact), allow_pickle=False) as f:
            required = ("graph_id_table", "user_graph", "item_graph", "item_to_cluster")
            if not set(required).issubset(set(f.files)):
                raise KeyError("Joint graph needs N24 graph_id_table artifact; rerun graph pretraining")
            raw = f["graph_id_table"].astype(np.float32)
            graph_user = f["user_graph"].astype(np.float32)
            graph_item = f["item_graph"].astype(np.float32)
            groups = f["item_to_cluster"].astype(np.int64)
        n = n_users + n_items
        if raw.shape != (n, 256) or graph_user.shape != (n_users, 256) or graph_item.shape != (n_items, 256):
            raise ValueError("Graph artifact table/propagation shapes mismatch")
        if (groups.shape != (n_items,) or groups[0] != -1
                or (groups[1:] < 0).any() or (groups[1:] >= k).any()):
            raise ValueError("Invalid graph community assignment")
        if not all(np.isfinite(v).all() for v in (raw, graph_user, graph_item)):
            raise ValueError("Nonfinite graph artifact")
        df = pd.read_pickle(train_path)
        if not {"uid", "iid", "label"}.issubset(df.columns):
            raise ValueError("TRAIN needs uid, iid, label")
        cols = df[["uid", "iid", "label"]]
        if cols.isnull().values.any():
            raise ValueError("Missing TRAIN interaction values")
        u = cols.uid.to_numpy(dtype=np.int64)
        i = cols.iid.to_numpy(dtype=np.int64)
        y = cols.label.to_numpy(dtype=np.int64)
        if (len(u) == 0 or (u < 0).any() or (u >= n_users).any()
                or (i <= 0).any() or (i >= n_items).any()
                or not np.isin(y, [0, 1]).all()):
            raise ValueError("Invalid TRAIN graph nodes or labels")
        positive = y == 1
        edge_keys = np.unique(u[positive] * n_items + i[positive])
        if not len(edge_keys):
            raise ValueError("TRAIN has no positive graph edges")
        adj = _sparse_normalize(edge_keys // n_items, edge_keys % n_items,
                                n_users, n_items)
        hist = _history_incidence(u[positive], i[positive], groups, n_users,
                                  n_items, k)
        safe_group = groups.copy()
        safe_group[0] = 0
        self.register_buffer("adjacency", adj, persistent=False)
        self.register_buffer("positive_history_incidence", hist, persistent=False)
        self.register_buffer("group_id", torch.from_numpy(safe_group), persistent=False)
        self.register_buffer("item_group_counts",
                             torch.from_numpy(np.bincount(groups[1:], minlength=k).astype(np.float32)),
                             persistent=False)
        self.id_embedding = nn.Parameter(torch.from_numpy(raw).clone())
        self._grad_audit_calls = 0
        self.id_embedding.register_hook(self._audit_gradient)
        self.num_users, self.num_items, self.k, self.layers = n_users, n_items, k, layers
        # Same fixed global scaling as NIGHT23 (only up to two scalar factors).
        mf_u = float(mf_user.detach().float().square().mean().sqrt().item())
        mf_i = float(mf_item.detach().float()[1:].square().mean().sqrt().item())
        g_u = float(np.sqrt(np.mean(graph_user ** 2)))
        g_i = float(np.sqrt(np.mean(graph_item[1:] ** 2)))
        if min(g_u, g_i) < 1e-9:
            raise ValueError("Graph propagation collapsed")
        self.user_scale = mf_u / g_u
        self.item_scale = mf_i / g_i
        # Ensure the raw embeddings and propagation depth match the NPZ
        # exported by pretraining; otherwise do not claim checkpoint parity.
        with torch.no_grad():
            uu, ii = self._propagate()
            if not torch.allclose(uu, torch.from_numpy(graph_user) * self.user_scale,
                                  rtol=3e-3, atol=3e-4):
                raise RuntimeError("Graph ID table does not reproduce pretraining user embeddings")
            if not torch.allclose(ii[1:], torch.from_numpy(graph_item[1:]) * self.item_scale,
                                  rtol=3e-3, atol=3e-4):
                raise RuntimeError("Graph ID table does not reproduce pretraining item embeddings")
        print(f"[N24-GRAPH-JOINT] trainable_id_table={n}x256 layers={layers} "
              f"train_edges={len(edge_keys)} K={k} "
              f"history_edges={int(positive.sum())} model_lr_group=graph "
              "graph_grad_from_LLM=True", flush=True)

    def _audit_gradient(self, grad):
        if self._grad_audit_calls < 3:
            print(f"[N24-GRAPH-GRAD] step={self._grad_audit_calls} "
                  f"norm={grad.detach().float().norm().item():.8g} "
                  f"finite={bool(torch.isfinite(grad).all())}", flush=True)
        self._grad_audit_calls += 1
        if not torch.isfinite(grad).all():
            raise FloatingPointError("N24 LightGCN gradient is nonfinite")
        return grad

    def _propagate(self):
        # CUDA sparse.mm does not support the FP16 sparse addmm path on
        # the target PyTorch build.  Disabling autocast explicitly avoids
        # PyTorch's autocast of sparse.mm to Half even when h.float().
        # DO NOT detach: FP32 sparse propagation must remain differentiable
        # all the way to id_embedding, through CoRA and the LLM loss.
        with torch.autocast(device_type=self.id_embedding.device.type,
                            enabled=False):
            adj = self.adjacency.float()
            h = self.id_embedding.float()
            result = h
            for _ in range(self.layers):
                h = torch.sparse.mm(adj, h)
                result = result + h
            result = result / (self.layers + 1)
            user = result[:self.num_users] * self.user_scale
            item = result[self.num_users:] * self.item_scale
            # item ID 0 is padding and must remain exactly zero.
            item = torch.cat([torch.zeros_like(item[:1]), item[1:]], dim=0)
        return user, item

    def forward(self, users, targets, global_counts, labels=None, tau=5.):
        users, targets = users.long().reshape(-1), targets.long().reshape(-1)
        if (users.numel() == 0 or users.numel() != targets.numel()
                or (users < 0).any() or (users >= self.num_users).any()
                or (targets <= 0).any() or (targets >= self.num_items).any()):
            raise ValueError("N24 invalid users/targets")
        user, item = self._propagate()
        # Differentiable community prototypes.  Community assignments are
        # fixed; no attempt to differentiate through hard clustering.
        means = item.new_zeros((self.k, item.shape[1]))
        means.index_add_(0, self.group_id[1:], item[1:])
        means = means / self.item_group_counts.clamp_min(1).unsqueeze(-1)
        residual = item - means[self.group_id]
        group = self.group_id[targets]
        u, i, p, r = user[users], item[targets], means[group], residual[targets]
        pair = torch.cat([u, i], dim=-1)
        # Original NIGHT20 leave-target-out *static* user/community counts.
        key = users * self.num_items + targets
        keys = global_counts["pair_keys"]
        ix = torch.searchsorted(keys, key)
        safe = ix.clamp(max=keys.numel()-1)
        found = (ix < keys.numel()) & (keys[safe] == key)
        zeros = torch.zeros_like(key, dtype=torch.float32)
        remove_total = torch.where(found, global_counts["pair_total"][safe], zeros)
        remove_pos = torch.where(found, global_counts["pair_pos"][safe], zeros)
        total = (global_counts["user_total"][users, group] - remove_total).clamp_min(0)
        pos = (global_counts["user_pos"][users, group] - remove_pos).clamp_min(0)
        prior = global_counts["group_prior"][group]
        prob = ((pos + tau * prior) / (total + tau)).clamp(1e-4, 1.-1e-4)
        coarse = torch.cat([u, p, (u * p).sum(-1, keepdim=True),
                            torch.logit(prob).unsqueeze(-1)], dim=-1)
        # Live differentiable pooled residual; no frozen residual snapshot.
        # The user/community sparse history aggregation has the same CUDA
        # AMP limitation as graph propagation. Keep both sparse operands FP32
        # and retain autograd through residual and graph node parameters.
        with torch.autocast(device_type=residual.device.type, enabled=False):
            hist_sum = torch.sparse.mm(
                self.positive_history_incidence.float(), residual.float()
            )
        hist_sum = hist_sum.view(self.num_users, self.k, item.shape[1])[users, group]
        hist_sum = hist_sum - remove_pos.unsqueeze(-1) * r
        history = torch.where(pos.unsqueeze(-1) > 0,
                              hist_sum / pos.clamp_min(1).unsqueeze(-1),
                              torch.zeros_like(r))
        fine = torch.cat([r, history, (u * r).sum(-1, keepdim=True)], dim=-1)
        auxiliary = None
        if labels is not None:
            label = labels.float().view(-1).to(pair.device)
            if not torch.isin(label, torch.tensor([0., 1.], device=label.device)).all():
                raise ValueError("N24 requires binary train labels for graph BCE")
            auxiliary = F.binary_cross_entropy_with_logits((u * i).sum(-1), label)
        if (pair.shape != (len(users), 512) or coarse.shape != (len(users), 514)
                or fine.shape != (len(users), 513)):
            raise RuntimeError("N24 graph condition shape invalid")
        if not all(torch.isfinite(x).all() for x in (pair, coarse, fine)):
            raise FloatingPointError("Nonfinite joint graph conditions")
        return pair, (coarse, fine), auxiliary
