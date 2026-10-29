"""Trainable LightGCN + hierarchical graph conditions for NIGHT24.

No hidden-state injection.  Graph representations are recomputed with autograd
each CoRA conditioning forward.  Cluster assignments stay fixed (discrete);
prototypes, item residuals, and positive-history residual pools are live.

Original MF input prompts, CTR/KD and other losses remain independent.
"""
from pathlib import Path
import math
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
                 layers=2, k=16, aggregation="uniform", fusion="fixed",
                 initial_lambda=0.1, user_lambda=None, item_lambda=None,
                 sparse_audit_interval=1):
        super().__init__()
        if layers not in (1, 2, 3) or k != 16:
            raise ValueError("N24 requires 1-3 LightGCN layers and K16")
        if aggregation not in ("uniform", "learned", "shallow_fixed", "shallow_learned"):
            raise ValueError("N27 graph aggregation mode unsupported")
        if aggregation.startswith("shallow") and layers != 3:
            raise ValueError("Shallow-prior weights require 3 LightGCN propagation steps")
        if fusion not in ("fixed", "fixed_pair", "dual", "degree"):
            raise ValueError("Graph fusion must be fixed/fixed_pair/dual/degree")
        if fusion == "fixed_pair":
            if user_lambda is None or item_lambda is None:
                raise ValueError("fixed_pair requires user_lambda and item_lambda")
            if not (0 <= user_lambda <= 0.5 and 0 <= item_lambda <= 0.5):
                raise ValueError("fixed_pair coefficients out of range [0, .5]")
        if not 0 <= initial_lambda <= 1.0:
            raise ValueError("Graph anchor lambda must be within [0,1]")
        if fusion in ("dual", "degree") and not (0 < initial_lambda < 0.5):
            raise ValueError("Trainable graph gates require initial lambda within (0,.5)")
        self.aggregation = aggregation
        self.fusion = fusion
        self.max_fusion = 0.5
        self.initial_lambda = float(initial_lambda)
        self.fixed_user_lambda = float(user_lambda) if user_lambda is not None else self.initial_lambda
        self.fixed_item_lambda = float(item_lambda) if item_lambda is not None else self.initial_lambda
        self.sparse_audit_interval = int(sparse_audit_interval)
        if self.sparse_audit_interval < 1:
            raise ValueError("sparse_audit_interval must be >= 1")
        self._n27_gate_grad_events = 0
        # Fixed lambda may legally be zero (N26 MF-only control). Only
        # learnable gates need the inverse sigmoid and nonzero initial lambda.
        initial_logit = (math.log(initial_lambda / (self.max_fusion - initial_lambda))
                         if fusion in ("dual", "degree") else 0.0)
        if fusion == "dual":
            self.dual_gate_logits = nn.Parameter(torch.full((2,), initial_logit))
            self.dual_gate_logits.register_hook(self._audit_gate_gradient)
        elif fusion == "degree":
            self.degree_gate_intercepts = nn.Parameter(torch.full((2,), initial_logit))
            self.degree_gate_slopes = nn.Parameter(torch.zeros(2))
            self.degree_gate_intercepts.register_hook(self._audit_gate_gradient)
            self.degree_gate_slopes.register_hook(self._audit_gate_gradient)
        if aggregation in ("learned", "shallow_learned"):
            # Preserve historical N26 learned-softmax=uniform behavior.
            prior = ([0.25] * 4 if aggregation == "learned"
                     else [0.4, 0.3, 0.2, 0.1])
            self.layer_logits = nn.Parameter(torch.tensor(
                [math.log(value) for value in prior], dtype=torch.float32
            ))
            self._n26_layer_grad_events = 0
            self.layer_logits.register_hook(self._audit_layer_gradient)
        elif aggregation == "shallow_fixed":
            self.register_buffer("fixed_layer_weights",
                                 torch.tensor([0.4, 0.3, 0.2, 0.1]),
                                 persistent=False)
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
        if fusion == "degree":
            # Count UNIQUE positive training-graph edges, not validation
            # interactions or repeated record rows.
            degrees_u = np.bincount(edge_keys // n_items, minlength=n_users)
            degrees_i = np.bincount(edge_keys % n_items, minlength=n_items)
            u_log = np.log1p(degrees_u.astype(np.float32))
            i_log = np.log1p(degrees_i.astype(np.float32))
            u_normal = u_log / max(float(u_log.max()), 1.)
            i_normal = i_log / max(float(i_log.max()), 1.)
            self.register_buffer("user_degree_log", torch.from_numpy(u_normal),
                                 persistent=False)
            self.register_buffer("item_degree_log", torch.from_numpy(i_normal),
                                 persistent=False)
        self.register_buffer("positive_history_incidence", hist, persistent=False)
        self.register_buffer("group_id", torch.from_numpy(safe_group), persistent=False)
        self.register_buffer("item_group_counts",
                             torch.from_numpy(np.bincount(groups[1:], minlength=k).astype(np.float32)),
                             persistent=False)
        self.id_embedding = nn.Parameter(torch.from_numpy(raw).clone())
        self._grad_audit_calls = 0
        self._n25_positive_grad_events = 0
        self._n24_bad_gradient_events = 0
        self._n24_amp_skip_total = 0
        self._n24_amp_skip_streak = 0
        self._n24_amp_success_total = 0
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
        # The artifact contains STANDARD UNIFORM aggregation at 3 hops.
        # Verify it independently for all modes (including biased weighting),
        # without mislabeling fixed/shallow as the original uniform artifact.
        with torch.no_grad():
            h = self.id_embedding.float()
            propagated = h
            for _ in range(self.layers):
                h = torch.sparse.mm(self.adjacency.float(), h)
                propagated = propagated + h
            propagated = propagated / (self.layers + 1)
            ref_user = propagated[:n_users] * self.user_scale
            ref_item = propagated[n_users:] * self.item_scale
            if not torch.allclose(ref_user, torch.from_numpy(graph_user) * self.user_scale,
                                  rtol=3e-3, atol=3e-4):
                raise RuntimeError("Graph ID table mismatch with UNIFORM pretraining user representation")
            if not torch.allclose(ref_item[1:], torch.from_numpy(graph_item[1:]) * self.item_scale,
                                  rtol=3e-3, atol=3e-4):
                raise RuntimeError("Graph ID table mismatch with UNIFORM pretraining item representation")
        print(f"[N24-GRAPH-JOINT] trainable_id_table={n}x256 layers={layers} "
              f"aggregation={self.aggregation} fusion={self.fusion} "
              f"init_lambda={self.initial_lambda:g} "
              f"user_lambda={self.fixed_user_lambda:g} item_lambda={self.fixed_item_lambda:g} "
              f"train_edges={len(edge_keys)} K={k} "
              f"history_edges={int(positive.sum())} model_lr_group=graph "
              "graph_grad_from_LLM=True", flush=True)

    def _audit_gradient(self, grad):
        # Sparse audit only controls diagnostic GPU synchronizations; gradient
        # is returned unchanged. GradScaler still checks every optimizer step.
        # interval=1 preserves old behavior in legacy experiments.
        if self.sparse_audit_interval > 1:
            tick = self._grad_audit_calls
            if tick >= 8 and tick % self.sparse_audit_interval != 0:
                self._grad_audit_calls += 1
                return grad
        # GradScaler.scale(loss).backward() invokes hooks BEFORE unscale_ and
        # the automatic nonfinite check in GradScaler.step().  Raising here
        # wrongly aborts on a recoverable AMP overflow, instead of letting
        # GradScaler skip the bad optimizer update and lower its scale.
        # NEVER replace nonfinite gradients with zeros: that would conceal
        # the overflow from GradScaler's found_inf logic.
        finite = bool(torch.isfinite(grad).all())
        norm = grad.detach().float().norm().item()
        positive = finite and norm > 0.0
        if positive and self._n25_positive_grad_events < 3:
            self._n25_positive_grad_events += 1
            print(f"[N25-GRAPH-RECEIVES-LLM-GRAD] backward={self._grad_audit_calls} "
                  f"norm={norm:.8g} positive_event={self._n25_positive_grad_events}", flush=True)
        if self._grad_audit_calls < 3 or (positive and self._n25_positive_grad_events <= 3
                                             and self._grad_audit_calls < 8):
            print(f"[N24-GRAPH-GRAD] step={self._grad_audit_calls} "
                  f"norm={norm:.8g} finite={finite}", flush=True)
        if not finite:
            self._n24_bad_gradient_events += 1
            if self._n24_bad_gradient_events <= 10 or self._n24_bad_gradient_events % 20 == 0:
                n_nan = int(torch.isnan(grad).sum().item())
                n_inf = int(torch.isinf(grad).sum().item())
                print(f"[N24-GRAPH-GRAD-NONFINITE] "
                      f"backward={self._grad_audit_calls} "
                      f"event={self._n24_bad_gradient_events} "
                      f"nan={n_nan} inf={n_inf} "
                      "allowing AMP GradScaler to skip this update", flush=True)
        self._grad_audit_calls += 1
        return grad


    def _audit_layer_gradient(self, grad):
        # As for the graph ID embedding, never intercept AMP's GradScaler
        # overflow handling. A nonfinite gradient must reach GradScaler.
        if not bool(torch.isfinite(grad).all()):
            print("[N26-LAYER-LOGIT-NONFINITE] detected; leaving gradient "
                  "untouched for GradScaler", flush=True)
            return grad
        if self._n26_layer_grad_events < 3 and grad.detach().abs().max().item() > 0:
            self._n26_layer_grad_events += 1
            print(
                f"[N26-LAYER-LOGIT-GRAD] event={self._n26_layer_grad_events} "
                f"grad_norm={grad.detach().float().norm().item():.8g} "
                f"grad={grad.detach().float().cpu().tolist()}",
                flush=True,
            )
        return grad

    def _audit_gate_gradient(self, grad):
        if bool(torch.isfinite(grad).all()) and grad.detach().abs().max().item() > 0:
            if self._n27_gate_grad_events < 3:
                self._n27_gate_grad_events += 1
                print(f"[N27-GATE-GRAD] mode={self.fusion} "
                      f"norm={grad.detach().float().norm().item():.8g} "
                      f"event={self._n27_gate_grad_events}", flush=True)
        elif not torch.isfinite(grad).all():
            print("[N27-GATE-NONFINITE] handing gradient to AMP GradScaler",
                  flush=True)
        return grad

    def fusion_weights(self, users, targets):
        """Return Bx1 user and item lambda, max 0.5, differentiable."""
        batch = users.shape[0]
        if self.fusion == "fixed":
            v = self.id_embedding.new_full((batch, 1), self.initial_lambda)
            return v, v
        if self.fusion == "fixed_pair":
            return (self.id_embedding.new_full((batch, 1), self.fixed_user_lambda),
                    self.id_embedding.new_full((batch, 1), self.fixed_item_lambda))
        if self.fusion == "dual":
            coeff = self.max_fusion * torch.sigmoid(self.dual_gate_logits.float())
            return (coeff[0].expand(batch, 1), coeff[1].expand(batch, 1))
        ud = self.user_degree_log[users]
        idg = self.item_degree_log[targets]
        u = self.max_fusion * torch.sigmoid(
            self.degree_gate_intercepts[0] + self.degree_gate_slopes[0] * ud
        )
        i = self.max_fusion * torch.sigmoid(
            self.degree_gate_intercepts[1] + self.degree_gate_slopes[1] * idg
        )
        return u.unsqueeze(1), i.unsqueeze(1)

    @torch.no_grad()
    def fusion_summary(self):
        if self.fusion == "fixed":
            return dict(mode="fixed", user_lambda=self.initial_lambda,
                        item_lambda=self.initial_lambda)
        if self.fusion == "fixed_pair":
            return dict(mode="fixed_pair", user_lambda=self.fixed_user_lambda,
                        item_lambda=self.fixed_item_lambda)
        if self.fusion == "dual":
            coeff = (self.max_fusion * torch.sigmoid(self.dual_gate_logits)).cpu().tolist()
            return dict(mode="dual", user_lambda=coeff[0], item_lambda=coeff[1],
                        gradient_events=self._n27_gate_grad_events)
        return dict(
            mode="degree", user_lambda_at_degree0=float(
                self.max_fusion * torch.sigmoid(self.degree_gate_intercepts[0])),
            item_lambda_at_degree0=float(
                self.max_fusion * torch.sigmoid(self.degree_gate_intercepts[1])),
            user_slope=float(self.degree_gate_slopes[0]),
            item_slope=float(self.degree_gate_slopes[1]),
            gradient_events=self._n27_gate_grad_events,
        )

    @torch.no_grad()
    def layer_weight_summary(self):
        if self.aggregation in ("learned", "shallow_learned"):
            return torch.softmax(self.layer_logits.float(), dim=0).cpu().tolist()
        if self.aggregation == "shallow_fixed":
            return self.fixed_layer_weights.cpu().tolist()
        return [1.0 / (self.layers + 1)] * (self.layers + 1)

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
            if self.aggregation == "uniform":
                result = h
                for _ in range(self.layers):
                    h = torch.sparse.mm(adj, h)
                    result = result + h
                result = result / (self.layers + 1)
            else:
                # Every propagated level contributes to the LLM loss.
                # No detaches and no rounding to fp16 before softmax.
                states = [h]
                for _ in range(self.layers):
                    h = torch.sparse.mm(adj, h)
                    states.append(h)
                if self.aggregation == "shallow_fixed":
                    weights = self.fixed_layer_weights.float()
                else:
                    weights = torch.softmax(self.layer_logits.float(), dim=0)
                result = sum(w * state for w, state in zip(weights, states))
            user = result[:self.num_users] * self.user_scale
            item = result[self.num_users:] * self.item_scale
            # item ID 0 is padding and must remain exactly zero.
            item = torch.cat([torch.zeros_like(item[:1]), item[1:]], dim=0)
        return user, item

    def forward(self, users, targets, global_counts, labels=None, tau=5.,
                use_conditions=True, compute_aux=True):
        users, targets = users.long().reshape(-1), targets.long().reshape(-1)
        if (users.numel() == 0 or users.numel() != targets.numel()
                or (users < 0).any() or (users >= self.num_users).any()
                or (targets <= 0).any() or (targets >= self.num_items).any()):
            raise ValueError("N24 invalid users/targets")
        user, item = self._propagate()
        if not use_conditions:
            # Exact graph_base branch: only selected trainable graph node
            # features are consumed by CoRA. Do not compute unused community
            # prototypes, history sparse.mm or graph BCE when weight=0.
            # This preserves full-graph LightGCN propagation + LLM gradients.
            pair = torch.cat([user[users], item[targets]], dim=-1)
            if compute_aux and labels is not None:
                aux = F.binary_cross_entropy_with_logits(
                    (user[users] * item[targets]).sum(-1),
                    labels.float().view(-1).to(pair.device))
            else:
                aux = None
            return pair, None, aux
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
        if labels is not None and compute_aux:
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
