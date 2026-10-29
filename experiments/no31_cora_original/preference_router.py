"""Preference-aware CoRA residual routers (R1-R4), NO graph propagation.

A small trainable MLP predicts bounded *multipliers* on the existing
Group8 CoRA-P residual. The original CoRA generator and frozen MF input
are unchanged; no new loss, no GNN or community features are introduced.

The initial multiplier is EXACTLY one in all four modes. Inputs use
ONLY existing frozen MF vectors and target-excluded positive user history
already present in a sample; no validation/test labels are referenced.
"""
import torch
from torch import nn
from torch.nn import functional as F


class PreferenceResidualRouter(nn.Module):
    MODES = ("global", "depth8", "projection2", "depth_projection32")
    PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")

    def __init__(self, mode, embedding_size=256, epsilon=0.1):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown preference router mode: {mode}")
        if embedding_size != 256:
            raise ValueError("Preference router requires MF dimension 256")
        if not 0 < epsilon <= 0.25:
            raise ValueError("Preference route epsilon must be in (0,.25]")
        self.mode = mode
        self.epsilon = float(epsilon)
        width = {"global": 1, "depth8": 8, "projection2": 2,
                 "depth_projection32": 32}[mode]
        self.head = nn.Sequential(
            nn.Linear(embedding_size * 2 + 3, 64),
            nn.GELU(),
            nn.Linear(64, width),
        )
        # No change to CoRA residual for ANY input at initialization.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self._positive_grad_events = 0
        self._backward_calls = 0
        self.head[-1].weight.register_hook(self._audit)

    def _audit(self, grad):
        tick = self._backward_calls
        self._backward_calls += 1
        # Audit first few events only. Never suppress, clamp or detach a grad.
        # GradScaler continues to do actual nonfinite checks at every step.
        if tick < 8:
            finite = bool(torch.isfinite(grad).all())
            norm = grad.detach().float().norm().item() if finite else float("nan")
            if finite and norm > 0:
                self._positive_grad_events += 1
            print(f"[N29-ROUTER-GRAD] mode={self.mode} backward={tick} "
                  f"norm={norm:.9g} finite={finite}", flush=True)
        return grad

    def preference_features(self, users, targets, history_ids, mf_user, mf_item,
                            padding_index):
        """[B, 515]: normalized user/item MF, UI cosine, history cosine, coverage.

        history_ids is a padded per-user item-history tensor from the existing
        recommendation sample. The CURRENT target is excluded even if it
        appears in the history. All item vectors are FROZEN snapshots.
        """
        if users.ndim != 1 or targets.ndim != 1 or users.shape != targets.shape:
            raise ValueError("Router expects flattened user/target ids")
        u = mf_user[users].float()
        i = mf_item[targets].float()
        uf = F.normalize(u, dim=-1, eps=1e-8)
        iv = F.normalize(i, dim=-1, eps=1e-8)
        similarity = (uf * iv).sum(dim=-1, keepdim=True)
        history_score = torch.zeros_like(similarity)
        history_fraction = torch.zeros_like(similarity)
        if history_ids is not None:
            hist = history_ids.to(device=targets.device, dtype=torch.long)
            if hist.ndim != 2 or hist.shape[0] != targets.shape[0]:
                raise ValueError("Router expects BxH padded item histories")
            if hist.shape[1] > 0:
                valid = ((hist >= 0) & (hist < mf_item.shape[0])
                         & (hist != int(padding_index))
                         & (hist != targets.unsqueeze(1)))
                safe = hist.clamp(min=0, max=mf_item.shape[0]-1)
                hist_vec = F.normalize(mf_item[safe].float(), dim=-1, eps=1e-8)
                cos = (hist_vec * iv.unsqueeze(1)).sum(-1)
                denom = valid.sum(1, keepdim=True).clamp_min(1)
                history_score = (cos * valid.to(cos.dtype)).sum(1, keepdim=True) / denom
                history_fraction = valid.float().mean(1, keepdim=True)
        return torch.cat((uf, iv, similarity, history_score, history_fraction), dim=-1)

    def forward(self, users, targets, history_ids, mf_user, mf_item, padding_index):
        x = self.preference_features(
            users, targets, history_ids, mf_user, mf_item, padding_index)
        raw = self.head(x)
        b = users.shape[0]
        if self.mode == "global":
            logits = raw.view(b, 1, 1).expand(b, 8, 4)
        elif self.mode == "depth8":
            logits = raw.view(b, 8, 1).expand(b, 8, 4)
        elif self.mode == "projection2":
            # Q/K = source-selection; V/O = content/output. Two learnable
            # dimensions; projectors in both categories share a multiplier.
            pair = raw.view(b, 1, 2)
            logits = torch.stack(
                (pair[..., 0], pair[..., 0], pair[..., 1], pair[..., 1]),
                dim=-1).expand(b, 8, 4)
        else:
            logits = raw.view(b, 8, 4)
        return 1.0 + self.epsilon * torch.tanh(logits)

    @torch.no_grad()
    def summary(self):
        last = self.head[-1]
        return dict(
            mode=self.mode, epsilon=self.epsilon,
            output_width=int(last.out_features),
            final_weight_norm=float(last.weight.detach().float().norm()),
            final_bias_norm=float(last.bias.detach().float().norm()),
            positive_gradient_events=int(self._positive_grad_events),
            backward_calls=int(self._backward_calls),
        )
