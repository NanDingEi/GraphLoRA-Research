"""Candidate-aware history evidence in the NGCF penultimate space.

Zero-initialized rank residual for the target-ID token; when enabled with fresh
weights, this branch initially preserves GraphLoRA's forward predictions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def accumulate_target_residuals(delta_rows, target_rows, residual):
    """Accumulate rank residuals into ID-token rows under mixed precision.

    GNN representations and delta_rows are FP32, while AMP may produce FP16
    residuals. index_add requires both operands to have identical dtype.
    Casting the residual to the accumulator dtype preserves gradients and
    performs the accumulation in FP32.
    """
    return delta_rows.index_add(
        0, target_rows, residual.to(dtype=delta_rows.dtype)
    )


def scale_candidate_delta_for_eval(delta, multiplier):
    """Scale A1's target-only residual at inference without changing GNN injection."""
    if delta is None or multiplier == 1.0:
        return delta
    return delta * multiplier


def scale_candidate_gate_for_eval(gate, multiplier):
    """Inference-only gate ablation; multiplier=0 makes the injection identity."""
    if gate is None or multiplier == 1.0:
        return gate
    if multiplier == 0.0:
        return torch.ones_like(gate)
    return 1.0 + multiplier * (gate - 1.0)


class CandidateHistoryFusion(nn.Module):
    def __init__(self, dim=256, rank=8, temperature=0.2):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.encoder = nn.Sequential(
            nn.Linear(4 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, rank),
        )
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(self, user, target, history, history_mask):
        """Return (rank_residual, attention_weights).

        user/target: [B,D]; history: [B,K,D]; history_mask: [B,K].
        Rows with no valid history yield exactly zero residual and weights.
        """
        user = user.float()
        target = target.float()
        history = history.float()
        history_mask = history_mask.bool()
        query = F.normalize(user + target, dim=-1)
        keys = F.normalize(history, dim=-1)
        scores = torch.einsum('bd,bkd->bk', query, keys) / self.temperature
        scores = scores.masked_fill(~history_mask, -1e4)
        weights = torch.softmax(scores, dim=-1) * history_mask.to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        evidence = torch.einsum('bk,bkd->bd', weights, history)
        features = torch.cat([user, target, evidence, target * evidence], dim=-1)
        residual = 0.1 * torch.tanh(self.encoder(features))
        residual = residual * history_mask.any(dim=-1, keepdim=True).to(residual.dtype)
        return residual, weights


class CandidateHistoryGate(nn.Module):
    """Candidate-aware, target-token-only modulation of GraphLoRA GNN injection.

    Returns an r-dimensional gate centered at exactly 1.0. A2 uses the
    original cosine(u+target, history) attention, while A3 learns candidate
    query/history key projections, keeping all other components identical.
    """
    def __init__(self, dim=256, rank=8, temperature=0.2,
                 attention="cosine"):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if attention not in ("cosine", "learned"):
            raise ValueError("attention must be 'cosine' or 'learned'")
        self.temperature = float(temperature)
        self.attention = attention
        if attention == "learned":
            self.query_proj = nn.Linear(dim, dim, bias=False)
            self.key_proj = nn.Linear(dim, dim, bias=False)
            nn.init.eye_(self.query_proj.weight)
            nn.init.eye_(self.key_proj.weight)
        self.gate_encoder = nn.Sequential(
            nn.Linear(4 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, rank),
        )
        nn.init.zeros_(self.gate_encoder[-1].weight)
        nn.init.zeros_(self.gate_encoder[-1].bias)

    def forward(self, user, target, history, history_mask):
        user = user.float()
        target = target.float()
        history = history.float()
        mask = history_mask.bool()
        if self.attention == "learned":
            # A3: explicit candidate-to-history matching; identity
            # initialization avoids an arbitrary random rotation.
            query = F.normalize(self.query_proj(target), dim=-1)
            keys = F.normalize(self.key_proj(history), dim=-1)
        else:
            # A2: retain A1's fixed user+candidate cosine attention.
            query = F.normalize(user + target, dim=-1)
            keys = F.normalize(history, dim=-1)
        scores = torch.einsum('bd,bkd->bk', query, keys) / self.temperature
        scores = scores.masked_fill(~mask, -1e4)
        weights = torch.softmax(scores.float(), dim=-1) * mask.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        evidence = torch.einsum('bk,bkd->bd', weights, history)
        features = torch.cat(
            [user, target, evidence, target * evidence], dim=-1
        )
        gate = 1.0 + 0.5 * torch.tanh(self.gate_encoder(features).float())
        # No history must not affect the original GNN injection.
        gate = torch.where(mask.any(dim=-1, keepdim=True), gate,
                           torch.ones_like(gate))
        return gate, weights
