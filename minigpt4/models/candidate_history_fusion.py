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
