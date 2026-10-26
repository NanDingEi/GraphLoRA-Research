#!/usr/bin/env python3
"""CPU preflight for explicit granular CF signals and zero-init LoRA rank routing."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from experiments.no31_cora_original.granularity_signals import build_coarse_fine_signals


def main():
    torch.manual_seed(42)
    user = torch.randn(3, 16)
    item = torch.randn(9, 16)
    users = torch.tensor([0, 1, 2])
    targets = torch.tensor([3, 4, 5])
    history = torch.tensor([[1, 2, 0], [2, 0, 0], [0, 0, 0]])
    coarse, fine = build_coarse_fine_signals(
        users, targets, history, user, item, padding_idx=0
    )
    assert coarse.shape == fine.shape == (3, 32)
    expected = torch.cat([user[users], item[targets]], -1)
    assert torch.equal(coarse, expected), "Coarse must equal original MF pair"
    assert torch.isfinite(fine).all()
    assert torch.allclose(coarse.norm(dim=-1), fine.norm(dim=-1), atol=1e-5)
    assert not torch.allclose(coarse, fine), "Fine must differ from original pair"
    # Padding embedding must never influence target-conditioned history pooling.
    modified = item.clone()
    modified[0] = 1e6
    _, fine_modified = build_coarse_fine_signals(
        users, targets, history, user, modified, padding_idx=0
    )
    assert torch.allclose(fine, fine_modified, atol=1e-5)
    # All-padding histories must be valid and target-user dependent.
    assert fine[2].abs().sum() > 0
    # Identical coarse vectors + different historical items => different fine.
    repeat_users = torch.tensor([0, 0])
    repeat_items = torch.tensor([3, 3])
    histories = torch.tensor([[1, 2, 0], [5, 6, 0]])
    same_coarse, distinct_fine = build_coarse_fine_signals(
        repeat_users, repeat_items, histories, user, item
    )
    assert torch.equal(same_coarse[0], same_coarse[1])
    assert not torch.allclose(distinct_fine[0], distinct_fine[1])
    # Rank gating must be an exact no-op at zero initialization.
    gate = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 8))
    nn.init.zeros_(gate[-1].weight)
    nn.init.zeros_(gate[-1].bias)
    weight = torch.randn(3, 64, 16)
    g = 0.1 * torch.tanh(gate(fine))
    factor = torch.cat([torch.ones_like(g), 1 + g], dim=-1)
    gated = weight * factor.unsqueeze(1)
    assert torch.equal(gated, weight)
    loss = gated.square().sum()
    loss.backward()
    assert gate[-1].weight.grad is not None
    assert gate[-1].weight.grad.abs().sum().item() > 0
    print("[NIGHT18-CPU-PASS] coarse exact parity; history-conditioned fine;"
          " padding-safe; finite; norm matched; rank gate exact identity and gradients")


if __name__ == "__main__":
    main()
