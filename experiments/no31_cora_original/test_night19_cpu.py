#!/usr/bin/env python3
"""CPU-only NIGHT19 preflight: signal decomposition, zero-init parity and routing."""
import os
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from experiments.no31_cora_original.adapter import OriginalCoRARuntime
from experiments.no31_cora_original.granularity_signals import build_coarse_fine_signals
from experiments.no31_cora_original.night19 import EXPERIMENTS


@contextmanager
def temporary_env(entries):
    old = {key: os.environ.get(key) for key in entries}
    try:
        os.environ.update(entries)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main():
    assert len(EXPERIMENTS) == 8
    assert len({row["id"] for row in EXPERIMENTS}) == 8

    torch.manual_seed(20261027)
    user = torch.randn(4, 16)
    item = torch.randn(12, 16)
    users = torch.tensor([0, 1, 2, 3])
    targets = torch.tensor([4, 5, 6, 7])
    history = torch.tensor([[1, 2, 0], [2, 3, 0], [0, 0, 0], [5, 6, 7]])
    coarse, fine, user_item, target_history = build_coarse_fine_signals(
        users, targets, history, user, item, padding_idx=0,
        return_components=True,
    )
    expected = torch.cat((user[users], item[targets]), dim=-1)
    assert torch.equal(coarse, expected), "Coarse signal must equal original MF pair"
    assert all(x.shape == (4, 32) for x in (fine, user_item, target_history))
    assert all(torch.isfinite(x).all() for x in (coarse, fine, user_item, target_history))
    assert torch.allclose(coarse.norm(dim=-1), fine.norm(dim=-1), atol=1e-5)
    assert torch.allclose(coarse.norm(dim=-1), user_item.norm(dim=-1), atol=1e-5)
    assert torch.allclose(coarse.norm(dim=-1)[[0, 1, 3]],
                          target_history.norm(dim=-1)[[0, 1, 3]], atol=1e-5)
    assert target_history[2].abs().sum().item() == 0.0
    assert not torch.allclose(user_item, target_history)

    expected_scales = {
        "r8": [1.0] * 8,
        "r16": [1.0] * 8,
        "late8": [0, 0, 0, 0, 0, 0, 1, 1],
        "late4": [0, 0, 0, 0, 0, 0, 0, 1],
        "coarse_to_fine": [0, 0, 0, 0, 0.25, 0.5, 0.75, 1],
        "fine_to_coarse": [1, 0.75, 0.5, 0.25, 0, 0, 0, 0],
        "history": [1.0] * 8,
        "user_item": [1.0] * 8,
    }
    common = {
        "NO31_CORA_ALPHA": "48",
        "NO31_CORA_PROJECTIONS": "q_proj,k_proj,v_proj,o_proj",
        "NO31_CORA_LAYERS": "0-31",
        "NO31_CORA_INPUT": "concat",
        "NO31_CORA_INJECTION_MODE": "shared",
        "NO31_CORA_RANK_PROFILE": "uniform",
        "NO31_CORA_GROUP_PROJECTORS": "group8",
        "NO31_CORA_GRANULARITY_MODE": "original",
        "NO31_CORA_RESIDUAL_RATIO": "0",
        "NO31_CORA_DEPTH_PROFILE": "uniform",
        "NO31_CORA_PROJECTION_SCALES": "q_proj=1,k_proj=1,v_proj=1,o_proj=1",
    }
    base = torch.randn(3, 64, 16)
    residual = torch.randn(3, 64, 16)
    for row in EXPERIMENTS:
        mode = row["fine_residual_mode"]
        with temporary_env({
            **common,
            "NO31_CORA_FINE_RESIDUAL_MODE": mode,
        }):
            runtime = OriginalCoRARuntime()
        assert runtime.group_projector_mode == "group8"
        assert runtime.fine_residual_mode == mode
        scales = [runtime.fine_residual_group_scale(i) for i in range(8)]
        assert scales == expected_scales[mode], (mode, scales)
        for group in range(8):
            merged = runtime.apply_fine_residual(base, residual, group)
            delta = merged - base
            if scales[group] == 0:
                assert torch.equal(merged, base), (mode, group)
            base_norm = base.float().norm(dim=(1, 2))
            delta_norm = delta.float().norm(dim=(1, 2))
            assert (delta_norm <= 0.1 * base_norm + 1e-5).all()
            if scales[group] > 0:
                # Random residuals exceed the cap, so the configured group
                # coefficient must remain visible after clipping.
                ratio = delta_norm / base_norm.clamp_min(1e-8)
                assert torch.allclose(
                    ratio, torch.full_like(ratio, 0.1 * scales[group]),
                    atol=1e-4, rtol=1e-4,
                ), (mode, group, ratio, scales[group])
            if mode == "r8":
                assert torch.equal(delta[..., :8], torch.zeros_like(delta[..., :8]))
        print(f"[NIGHT19-CPU] {mode}: group scales={scales}")

    # Zero-init must preserve the base A exactly, while the residual head can learn.
    head = nn.Sequential(nn.Linear(32, 8), nn.GELU(), nn.Linear(8, 64 * 16))
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    features = torch.randn(3, 32)
    base_a = torch.randn(3, 64, 16)
    delta_a = head(features).view(3, 64, 16)
    assert torch.equal(delta_a, torch.zeros_like(delta_a))
    merged_a = base_a + delta_a
    assert torch.equal(merged_a, base_a), "Zero-init residual must exactly preserve base A"
    probe = torch.randn_like(merged_a)
    (merged_a * probe).sum().backward()
    assert head[-1].weight.grad is not None
    assert head[-1].weight.grad.abs().sum().item() > 0
    print("[NIGHT19-CPU-PASS] coarse parity, decomposed fine signals, all 8 routes, "
          "10% residual cap, rank8 mask, exact zero-init parity and gradient flow")


if __name__ == "__main__":
    main()
