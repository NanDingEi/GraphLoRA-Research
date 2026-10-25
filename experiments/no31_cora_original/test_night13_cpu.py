#!/usr/bin/env python3
"""CPU-only checks for CoRA's four-group rank profile; no Vicuna/GPU required."""
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.no31_cora_original.adapter import OriginalCoRARuntime
from experiments.no31_cora_original.weight_generator import PerceptualWeightsGenerator


@contextmanager
def temporary_env(entries):
    old = {key: os.environ.get(key) for key in entries}
    try:
        os.environ.update({key: str(value) for key, value in entries.items()})
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main():
    env = {
        "NO31_CORA_ALPHA": "48",
        "NO31_CORA_PROJECTIONS": "q_proj,k_proj,v_proj,o_proj",
        "NO31_CORA_LAYERS": "0-31",
        "NO31_CORA_INPUT": "concat",
        "NO31_CORA_INJECTION_MODE": "shared",
        "NO31_CORA_RANK_PROFILE": "4-6-8-10",
            "NO31_CORA_GROUP_PROJECTORS": "off",
        "NO31_CORA_RESIDUAL_RATIO": "0",
        "NO31_CORA_DEPTH_PROFILE": "uniform",
        "NO31_CORA_PROJECTION_SCALES": "q_proj=1,k_proj=1,v_proj=1,o_proj=1",
    }
    with temporary_env(env):
        runtime = OriginalCoRARuntime()
        assert runtime.rank_profile == (4, 6, 8, 10)
        assert runtime.scaling == 3.0
        expected = {0: 4, 7: 4, 8: 6, 15: 6, 16: 8, 23: 8, 24: 10, 31: 10}
        for layer, rank in expected.items():
            assert runtime.rank_for_layer(layer) == rank, (layer, runtime.rank_for_layer(layer))
        assert runtime.expected_targets == 128

    torch.manual_seed(42)
    generator = PerceptualWeightsGenerator(
        num_queries=4,
        user_embedding_size=16,
        hidden_size=32,
        r=16,
        input_mode="concat",
        injection_mode="shared",
        rank_profile=(4, 6, 8, 10),
    )
    features = torch.randn(3, 32)
    weights = generator(features)
    assert set(weights) == {4, 6, 8, 10}
    for rank in (4, 6, 8, 10):
        weight = weights[rank]
        assert weight.shape == (3, 32, rank), (rank, weight.shape)
        assert torch.isfinite(weight).all()
        assert torch.allclose(weight.flatten(1).norm(dim=-1), torch.ones(3), atol=1e-5)

    loss = sum(
        (weight * torch.randn_like(weight)).sum()
        for weight in weights.values()
    )
    loss.backward()
    grad = generator.final.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert grad.abs().sum().item() > 0, "shared rank-10 output head has no gradient"

    print(
        "[NIGHT13-CPU-PASS] layer boundaries, 128 targets, alpha/r=3, "
        "four normalized dynamic-A shapes and gradients verified",
        flush=True,
    )


if __name__ == "__main__":
    main()
