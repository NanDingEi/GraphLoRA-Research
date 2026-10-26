#!/usr/bin/env python3
"""CPU-only checks for NIGHT17 progressive CoRA layer-to-projector mappings."""
import os
import torch

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.no31_cora_original.weight_generator import PerceptualWeightsGenerator
from experiments.no31_cora_original.adapter import OriginalCoRARuntime

MODES = (
    "group8",
    "coarse_to_fine",
    "fine_to_coarse",
    "hierarchical_residual",
)
EXPECTED_COUNTS = {mode: 8 for mode in MODES}


def make_generator(mode):
    return PerceptualWeightsGenerator(
        num_queries=4,
        user_embedding_size=16,
        hidden_size=32,
        r=16,
        input_mode="concat",
        injection_mode="shared",
        group_projector_mode=mode,
    )


def runtime_mapping(mode):
    overrides = {
        "NO31_CORA_ALPHA": "48",
        "NO31_CORA_PROJECTIONS": "q_proj,k_proj,v_proj,o_proj",
        "NO31_CORA_LAYERS": "0-31",
        "NO31_CORA_INPUT": "concat",
        "NO31_CORA_INJECTION_MODE": "shared",
        "NO31_CORA_RANK_PROFILE": "uniform",
        "NO31_CORA_GROUP_PROJECTORS": mode,
        "NO31_CORA_RESIDUAL_RATIO": "0",
        "NO31_CORA_DEPTH_PROFILE": "uniform",
        "NO31_CORA_PROJECTION_SCALES": "q_proj=1,k_proj=1,v_proj=1,o_proj=1",
    }
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        os.environ.update(overrides)
        runtime = OriginalCoRARuntime()
        return [runtime.group_for_layer(layer) for layer in range(32)]
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def output_dict(output):
    assert isinstance(output, dict), type(output)
    return output


def main():
    torch.manual_seed(20261026)
    features = torch.randn(3, 32)

    # P4's uniform 4-layer grouping is the controlled same-session anchor.
    expected = {
        "group8": [layer // 4 for layer in range(32)],
        "coarse_to_fine": (
            [0] * 8 + [1] * 8 + [2] * 4 + [3] * 4
            + [4] * 2 + [5] * 2 + [6] * 2 + [7] * 2
        ),
        "fine_to_coarse": (
            [layer // 2 for layer in range(8)]
            + [4] * 4 + [5] * 4 + [6] * 8 + [7] * 8
        ),
        "hierarchical_residual": (
            [0] * 8 + [1] * 8 + [2] * 4 + [3] * 4
            + [4] * 2 + [5] * 2 + [6] * 2 + [7] * 2
        ),
    }
    for mode in MODES:
        actual = runtime_mapping(mode)
        assert actual == expected[mode], (mode, actual, expected[mode])
        assert min(actual) == 0 and max(actual) == 7, mode

    # Added projectors must preserve the base generator's initialization and
    # global CPU RNG stream so sampler/P-layer initialization remains aligned.
    torch.manual_seed(424242)
    base_gen = make_generator("off").eval()
    base_rng = torch.get_rng_state().clone()
    base_output = base_gen(features)
    outputs = {}
    generators = {}
    for mode in MODES:
        torch.manual_seed(424242)
        gen = make_generator(mode).eval()
        generators[mode] = gen
        assert torch.equal(torch.get_rng_state(), base_rng), f"RNG changed: {mode}"
        assert len(gen.group_projectors) == EXPECTED_COUNTS[mode], mode
        output = output_dict(gen(features))
        assert set(output) == set(range(8)), (mode, output.keys())
        for group_idx, weight in output.items():
            assert weight.shape == (3, 32, 16), (mode, group_idx, weight.shape)
            assert torch.isfinite(weight).all(), (mode, group_idx)
            assert torch.allclose(weight, base_output, atol=1e-6, rtol=0), (
                mode, group_idx, (weight - base_output).abs().max().item()
            )
            assert torch.allclose(
                weight.flatten(1).norm(dim=-1), torch.ones(3), atol=1e-5
            ), (mode, group_idx)
        outputs[mode] = output

    # Every new projector branch must receive gradients after a differentiable
    # loss; hierarchical mode has one shared branch plus seven depth residuals.
    for mode in MODES:
        gen = generators[mode]
        gen.train()
        gen.zero_grad(set_to_none=True)
        torch.manual_seed(98765)
        result = output_dict(gen(features))
        loss = sum((weight * torch.randn_like(weight)).sum()
                   for weight in result.values())
        loss.backward()
        assert gen.final.weight.grad is not None
        assert gen.final.weight.grad.abs().sum().item() > 0, mode
        for idx, projector in enumerate(gen.group_projectors):
            grad = projector[-1].weight.grad
            assert grad is not None and grad.abs().sum().item() > 0, (mode, idx)

    # Perturbing one independent residual branch must only affect its group.
    for mode in MODES:
        gen = make_generator(mode).eval()
        before = output_dict(gen(features))
        with torch.no_grad():
            gen.group_projectors[2][-1].bias.fill_(0.1)
        after = output_dict(gen(features))
        changed = [
            idx for idx in before
            if not torch.allclose(before[idx], after[idx], atol=1e-7, rtol=0)
        ]
        assert changed == [2], (mode, changed)

    print(
        "[NIGHT17-CPU-PASS] C2F/F2C/uniform/hierarchical mappings, "
        "G2 initialization parity, RNG preservation, projector gradients, "
        "and group isolation all verified.",
        flush=True,
    )


if __name__ == "__main__":
    main()
