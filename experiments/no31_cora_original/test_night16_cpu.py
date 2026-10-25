#!/usr/bin/env python3
"""CPU-only tests for NIGHT16 grouped projector architectures."""
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.no31_cora_original.weight_generator import PerceptualWeightsGenerator
from experiments.no31_cora_original.night16 import EXPERIMENTS, validate

MODES = [row["group_projectors"] for row in EXPERIMENTS]
MODULE_COUNTS = {
    "shared": 1, "group4": 4, "group2": 2, "group8": 8,
    "shared_trunk4": 0, "shared_group4": 5,
    "rank_group4": 0, "group4_linear": 4,
}
OUTPUT_GROUP_COUNTS = {
    "shared": 4, "group4": 4, "group2": 2, "group8": 8,
    "shared_trunk4": 4, "shared_group4": 4,
    "rank_group4": 4, "group4_linear": 4,
}


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


def output_dict(output):
    if isinstance(output, dict):
        return output
    raise AssertionError("Grouped projector modes must return a dict")


def main():
    validate()
    assert len(EXPERIMENTS) == 8
    torch.manual_seed(20261025)
    features = torch.randn(3, 32)
    generators, outputs, rng_states = {}, {}, {}

    # Identical seeds must preserve G2's original QFormer/final initialization
    # and the RNG stream used later for CoRA-P and the data sampler.
    for mode in ["off"] + MODES:
        torch.manual_seed(424242)
        gen = make_generator(mode).eval()
        generators[mode] = gen
        rng_states[mode] = torch.get_rng_state().clone()
        outputs[mode] = gen(features)

    base = outputs["off"]
    base_gen = generators["off"]
    for mode in MODES:
        gen = generators[mode]
        result = output_dict(outputs[mode])
        assert set(result) == set(range(OUTPUT_GROUP_COUNTS[mode])), (mode, result.keys())
        assert len(getattr(gen, "group_projectors", [])) == MODULE_COUNTS[mode], mode
        assert torch.equal(rng_states["off"], rng_states[mode]), f"RNG changed: {mode}"
        assert torch.equal(base_gen.final.weight, gen.final.weight), mode
        assert torch.equal(base_gen.final.bias, gen.final.bias), mode
        assert torch.equal(base_gen.query_tokens, gen.query_tokens), mode
        assert torch.equal(
            base_gen.qformer.encoder.layer[0].attention.self.query.weight,
            gen.qformer.encoder.layer[0].attention.self.query.weight,
        ), mode
        for group_idx, weight in result.items():
            assert weight.shape == (3, 32, 16), (mode, group_idx, weight.shape)
            assert torch.allclose(weight, base, atol=1e-6, rtol=0), (
                mode, group_idx, (weight - base).abs().max().item()
            )
            assert torch.isfinite(weight).all()
            assert torch.allclose(
                weight.flatten(1).norm(dim=-1), torch.ones(3), atol=1e-5
            ), mode

    # Check each architecture's trainable output branch receives gradients.
    torch.manual_seed(98765)
    for mode in MODES:
        gen = generators[mode]
        gen.zero_grad(set_to_none=True)
        result = output_dict(gen(features))
        loss = sum((w * torch.randn_like(w)).sum() for w in result.values())
        loss.backward()
        assert gen.final.weight.grad is not None
        assert gen.final.weight.grad.abs().sum().item() > 0, mode
        if mode == "rank_group4":
            grad = gen.group_rank_delta.grad
            assert grad is not None and grad.abs().sum().item() > 0, mode
        elif mode == "shared_trunk4":
            for idx, head in enumerate(gen.group_projector_heads):
                assert head.weight.grad is not None
                assert head.weight.grad.abs().sum().item() > 0, (mode, idx)
        else:
            for idx, projector in enumerate(gen.group_projectors):
                grad = projector[-1].weight.grad
                assert grad is not None and grad.abs().sum().item() > 0, (mode, idx)

    # Check independent groups do not leak into neighboring groups.
    for mode in ("group2", "group4", "group8", "shared_group4",
                 "shared_trunk4", "rank_group4", "group4_linear"):
        gen = make_generator(mode).eval()
        before = output_dict(gen(features))
        with torch.no_grad():
            if mode == "group2":
                gen.group_projectors[1][-1].bias.fill_(0.1)
                target = 1
            elif mode == "group4":
                gen.group_projectors[2][-1].bias.fill_(0.1)
                target = 2
            elif mode == "group8":
                gen.group_projectors[2][-1].bias.fill_(0.1)
                target = 2
            elif mode == "shared_group4":
                gen.group_projectors[3][-1].bias.fill_(0.1)
                target = 2
            elif mode == "shared_trunk4":
                gen.group_projector_heads[2].bias.fill_(0.1)
                target = 2
            elif mode == "rank_group4":
                gen.group_rank_delta[2, 0, 1] = 0.1
                target = 2
            else:
                gen.group_projectors[2][-1].bias.fill_(0.1)
                target = 2
        after = output_dict(gen(features))
        for idx in before:
            changed = not torch.allclose(before[idx], after[idx], atol=1e-7, rtol=0)
            assert changed == (idx == target), (mode, idx, target, changed)

    print(
        "[NIGHT16-CPU-PASS] all eight modes preserve initial G2 A/RNG, "
        "have correct output groups and gradients, and isolate group-specific changes",
        flush=True,
    )


if __name__ == "__main__":
    main()
