#!/usr/bin/env python3
"""NIGHT15 CPU-only projector verification: shape, identity, RNG, gradients, groups."""
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.no31_cora_original.adapter import OriginalCoRARuntime
from experiments.no31_cora_original.weight_generator import PerceptualWeightsGenerator
from experiments.no31_cora_original.night15 import EXPERIMENTS, validate

MODES = ("off", "shared", "group4")


@contextmanager
def env_override(values):
    previous = {k: os.environ.get(k) for k in values}
    try:
        os.environ.update({key: str(value) for key, value in values.items()})
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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


def main():
    validate()
    assert len(EXPERIMENTS) == 3
    assert tuple(row["group_projectors"] for row in EXPERIMENTS) == MODES

    for mode in MODES:
        with env_override({
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
        }):
            runtime = OriginalCoRARuntime()
            assert runtime.group_projector_mode == mode
            assert runtime.rank_profile is None and runtime.rank == 16
            assert runtime.scaling == 3.0
            assert runtime.expected_targets == 128
            assert all(runtime.rank_for_layer(j) == 16 for j in range(32))

    # All variants must have the same common QFormer/final initialization,
    # leave the post-construction global RNG stream unmodified, and produce
    # the same initial rank16 dynamic A.
    torch.manual_seed(20261025)
    features = torch.randn(3, 32)
    outputs = {}
    states = {}
    models = {}
    for mode in MODES:
        torch.manual_seed(424242)
        generator = make_generator(mode).eval()
        states[mode] = torch.get_rng_state().clone()
        models[mode] = generator
        outputs[mode] = generator(features)

    common = models["off"]
    assert torch.equal(states["off"], states["shared"])
    assert torch.equal(states["off"], states["group4"])
    for mode in ("shared", "group4"):
        g = models[mode]
        assert torch.equal(common.final.weight, g.final.weight)
        assert torch.equal(common.final.bias, g.final.bias)
        assert torch.equal(common.query_tokens, g.query_tokens)
        assert torch.equal(
            common.qformer.encoder.layer[0].attention.self.query.weight,
            g.qformer.encoder.layer[0].attention.self.query.weight,
        )
        weights = outputs[mode]
        assert set(weights) == {0, 1, 2, 3}
        assert len(g.group_projectors) == (1 if mode == "shared" else 4)
        for group_idx in range(4):
            w = weights[group_idx]
            assert w.shape == (3, 32, 16)
            assert torch.equal(w, outputs["off"]), (
                mode, group_idx, "zero-initialized projector changed initial A"
            )
            assert torch.isfinite(w).all()
            assert torch.allclose(
                w.flatten(1).norm(dim=-1),
                torch.ones(3),
                atol=1e-5,
            )
        for p in g.group_projectors:
            assert p[2].weight.count_nonzero().item() == 0
            assert p[2].bias.count_nonzero().item() == 0

    # The projections are independently addressed: perturbing group 2 must
    # not affect groups 0,1,3.
    group4 = models["group4"]
    with torch.no_grad():
        group4.group_projectors[2][2].bias.fill_(0.1)
    changed = group4(features)
    for group_idx in range(4):
        is_different = not torch.allclose(
            changed[group_idx], outputs["off"], atol=1e-6, rtol=0.0
        )
        assert is_different == (group_idx == 2), group_idx

    # Backprop through the shared head and group-specific projector outputs.
    group4.zero_grad(set_to_none=True)
    loss = 0
    for group_idx, w in changed.items():
        loss = loss + (w * torch.randn_like(w)).sum()
    loss.backward()
    for group_idx in range(4):
        grad = group4.group_projectors[group_idx][2].weight.grad
        assert grad is not None and grad.isfinite().all()
        assert grad.abs().sum().item() > 0, (
            f"group {group_idx} projector received no gradient"
        )
    final_grad = group4.final.weight.grad
    assert final_grad is not None and final_grad.abs().sum().item() > 0

    # Real dimensions: each 512->32->512 projector has 33,312 params.
    per_projector = (512 * 32 + 32) + (32 * 512 + 512)
    assert per_projector == 33_312
    assert per_projector * 4 == 133_248
    assert sum(p.numel() for p in models["shared"].group_projectors.parameters()) == (
        (32 * 32 + 32) + (32 * 32 + 32)
    )
    assert sum(p.numel() for p in models["group4"].group_projectors.parameters()) == (
        4 * ((32 * 32 + 32) + (32 * 32 + 32))
    )
    print(
        "[NIGHT15-CPU-PASS] G2/shared/group4 initial A identical, "
        "QFormer/final weights identical, post-init RNG identical, "
        "group addressing independent, gradients present; "
        "real extra params shared=33312 group4=133248",
        flush=True,
    )


if __name__ == "__main__":
    main()
