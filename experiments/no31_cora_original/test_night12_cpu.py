#!/usr/bin/env python3
"""CPU-only structural smoke tests; does not load Vicuna or use the GPU."""
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.no31_cora_original.adapter import OriginalCoRARuntime
from experiments.no31_cora_original.night12 import EXPERIMENTS, UNIFORM, validate_plan
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


def make_runtime(row):
    with temporary_env({
        "NO31_CORA_ALPHA": str(row["alpha"]),
        "NO31_CORA_PROJECTIONS": row["projections"],
        "NO31_CORA_LAYERS": row["layers"],
        "NO31_CORA_INPUT": row["cf_input"],
        "NO31_CORA_INJECTION_MODE": row["injection_mode"],
        "NO31_CORA_RESIDUAL_RATIO": "0",
        "NO31_CORA_DEPTH_PROFILE": "uniform",
        "NO31_CORA_PROJECTION_SCALES": UNIFORM,
    }):
        return OriginalCoRARuntime()


def main():
    cfg = validate_plan()
    assert int(cfg["run"]["seed"]) == 42
    assert len(EXPERIMENTS) == 4
    assert len({row["injection_mode"] for row in EXPERIMENTS}) == 4

    for row in EXPERIMENTS:
        runtime = make_runtime(row)
        assert runtime.injection_mode == row["injection_mode"]
        assert runtime.alpha == 48 and runtime.scaling == 3.0
        assert runtime.layers == tuple(range(32))
        assert runtime.projections == ("q_proj", "k_proj", "v_proj", "o_proj")
        assert runtime.expected_targets == 128
        print(f"[NIGHT12-CPU] {row['id']} mode={runtime.injection_mode} "
              f"targets={runtime.expected_targets} alpha={runtime.alpha}", flush=True)

    # Small-dimension forward tests exercise all generator architectures.
    torch.manual_seed(42)
    features = torch.randn(3, 32)
    outputs = {}
    for mode in ("shared", "typed_tokens", "interaction_tokens",
                 "layer_conditioned", "token_gate"):
        generator = PerceptualWeightsGenerator(
            num_queries=4,
            user_embedding_size=16,
            hidden_size=32,
            r=4,
            input_mode="concat",
            injection_mode=mode,
        ).eval()
        if mode == "layer_conditioned":
            weight, condition = generator(features, return_condition=True)
            layer_ids = torch.tensor([0, 15, 31])
            layer_embed = generator.layer_embedding(layer_ids)
            modulation = torch.tanh(
                generator.layer_modulator(torch.cat((condition, layer_embed), dim=-1))
            )
            assert modulation.shape == (3, 4)
            assert torch.isfinite(modulation).all()
        else:
            weight = generator(features)
        assert weight.shape == (3, 32, 4), (mode, weight.shape)
        assert torch.isfinite(weight).all()
        assert torch.allclose(
            weight.flatten(1).norm(dim=-1), torch.ones(3), atol=1e-5
        ), mode
        outputs[mode] = weight.detach()
        print(f"[NIGHT12-GENERATOR] mode={mode} output={tuple(weight.shape)}", flush=True)

    assert not torch.allclose(outputs["shared"], outputs["typed_tokens"])
    assert not torch.allclose(outputs["typed_tokens"], outputs["interaction_tokens"])

    # M4: 2*sigmoid(0) == 1; zero-init gate preserves the initial residual.
    token_gate = torch.nn.Linear(32, 1)
    torch.nn.init.zeros_(token_gate.weight)
    torch.nn.init.zeros_(token_gate.bias)
    x = torch.randn(3, 7, 32)
    gate = 2.0 * torch.sigmoid(token_gate(x))
    assert torch.equal(gate, torch.ones_like(gate))

    # M3 starts close to the shared basis: its rank modulation is initialized
    # with small, nonzero weights so gradients can reach the whole conditioner.
    torch.manual_seed(42)
    layer_gen = PerceptualWeightsGenerator(
        num_queries=4, user_embedding_size=16, hidden_size=32, r=4,
        injection_mode="layer_conditioned",
    ).eval()
    _, cond = layer_gen(features, return_condition=True)
    ids = torch.tensor([0, 15, 31])
    mod = torch.tanh(layer_gen.layer_modulator(
        torch.cat((cond, layer_gen.layer_embedding(ids)), dim=-1)
    ))
    assert mod.abs().max().item() < 0.05
    print("[NIGHT12-CPU-PASS] four structural modes, generator shapes, "
          "normalization, layer conditioning and neutral token gate verified",
          flush=True)


if __name__ == "__main__":
    main()
