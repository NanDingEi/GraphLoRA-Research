#!/usr/bin/env python3
"""CPU-only preflight for NIGHT11; does not load the 7B model or use a GPU."""
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.no31_cora_original.adapter import OriginalCoRARuntime
from experiments.no31_cora_original.night11 import (
    EXPERIMENTS, QK_HALF_VO_FULL, UNIFORM, validate_plan,
)


@contextmanager
def env_override(values):
    old = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update({key: str(value) for key, value in values.items()})
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def runtime_for(row):
    with env_override({
        "NO31_CORA_ALPHA": row["alpha"],
        "NO31_CORA_PROJECTIONS": row["projections"],
        "NO31_CORA_LAYERS": row["layers"],
        "NO31_CORA_INPUT": row["cf_input"],
        "NO31_CORA_RESIDUAL_RATIO": row["residual_ratio"],
        "NO31_CORA_DEPTH_PROFILE": row["depth_profile"],
        "NO31_CORA_PROJECTION_SCALES": row["projection_scales"],
    }):
        return OriginalCoRARuntime()


def main():
    validate_plan()
    assert len(EXPERIMENTS) == 8
    assert all(row["seed"] == 42 for row in EXPERIMENTS)
    for row in EXPERIMENTS:
        rt = runtime_for(row)
        assert rt.alpha == row["alpha"]
        assert rt.scaling == row["alpha"] / 16
        assert rt.expected_targets == 128
        assert rt.residual_ratio == row["residual_ratio"]
        assert rt.depth_profile == row["depth_profile"]
        assert rt.projection_scales["q_proj"] == (
            0.5 if row["projection_scales"] == QK_HALF_VO_FULL else 1.0
        )
        assert rt.input_mode == "concat"
        assert rt.layers == tuple(range(32))
        print(f"[NIGHT11-CPU] {row['id']} alpha={rt.alpha} "
              f"scale={rt.scaling:g} P={rt.expected_targets} "
              f"p_lr={row['p_lr']} residual_ratio={rt.residual_ratio:g} "
              f"projection_scales={row['projection_scales']}", flush=True)

    base = torch.ones(2, 3, 16)
    residual = torch.full_like(base, 0.1)

    # Uniform projection scales must preserve the original CoRA residual exactly.
    default_rt = runtime_for(EXPERIMENTS[0])
    assert torch.equal(
        default_rt.apply_residual_controls(base, residual, layer_index=0, projection="q_proj"),
        residual,
    )

    # N7's residual norm cap must bound each token to 25% of the base norm.
    capped_rt = runtime_for(EXPERIMENTS[6])
    large = torch.full_like(base, 10.0)
    capped = capped_rt.apply_residual_controls(base, large, layer_index=15, projection="q_proj")
    ratio = capped.float().norm(dim=-1) / base.float().norm(dim=-1).clamp_min(1e-12)
    assert torch.all(ratio <= 0.25001), ratio

    # N8 halves Q/K residuals while leaving V/O residuals unchanged.
    scaled_rt = runtime_for(EXPERIMENTS[7])
    q_out = scaled_rt.apply_residual_controls(base, residual, layer_index=0, projection="q_proj")
    k_out = scaled_rt.apply_residual_controls(base, residual, layer_index=0, projection="k_proj")
    v_out = scaled_rt.apply_residual_controls(base, residual, layer_index=0, projection="v_proj")
    o_out = scaled_rt.apply_residual_controls(base, residual, layer_index=0, projection="o_proj")
    assert torch.allclose(q_out, residual * 0.5)
    assert torch.allclose(k_out, residual * 0.5)
    assert torch.equal(v_out, residual)
    assert torch.equal(o_out, residual)
    print("[NIGHT11-CPU-PASS] eight configs, seed42, original-path parity, "
          "alpha sweep, residual cap and projection-wise scales verified", flush=True)


if __name__ == "__main__":
    main()
