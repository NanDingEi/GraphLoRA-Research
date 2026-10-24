#!/usr/bin/env python3
"""CPU-only checks for NIGHT10 controls; no 7B model or GPU is loaded."""
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.no31_cora_original.adapter import OriginalCoRARuntime
from experiments.no31_cora_original.night10 import EXPERIMENTS, validate_plan

@contextmanager
def env_override(values):
    old={k:os.environ.get(k) for k in values}
    try:
        os.environ.update({k:str(v) for k,v in values.items()})
        yield
    finally:
        for k,v in old.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v

def runtime_for(row):
    with env_override({
        "NO31_CORA_ALPHA":row["alpha"],
        "NO31_CORA_PROJECTIONS":row["projections"],
        "NO31_CORA_LAYERS":row["layers"],
        "NO31_CORA_INPUT":row["cf_input"],
        "NO31_CORA_RESIDUAL_RATIO":row["residual_ratio"],
        "NO31_CORA_DEPTH_PROFILE":row["depth_profile"],
    }):
        return OriginalCoRARuntime()

def main():
    validate_plan()
    assert len(EXPERIMENTS)==8
    assert all(row["seed"]==42 for row in EXPERIMENTS)
    for row in EXPERIMENTS:
        rt=runtime_for(row)
        assert rt.alpha==row["alpha"]
        assert rt.scaling==row["alpha"]/16
        assert rt.expected_targets==128 if row["projections"]=="q_proj,k_proj,v_proj,o_proj" else 64
        assert rt.residual_ratio==row["residual_ratio"]
        assert rt.depth_profile==row["depth_profile"]
        assert rt.input_mode=="concat" and rt.layers==tuple(range(32))
        print(f"[NIGHT10-CPU] {row['id']} seed=42 alpha={rt.alpha} "
              f"scale={rt.scaling:g} P={rt.expected_targets} "
              f"p_lr={row['p_lr']} ratio={rt.residual_ratio:g} "
              f"profile={rt.depth_profile}",flush=True)

    # Original default path must remain unchanged.
    base=torch.ones(2,3,16)
    residual=torch.full_like(base,0.1)
    control=runtime_for(EXPERIMENTS[1])
    assert torch.equal(control.apply_residual_controls(base,residual,0),residual)

    # The cap must bound the residual norm to at most 50% of base per token.
    capped=runtime_for(EXPERIMENTS[6])
    large=torch.full_like(base,10.0)
    out=capped.apply_residual_controls(base,large,layer_index=15)
    ratio=out.float().norm(dim=-1)/(base.float().norm(dim=-1)+1e-12)
    assert torch.all(ratio<=0.50001),ratio

    # Late-linear profile retains all layers but smoothly attenuates early layers.
    late=runtime_for(EXPERIMENTS[7])
    assert abs(late.layer_scale(0)-0.5)<1e-12
    assert abs(late.layer_scale(31)-1.0)<1e-12
    half=late.apply_residual_controls(base,residual,layer_index=0)
    full=late.apply_residual_controls(base,residual,layer_index=31)
    assert torch.allclose(half,residual*0.5)
    assert torch.allclose(full,residual)
    print("[NIGHT10-CPU-PASS] 8 configs, seed42 invariant, original path parity, "
          "residual cap and depth profile verified",flush=True)

if __name__=="__main__":
    main()
