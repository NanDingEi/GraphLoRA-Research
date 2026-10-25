#!/usr/bin/env python3
"""Low-cost CPU smoke for the 8 CoRA settings; no 7B weights or GPU."""
import os
from pathlib import Path
import sys
from contextlib import contextmanager

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from experiments.no31_cora_original.night8 import EXPERIMENTS, selected_targets
from experiments.no31_cora_original.adapter import OriginalCoRARuntime
from experiments.no31_cora_original.weight_generator import PerceptualWeightsGenerator
import torch


@contextmanager
def temporary_env(entries):
    old={k:os.environ.get(k) for k in entries}
    try:
        os.environ.update(entries)
        yield
    finally:
        for k,val in old.items():
            if val is None:
                os.environ.pop(k,None)
            else:
                os.environ[k]=val


def main():
    assert len(EXPERIMENTS)==8
    assert len(set(e["id"] for e in EXPERIMENTS))==8
    expected={"R1":64,"R2":64,"R3":64,"R4":32,
              "R5":128,"R6":128,"R7":128,"R8":128}
    for row in EXPERIMENTS:
        with temporary_env({
            "NO31_CORA_ALPHA": str(row["alpha"]),
            "NO31_CORA_PROJECTIONS": row["projections"],
            "NO31_CORA_LAYERS": row["layers"],
            "NO31_CORA_INPUT": row["cf_input"],
            "NO31_CORA_INJECTION_MODE": "shared",
            "NO31_CORA_RESIDUAL_RATIO": "0",
            "NO31_CORA_DEPTH_PROFILE": "uniform",
            "NO31_CORA_PROJECTION_SCALES": "q_proj=1,k_proj=1,v_proj=1,o_proj=1",
            "NO31_CORA_RANK_PROFILE": "uniform",
        }):
            runtime=OriginalCoRARuntime()
        assert runtime.expected_targets==selected_targets(row)==expected[row["id"]]
        assert runtime.alpha==row["alpha"]
        assert runtime.scaling==row["alpha"]/16
        assert runtime.input_mode==row["cf_input"]
        assert runtime.injection_mode=="shared"
        print(f"[NIGHT8-CPU] {row['id']} target_modules={runtime.expected_targets} "
              f"alpha={runtime.alpha} input={runtime.input_mode}")
    # Generator shape and input-mode intervention; use reduced dimensions
    # strictly for CPU smoke, not for actual R1-R8 training.
    torch.manual_seed(20261024)
    generator=PerceptualWeightsGenerator(
        num_queries=4,user_embedding_size=16,hidden_size=32,r=4,
    ).eval()
    features=torch.randn(3,32)
    concat_w=generator(features)
    assert concat_w.shape==(3,32,4)
    assert torch.allclose(concat_w.flatten(1).norm(dim=-1),
                          torch.ones(3),atol=1e-5)
    generator.input_mode="two_tokens"
    two_w=generator(features)
    assert two_w.shape==concat_w.shape
    assert torch.allclose(two_w.flatten(1).norm(dim=-1),
                          torch.ones(3),atol=1e-5)
    assert (concat_w-two_w).abs().max().item()>1e-6, (
        "R5 did not alter QFormer encoder representation"
    )
    print("[NIGHT8-CPU-PASS] 8 distinct config encodings; R5 changes "
          "QFormer output; generator shape and L2 normalization preserved")


if __name__=="__main__":
    main()
