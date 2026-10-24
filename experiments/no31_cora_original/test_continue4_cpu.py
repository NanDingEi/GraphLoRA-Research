#!/usr/bin/env python3
"""CPU-only checks for the four two-total-epoch CoRA arms."""
import os
from contextlib import contextmanager
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.no31_cora_original.continue4_two_epochs import EXPERIMENTS, BY_ID
from experiments.no31_cora_original.adapter import OriginalCoRARuntime

@contextmanager
def override(**values):
    old={k:os.environ.get(k) for k in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key,value in old.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value

def main():
    assert len(EXPERIMENTS)==len(BY_ID)==4
    expected={
        "N1":(32,"v_proj,o_proj",64,"resume"),
        "N2":(64,"q_proj,k_proj,v_proj,o_proj",128,"resume"),
        "N3":(48,"q_proj,k_proj,v_proj,o_proj",128,"resume"),
        "N4":(48,"v_proj,o_proj",64,"fresh"),
    }
    for row in EXPERIMENTS:
        alpha,projections,targets,mode=expected[row["id"]]
        assert row["alpha"]==alpha
        assert row["projections"]==projections
        assert row["seed"]==42
        assert row["mode"]==mode
        assert row["layers"]=="0-31" and row["cf_input"]=="concat"
        assert row["p_lr"]==1.0
        assert (row["checkpoint"] is not None)==(mode=="resume")
        if mode=="resume":
            assert row["start_auc"] is not None and row["start_uauc"] is not None
            assert Path(row["checkpoint"]).name=="checkpoint_0.pth"
        with override(
            NO31_CORA_ALPHA=str(alpha),
            NO31_CORA_PROJECTIONS=projections,
            NO31_CORA_LAYERS="0-31",
            NO31_CORA_INPUT="concat",
        ):
            runtime=OriginalCoRARuntime()
        assert runtime.alpha==alpha
        assert runtime.scaling==alpha/16
        assert runtime.expected_targets==targets
        assert runtime.layers==tuple(range(32))
        assert runtime.projections==tuple(projections.split(","))
        assert runtime.input_mode=="concat"
        print(f"[CONTINUE4-CPU] {row['id']} mode={mode} alpha={alpha} "
              f"scale={runtime.scaling:g} P={targets} seed=42",flush=True)
    assert len({(r["mode"],r["alpha"],r["projections"]) for r in EXPERIMENTS})==4
    print("[CONTINUE4-CPU-PASS] four unique configs; all seed42; "
          "resume arms restore epoch0 state; fresh arm runs epochs0+1",flush=True)

if __name__=="__main__":
    main()
