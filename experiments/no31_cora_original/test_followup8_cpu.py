#!/usr/bin/env python3
"""Cheap no-GPU smoke for CoRA follow-up F1-F8 configuration wiring."""
import os
from contextlib import contextmanager
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from experiments.no31_cora_original.followup8 import FOLLOWUP8, LOOKUP
from experiments.no31_cora_original.night8 import selected_targets
from experiments.no31_cora_original.adapter import OriginalCoRARuntime

@contextmanager
def environ_override(**values):
    old={k:os.environ.get(k) for k in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key,value in old.items():
            if value is None:
                os.environ.pop(key,None)
            else:
                os.environ[key]=value

def main():
    assert len(FOLLOWUP8)==len(LOOKUP)==8
    expected={
        "F1":(64,64,42),
        "F2":(64,96,42),
        "F3":(128,96,42),
        "F4":(128,48,42),
        "F5":(32,32,42),
        "F6":(32,32,42),
        "F7":(64,32,43),
        "F8":(128,64,43),
    }
    combos=set()
    for row in FOLLOWUP8:
        with environ_override(
            NO31_CORA_ALPHA=str(row["alpha"]),
            NO31_CORA_PROJECTIONS=row["projections"],
            NO31_CORA_LAYERS=row["layers"],
            NO31_CORA_INPUT=row["cf_input"],
        ):
            runtime=OriginalCoRARuntime()
        e_targets,e_alpha,e_seed=expected[row["id"]]
        assert runtime.expected_targets==e_targets==selected_targets(row)
        assert runtime.alpha==e_alpha==row["alpha"]
        assert row["seed"]==e_seed
        assert runtime.scaling==runtime.alpha/16
        assert runtime.layers==tuple(range(32))
        assert runtime.input_mode=="concat"
        combo=(runtime.projections,runtime.alpha,row["seed"])
        assert combo not in combos,"Identical duplicate experiment"
        combos.add(combo)
        print(f"[FOLLOWUP8-CPU] {row['id']} "
              f"P={e_targets} alpha={e_alpha} "
              f"scale={runtime.scaling} seed={e_seed}",flush=True)
    # Ensure seed43 confirmation trials are EXACT architectural matches
    # to original R2/R7, with only the seed changed.
    r2=next(e for e in FOLLOWUP8 if e["id"]=="F7")
    r7=next(e for e in FOLLOWUP8 if e["id"]=="F8")
    assert r2["projections"]=="v_proj,o_proj" and r2["alpha"]==32
    assert r7["projections"]=="q_proj,k_proj,v_proj,o_proj" and r7["alpha"]==64
    assert r2["p_lr"]==r7["p_lr"]==1.0
    assert r2["seed"]==r7["seed"]==43
    print("[FOLLOWUP8-CPU-PASS] eight distinct, validated CoRA "
          "settings; paired seed43 comparison preserved",flush=True)

if __name__=="__main__":
    main()
