#!/usr/bin/env python3
"""R5: re-use the EXACT R4 runtime hooks before running the existing R5 resumer.

R4 adapters are NOT constructor kwargs. R4 creates them by monkey-patching
MiniGPT4Rec_v2.__init__, compute_feedback_gcn and MyLinear.forward, then
patching from_config to freeze exactly the intended parameters. R5 must
reinstall those same hooks before task.build_model().

Source of truth: experiments/collab_injection_v1/r4_entry.py,
r3_paper_adapters.py and r4_freeze.py, branch fix/collab-gated-optimizer.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

ROOT=Path("/home/bbx/user_lx/GraphLoRA_collab_v1")
EXPERIMENTS=ROOT/"experiments"/"collab_injection_v1"

def main():
    parser=argparse.ArgumentParser(description="Resume exactly one R4 epoch using its original adapter runtime")
    parser.add_argument("--round", required=True, choices=("01","02","03"))
    args=parser.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0,str(ROOT))
    sys.path.insert(0,str(EXPERIMENTS))
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("R5 requires CUDA GPU0")
    from r3_paper_adapters import install_runtime
    from r4_freeze import MODES, install_freeze, assert_frozen_unchanged

    mode,local_v=MODES[args.round]
    if mode not in ("cflora","cora"):
        raise RuntimeError(f"Unexpected R4 mode for {args.round}: {mode}")
    if (args.round=="03")!=bool(local_v):
        raise RuntimeError("R4 mode/freezing policy mismatch")
    script=ROOT/"r5_extend_one_epoch.py"
    if not script.is_file():
        raise FileNotFoundError(f"R5 resumer missing: {script}")

    # Must run BEFORE importing the training entrypoint or constructing a model.
    runtime=install_runtime(mode)
    install_freeze(args.round)
    print("[R5-R4-RUNTIME-INSTALLED] "+json.dumps({
        "round":args.round,"method":mode,
        "unfreeze_L31_V_B":bool(local_v),
        "r4_adapter_factory":"r3_paper_adapters.install_runtime",
        "r4_freeze_factory":"r4_freeze.install_freeze"},sort_keys=True),
        flush=True)

    # The user's existing R5 script contains checkpoint source paths,
    # exact optimizer/epoch restoration and trainable/frozen-state checks.
    old_argv=sys.argv[:]
    sys.argv=[str(script),"--round",args.round]
    try:
        runpy.run_path(str(script),run_name="__main__")
        model=runtime._injected_model
        if model is None or getattr(model,"r3_adapter",None) is None:
            raise RuntimeError("R5 did not construct an R4 adapter-equipped model")
        if hasattr(model,"r4_frozen_baseline_hash"):
            assert_frozen_unchanged(model,model.r4_frozen_baseline_hash)
        counters=dict(runtime.counters)
        key="cf_forward" if mode=="cflora" else "cora_forward"
        if not all(counters.get(t+"_"+key,0)>0 for t in ("q","v")):
            raise RuntimeError(f"R5 adapter did not run through Q/V: {counters}")
        print("[R5-R4-RUNTIME-VERIFIED] "+json.dumps({
            "round":args.round,"method":mode,"counters":counters},
            sort_keys=True),flush=True)
    finally:
        sys.argv=old_argv

if __name__=="__main__":
    main()
