#!/usr/bin/env python3
"""R4 requires runtime parameter isolation AFTER loading original S2.

Use for both training and checkpoint evaluation. Never use vanilla
train_collm_mf_din.py to evaluate the frozen-adapter R4 checkpoints.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"]="0"
ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))

import torch
import yaml
from r3_paper_adapters import install_runtime
from r4_freeze import MODES,install_freeze,assert_frozen_unchanged


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cfg-path",required=True)
    p.add_argument("--round",choices=tuple(MODES),required=True)
    args=p.parse_args()
    path=Path(args.cfg_path).resolve()
    if not path.is_file() or not torch.cuda.is_available():
        raise RuntimeError("R4 requires an existing config and CUDA GPU0")
    cfg=yaml.safe_load(path.read_text(encoding="utf-8"))
    m,r=cfg["model"],cfg["run"]
    mode,local_v=MODES[args.round]
    if m.get("r4_round")!=args.round or m.get("r3_mode")!=mode:
        raise RuntimeError("R4 runtime/config method mismatch")
    if m.get("collab_mode")!="legacy" or int(m.get("gcn_layer",-1))!=31:
        raise RuntimeError("R4 requires original S2 L31 legacy mode")
    if not bool(m.get("fix_autocast_cache",False)):
        raise RuntimeError("R4 requires original P2B autocast cache fix")
    if abs(float(r.get("lr_scale_other",1))-1.0)>1e-12:
        raise RuntimeError("R4 adapter LR multiplier must be 1")
    if args.round=="03" and abs(float(r.get("lr_scale_lora",0))-.1)>1e-12:
        raise RuntimeError("R4-03 L31 V-B LR multiplier must be 0.1")
    rt=install_runtime(mode)
    install_freeze(args.round)

    import train_collm_mf_din as entry
    sys.argv=["train_collm_mf_din.py","--cfg-path",str(path)]
    print(f"[R4-ENTRY] round={args.round} method={mode} "
          f"train_local_L31_V_B={bool(local_v)}",flush=True)
    try:
        entry.main()
    finally:
        model=rt._injected_model
        if model is not None and hasattr(model,"r4_frozen_baseline_hash"):
            assert_frozen_unchanged(model,model.r4_frozen_baseline_hash)
        print("[R4-COUNTERS] "+json.dumps(dict(rt.counters)),flush=True)


if __name__=="__main__":
    main()
