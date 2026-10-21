#!/usr/bin/env python3
"""Entrypoint for R2 training/eval checkpoints.

This wraps the *original* train_collm_mf_din.py runner in the same Python
process so R2 inference/training hooks are installed before model creation.
DO NOT run an R2-trained checkpoint with vanilla train_collm_mf_din.py;
its config metadata is not by itself sufficient to restore R2 hooks.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"]="0"
ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))

import torch
import yaml
from r2_collab_runtime import install_runtime


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cfg-path",required=True)
    p.add_argument("--mode",required=True)
    a=p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0" or not torch.cuda.is_available():
        raise RuntimeError("R2 requires server GPU0")
    cfg=Path(a.cfg_path).resolve()
    if not cfg.is_file():
        raise FileNotFoundError(cfg)
    data=yaml.safe_load(cfg.read_text(encoding="utf-8"))
    declared=data.get("model",{}).get("r2_mode")
    if declared!=a.mode:
        raise RuntimeError(f"R2 mode mismatch: requested {a.mode}, config {declared}")
    if str(data["model"].get("collab_mode"))!="legacy" or int(
        data["model"].get("gcn_layer",0))!=31:
        raise RuntimeError("R2 only supports original S2 L31 legacy")
    if not bool(data["model"].get("fix_autocast_cache",False)):
        raise RuntimeError("R2 training must use verified FX autocast fix")
    runtime=install_runtime(a.mode)
    import train_collm_mf_din as entry
    sys.argv=["train_collm_mf_din.py","--cfg-path",str(cfg)]
    print(f"[R2-ENTRY] mode={a.mode} source={cfg} no-logit-fusion",flush=True)
    try:
        entry.main()
    finally:
        import json
        print("[R2-COUNTERS]",json.dumps(runtime.state()),flush=True)


if __name__=="__main__":
    main()
