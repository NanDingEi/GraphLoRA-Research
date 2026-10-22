#!/usr/bin/env python3
"""R3 S2 training/evaluation entry, mandatory for R3 adapter checkpoints."""
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


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cfg-path",required=True)
    p.add_argument("--method",choices=("cflora","cora"),required=True)
    a=p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("R3 requires CUDA GPU0")
    cfgpath=Path(a.cfg_path).expanduser().resolve()
    if not cfgpath.is_file():
        raise FileNotFoundError(cfgpath)
    cfg=yaml.safe_load(cfgpath.read_text(encoding="utf-8"))
    model=cfg["model"]
    if (model.get("r3_mode")!=a.method or
            model.get("collab_mode")!="legacy" or
            int(model.get("gcn_layer",-1))!=31):
        raise RuntimeError("R3 config method, legacy mode and L31 must match CLI")
    if not bool(model.get("fix_autocast_cache",False)):
        raise RuntimeError("R3 requires FX autocast cache fix for warm start")
    rt=install_runtime(a.method)
    import train_collm_mf_din as entry
    sys.argv=["train_collm_mf_din.py","--cfg-path",str(cfgpath)]
    print(f"[R3-ENTRY] method={a.method} cfg={cfgpath} "
          "No output-logit modification",flush=True)
    try:
        entry.main()
    finally:
        print("[R3-COUNTERS]",json.dumps(dict(rt.counters)),flush=True)


if __name__=="__main__":
    main()
