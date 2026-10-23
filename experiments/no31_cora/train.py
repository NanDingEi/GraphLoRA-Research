#!/usr/bin/env python3
"""Use original NO31 CTR/KD model, optionally attach the CoRA module.

Example (foreground on GPU0):
  CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora/train.py \
      --mode cora --cfg-path logs/NO31_CORA_20261023/CORA.yaml
"""
import argparse
import os
from pathlib import Path
import sys
import json

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def main():
    args=argparse.ArgumentParser()
    args.add_argument("--cfg-path",required=True)
    args.add_argument("--mode",choices=("cora","control"),default="cora")
    args.add_argument("--cora-gain",type=float,default=.1)
    args.add_argument("--allow-cora-to-update-gnn",action="store_true")
    opt=args.parse_args()
    cfg=Path(opt.cfg_path).resolve()
    if not cfg.is_file():
        raise FileNotFoundError(cfg)
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
        raise RuntimeError("NO31 CoRA must use CUDA_VISIBLE_DEVICES=0")
    os.environ["S2_INPUT_ALPHA"]="0"
    os.environ["S2_SMALL_KD"]="0.05"
    os.environ["PYTHONUNBUFFERED"]="1"
    os.chdir(ROOT)
    sys.path[:0]=[str(HERE),str(ROOT)]

    import yaml
    c=yaml.safe_load(cfg.read_text())
    m,r=c["model"],c["run"]
    if (float(m["gnn_scale"])!=0 or int(m["gcn_layer"])!=31 or
        float(m["gnn_conf_loss_weight"])!=.05 or
        not m["gnn_semantic_align"] or
        not m["train_gnn_with_rec_loss"] or
        m["rec_model"]!="MF" or m["lora_gnn"]!="ngcf"):
        raise RuntimeError("Invalid original NO31 CTR/KD model controls")
    if (r["seed"]!=42 or r["max_epoch"]!=3 or
        int(r.get("no31_stop_after_epochs",0)) not in (1,2) or
        int(r["accum_grad_iters"])!=3 or int(r["batch_size_train"])!=3):
        raise RuntimeError("Unexpected original NO31 seed/schedule/batch")
    if m.get("ckpt") or r.get("resume_ckpt_path"):
        raise RuntimeError("NO31 starts fresh with pretrained MF, not S2/R4 checkpoints")
    if not Path(m["rec_config"]["pretrained_path"]).is_file():
        raise FileNotFoundError("Original pretrained MF checkpoint missing")
    if float(m.get("proj_reg_weight",0))!=.0005 or float(m["gnn_semantic_weight"])!=.0001:
        raise RuntimeError("Original S2 semantic/projector losses not preserved")
    runtime=None
    if opt.mode=="cora":
        from runtime import install
        runtime=install(gain=opt.cora_gain,
                        detach_gnn=not opt.allow_cora_to_update_gnn)
    print(f"[NO31-CORA-CONFIG] mode={opt.mode} seed=42 scale=0 "
          f"alpha=0 CTR=.05 KD=.05 epochs={r['no31_stop_after_epochs']} "
          f"cora_gain={opt.cora_gain} detach_gnn={not opt.allow_cora_to_update_gnn} "
          f"source={cfg}",flush=True)
    import train_collm_mf_din
    sys.argv=["train_collm_mf_din.py","--cfg-path",str(cfg)]
    try:
        train_collm_mf_din.main()
    finally:
        if runtime is not None:
            data=dict(runtime.counters)
            print("[NO31-CORA-COUNTERS] "+json.dumps(data,sort_keys=True),flush=True)
            if (data.get("q_cora_forward",0)==0 or
                data.get("v_cora_forward",0)==0):
                raise RuntimeError("NO31 CoRA Q/V forward paths were not used")


if __name__=="__main__":
    main()
