#!/usr/bin/env python3
"""Audit the REAL S2 checkpoint config & model parameter keys against P3.
Pure CPU metadata audit; no training, imports of model code, or checkpoint modification.
Run on your own trusted checkpoints only (torch.load(weights_only=False)).
"""
import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import torch

S2 = "/home/bbx/user_lx/GraphLoRA/output/S2_groups_acc3/20261010223/checkpoint_best.pth"
P3 = "/home/bbx/user_lx/GraphLoRA/minigpt4/output/LAYER_P3_L31_BS3/20261017102/checkpoint_best.pth"

def load(path):
    p=Path(path)
    if not p.is_file():
        raise SystemExit(f"[AUDIT-ERROR] checkpoint missing: {p}")
    try:
        ckpt=torch.load(p, map_location="cpu", mmap=True, weights_only=False)
    except (TypeError, RuntimeError, ValueError):
        ckpt=torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(ckpt,Mapping) or not isinstance(ckpt.get("config"),Mapping):
        raise SystemExit(f"[AUDIT-ERROR] checkpoint does not have a config dict: {p}")
    return ckpt

def flat(x, prefix=""):
    out={}
    if isinstance(x,Mapping):
        for k,v in x.items():out.update(flat(v, prefix+"."+str(k) if prefix else str(k)))
    else:
        out[prefix]=x
    return out

def text(x):
    try:return json.dumps(x,sort_keys=True,ensure_ascii=False,default=str)
    except Exception:return repr(x)

def describe(label,ckpt):
    c=flat(ckpt["config"])
    m=ckpt.get("model",{})
    qvbias=sorted(k for k in m if (".layers.31.self_attn.q_proj.bias" in k or
                                  ".layers.31.self_attn.v_proj.bias" in k))
    qvnames=sorted(k for k in m if ".layers.31.self_attn." in k and
                   (".q_proj." in k or ".v_proj." in k))
    print(f"[AUDIT-{label}] epoch={ckpt.get('epoch')} model_tensors={len(m)} "
          f"layer31_extra_qv_bias_count={len(qvbias)}")
    for k in qvbias: print(f"[AUDIT-{label}-BIAS] {k} shape={list(m[k].shape)}")
    for key in ("model.gcn_layer","model.preserve_source_qv_bias",
                "model.d1_mode","model.gnn_semantic_align",
                "model.gnn_semantic_weight","model.train_gnn_with_rec_loss",
                "model.fix_autocast_cache","model.freeze_rec","model.freeze_proj",
                "model.freeze_lora","model.lora_gnn","model.gnn_scale",
                "model.proj_reg_weight","model.proj_reg_items",
                "run.batch_size_train","run.accum_grad_iters","run.warmup_steps",
                "run.max_epoch","run.stop_after_epoch","run.seed","run.init_lr",
                "run.min_lr","run.lr_scale_gnn","run.lr_scale_rec",
                "run.lr_scale_proj","run.lr_scale_lora","run.weight_decay"):
        print(f"[AUDIT-{label}-CFG] {key} = {text(c.get(key,'<ABSENT>'))}")
    return c

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("--s2", default=S2)
    pa.add_argument("--p3", default=P3)
    args=pa.parse_args()
    s2=load(args.s2)
    p3=load(args.p3)
    a=describe("S2",s2)
    b=describe("P3",p3)
    print("[AUDIT-DIFF] Every effective checkpoint config difference:")
    for k in sorted(set(a)|set(b)):
        av,bv=a.get(k,"<ABSENT>"),b.get(k,"<ABSENT>")
        if text(av)!=text(bv):
            print(f"[AUDIT-DIFF] {k}: S2={text(av)} P3={text(bv)}")
    print("[AUDIT-DONE] Read-only metadata comparison; no training/test.")

if __name__=="__main__":
    main()
