#!/usr/bin/env python3
"""Read-only analysis of S2 joint-optimization checkpoints.

Compares trained model tensors, AdamW step counters, and final GradScaler values
for historical-S2 L31 ON, L31 DETACH, and optional L31 NoGNN arms.

No GPU, no train, no checkpoint modification.
Run only on trusted, locally generated torch checkpoints; torch.load with
weights_only=False can execute pickle code.

Usage:
  python audit_S2_jointopt_checkpoints.py | tee logs/S2_jointopt_audit.log
  python audit_S2_jointopt_checkpoints.py --on PATH --detach PATH --nogNN PATH
"""
import argparse
import collections
import glob
import math
import os
from pathlib import Path

import torch

ROOT = "/home/bbx/user_lx/GraphLoRA"
S2_RUN = "20261017_143611"
DETACH_RUN = "20261017_193815"

def only(pattern, required=True):
    xs = sorted(glob.glob(pattern))
    if not xs:
        if required:
            raise RuntimeError(f"No checkpoint matches: {pattern}")
        return None
    if len(xs)!=1:
        raise RuntimeError(f"Expected one checkpoint, found {len(xs)} for {pattern}: {xs}")
    return xs[0]

def pgroup(k):
    # Categorize SAVED trainable parameters, no optimizer/parameter id guesses.
    if "rec_encoder." in k:
        return "MF_rec_encoder"
    if k.startswith("gcn.") or ".gcn." in k:
        return "GNN"
    if "llama_proj." in k:
        return "Projector"
    if "lora_" in k or ".lora_A." in k or ".lora_B." in k:
        return "LoRA"
    if ".self_attn.q_proj.bias" in k or ".self_attn.v_proj.bias" in k:
        return "Q_V_bias"
    return "Other"

def load(path):
    try: obj=torch.load(path,map_location="cpu",mmap=True,weights_only=False)
    except (TypeError,RuntimeError,ValueError):
        obj=torch.load(path,map_location="cpu",weights_only=False)
    assert isinstance(obj,dict) and isinstance(obj.get("model"),dict)
    assert isinstance(obj.get("optimizer"),dict)
    return obj

def step_of(x):
    if x is None:return None
    if isinstance(x,torch.Tensor):
        if x.numel()!=1:return None
        return int(x.item())
    return int(x)

def optimizer_report(tag,obj):
    optim=obj["optimizer"]
    state=optim.get("state",{})
    seen=set()
    all_steps=[]
    print(f"[S2-AUDIT-OPT] {tag} total_optimizer_state_entries={len(state)}")
    for group in optim.get("param_groups",[]):
        ids=group.get("params",[])
        name=group.get("group_name","<unnamed>")
        present=[int(i) for i in ids if i in state]
        steps=[step_of(state[i].get("step")) for i in present]
        steps=[x for x in steps if x is not None]
        all_steps.extend(steps)
        seen.update(ids)
        freq=collections.Counter(steps)
        counts=",".join(f"{s}:{n}" for s,n in sorted(freq.items()))
        print(f"[S2-AUDIT-GROUP] {tag} {name} tensors={len(ids)} "
              f"optimizer_states={len(present)} missing_states={len(ids)-len(present)} "
              f"steps={counts or '<none>'} "
              f"lr_scale={group.get('lr_scale')} "
              f"weight_decay={group.get('weight_decay')}")
    if all_steps:
        c=collections.Counter(all_steps)
        print(f"[S2-AUDIT-STEPS] {tag} "
              f"min={min(all_steps)} max={max(all_steps)} "
              f"frequencies={dict(sorted(c.items()))}")
    else:
        print(f"[S2-AUDIT-STEPS] {tag} NO_ADAM_STEPS")
    scaler=obj.get("scaler")
    if isinstance(scaler,dict):
        print(f"[S2-AUDIT-SCALER] {tag} scale={scaler.get('scale')} "
              f"growth_tracker={scaler.get('_growth_tracker')} "
              f"growth_interval={scaler.get('growth_interval')}")
    else:
        print(f"[S2-AUDIT-SCALER] {tag} {scaler}")

def compare(a,b,tag_a,tag_b):
    ma,mb=a["model"],b["model"]
    ka,kb=set(ma),set(mb)
    print(f"[S2-AUDIT-KEYS] {tag_a}->{tag_b} "
          f"shared={len(ka&kb)} only_{tag_a}={len(ka-kb)} only_{tag_b}={len(kb-ka)}")
    for name,keys in ((tag_a,ka-kb),(tag_b,kb-ka)):
        for x in sorted(keys)[:15]:print(f"[S2-AUDIT-ONLY] {name} {x}")
    stats=collections.defaultdict(lambda:{"total":0,"changed":0,"same":0,
                                         "numel":0,"max_abs":0.0,
                                         "sum_diff_sq":0.0,"sum_norm_sq":0.0,
                                         "nonfinite":0})
    top=[]
    for key in sorted(ka&kb):
        x,y=ma[key],mb[key]
        if not isinstance(x,torch.Tensor) or not isinstance(y,torch.Tensor):
            continue
        group=pgroup(key)
        q=stats[group]
        q["total"]+=1
        if x.shape != y.shape:
            print(f"[S2-AUDIT-SHAPE-MISMATCH] {key} {tuple(x.shape)} {tuple(y.shape)}")
            continue
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            q["nonfinite"]+=1
            print(f"[S2-AUDIT-NONFINITE] {key}")
            continue
        xx,yy=x.float(),y.float()
        d=xx-yy
        eq=bool(torch.equal(x,y))
        q["same" if eq else "changed"]+=1
        q["numel"]+=x.numel()
        ds=float((d.double()**2).sum().item())
        ns=float((xx.double()**2).sum().item())
        q["sum_diff_sq"]+=ds
        q["sum_norm_sq"]+=ns
        max_abs=float(d.abs().max().item()) if d.numel() else 0.0
        q["max_abs"]=max(q["max_abs"],max_abs)
        if not eq:
            top.append((math.sqrt(ds)/max(math.sqrt(ns),1e-12),
                        math.sqrt(ds),max_abs,key))
    for g,q in sorted(stats.items()):
        relative=math.sqrt(q["sum_diff_sq"])/max(math.sqrt(q["sum_norm_sq"]),1e-12)
        absolute=math.sqrt(q["sum_diff_sq"])
        print(f"[S2-AUDIT-PARAM] {tag_a}->{tag_b} module={g} "
              f"shared_tensors={q['total']} equal={q['same']} changed={q['changed']} "
              f"numel={q['numel']} delta_l2={absolute:.9g} "
              f"relative_delta_l2={relative:.9g} max_abs_delta={q['max_abs']:.9g} "
              f"nonfinite_tensors={q['nonfinite']}")
    for rel,delta,mx,key in sorted(top,reverse=True)[:20]:
        print(f"[S2-AUDIT-LARGEST] {tag_a}->{tag_b} "
              f"relative={rel:.8g} delta_l2={delta:.8g} max_abs={mx:.8g} {key}")

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("--on")
    pa.add_argument("--detach")
    pa.add_argument("--nogNN")
    args=pa.parse_args()
    paths={
        "ON":args.on or only(f"{ROOT}/logs/S2_STRICT_{S2_RUN}/L31_output/*/checkpoint_best.pth"),
        "DETACH":args.detach or only(f"{ROOT}/logs/S2_STRICT_DETACH_{DETACH_RUN}/DETACH_GNN_output/*/checkpoint_best.pth"),
    }
    paths["NOGNN"]=args.nogNN or only(f"{ROOT}/logs/S2_STRICT_NOGNN_*/NO_GNN_output/*/checkpoint_best.pth",required=False)
    objs={}
    for name,path in paths.items():
        if not path:
            print(f"[S2-AUDIT-WARN] {name} checkpoint not found; skipping")
            continue
        obj=load(path)
        cfg=obj["config"]
        model_cfg=cfg.get("model",{})
        run_cfg=cfg.get("run",{})
        print(f"[S2-AUDIT-FILE] {name} path={path} epoch={obj.get('epoch')} "
              f"gcn_layer={model_cfg.get('gcn_layer')} "
              f"gnn_scale={model_cfg.get('gnn_scale')} "
              f"rec_loss_updates_gnn={model_cfg.get('train_gnn_with_rec_loss')} "
              f"seed={run_cfg.get('seed')},train_batch={run_cfg.get('batch_size_train')} "
              f"accum={run_cfg.get('accum_grad_iters')},amp={run_cfg.get('amp')}")
        assert obj.get("epoch")==0,"Expected epoch0 checkpoint; stop if unexpected"
        assert model_cfg.get("gcn_layer")==31,"This audit expects L31"
        optimizer_report(name,obj)
        objs[name]=obj
    compare(objs["ON"],objs["DETACH"],"ON","DETACH")
    if "NOGNN" in objs:
        compare(objs["ON"],objs["NOGNN"],"ON","NOGNN")
        compare(objs["DETACH"],objs["NOGNN"],"DETACH","NOGNN")
    print("[S2-AUDIT-DONE] Read-only checkpoint state/parameter comparison; no training")

if __name__=="__main__":
    main()
