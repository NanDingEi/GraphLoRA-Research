#!/usr/bin/env python3
"""P2-A: one short FIXED-cache training smoke vs existing unfixed P0 smoke.

Original P0_CONTROL_SMOKE.yaml / model checkpoint are REUSED UNCHANGED.
Generate exactly one new config differing in:
  model.fix_autocast_cache: false -> true
  run.output_dir: a new isolated run folder
Run 24 train steps (same as P0 control smoke); compare each LoRA layer's
saved parameter updates against original champion. No Valid/Test pass.

Modes: --stage prepare | compare. Launch training in foreground MANUALLY
between them. No background jobs, no extra full epoch, no GPU inference in
this script itself. CPU-only checkpoint comparisons.
"""
import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path

import torch
import yaml

ROOT=Path(__file__).resolve().parents[2]
RUN_DEFAULT=ROOT/"logs/collab_v1/S2_L30_AUX_P0_20261021_165700"
LAYER_RE=re.compile(r"(?:^|\.)layers\.(\d+)\.")
LORA_RE=re.compile(r"(?:^|\.)(lora_A|lora_B)(?:\.|$)")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(4*1024*1024),b""):
            h.update(c)
    return h.hexdigest()


def load(path):
    path=Path(path).resolve()
    try:
        obj=torch.load(str(path),map_location="cpu",mmap=True,weights_only=False)
    except TypeError:
        obj=torch.load(str(path),map_location="cpu")
    if not isinstance(obj,dict) or not isinstance(obj.get("model"),dict):
        raise ValueError(f"Invalid checkpoint: {path}")
    return obj["model"]


def paths(run):
    state=json.loads((run/"state.json").read_text(encoding="utf-8"))
    champion=Path(state["champion"]).resolve()
    if not champion.is_file() or sha(champion)!=state["champion_sha256"]:
        raise RuntimeError("S2 champion missing or checksum changed")
    prev=run/"P0_CONTROL_SMOKE.yaml"
    if not prev.is_file():
        raise FileNotFoundError(prev)
    original=yaml.safe_load(prev.read_text(encoding="utf-8"))
    cfg=run/"P2A_FIX_CACHE_SMOKE.yaml"
    newdir=Path(state["output_dir"]).resolve()/"P2A_FIX_CACHE_SMOKE"
    olddir=Path(original["run"]["output_dir"]).resolve()
    old_ckpts=list(olddir.rglob("checkpoint_0.pth"))
    if len(old_ckpts)!=1:
        raise RuntimeError("Expected exactly one original S2 P0 control smoke checkpoint")
    if (int(original["run"].get("max_train_steps",-1)) !=
            int(state["smoke_steps"])):
        raise RuntimeError("Original smoke steps mismatch original state")
    model=original["model"]
    if (model.get("collab_mode")!="legacy" or
            int(model.get("gcn_layer",-1))!=31 or
            abs(float(model.get("gnn_scale",-1))-0.1)>1e-10 or
            not bool(model.get("gnn_semantic_align",False)) or
            bool(model.get("freeze_lora",False)) or
            bool(model.get("fix_autocast_cache",False)) or
            float(model.get("l30_aux_weight",0))!=0 or
            Path(model["ckpt"]).resolve()!=champion):
        raise RuntimeError("Original P0 control smoke is not unpatched S2")
    if int(original["run"].get("seed",-1))!=int(state["seed"]):
        raise RuntimeError("Seed mismatch in original control smoke")
    return state,champion,original,old_ckpts[0],cfg,newdir


def prepare(run):
    state,champion,original,old_ckpt,cfg,newdir=paths(run)
    if cfg.exists() or newdir.exists():
        raise RuntimeError("Fixed smoke config/output already exists. "
                           "Do not overwrite or restart blindly. Run --stage compare "
                           "if it has completed.")
    fixed=copy.deepcopy(original)
    fixed["model"]["fix_autocast_cache"]=True
    fixed["run"]["output_dir"]=str(newdir)
    # Don't silently change LR, random seed, scheduler, accumulation or
    # the exact 24-step smoke horizon.
    fixed_check=copy.deepcopy(fixed)
    fixed_check["model"].pop("fix_autocast_cache")
    fixed_check["run"]["output_dir"]=original["run"]["output_dir"]
    if fixed_check!=original:
        raise AssertionError("New fixed smoke differs from original beyond 2 keys")
    cfg.write_text(yaml.safe_dump(fixed,sort_keys=False),
                   encoding="utf-8")
    print(f"[P2A-READY] original_control_smoke={old_ckpt}",flush=True)
    print(f"[P2A-READY] fixed_config={cfg}",flush=True)
    print(f"[P2A-READY] planned_output={newdir}",flush=True)
    print(f"[P2A-READY] steps={state['smoke_steps']} "
          f"seed={state['seed']} same S2 checkpoint; only autocast-cache flag differs",
          flush=True)


def compare(run):
    state,champion,original,old_ckpt,cfg,newdir=paths(run)
    if not cfg.exists():
        raise FileNotFoundError("First execute --stage prepare")
    fixed=yaml.safe_load(cfg.read_text(encoding="utf-8"))
    reverted=copy.deepcopy(fixed)
    reverted["model"].pop("fix_autocast_cache",None)
    reverted["run"]["output_dir"]=original["run"]["output_dir"]
    if (fixed["model"].get("fix_autocast_cache") is not True or
            reverted!=original):
        raise RuntimeError("Fixed smoke YAML changed beyond two audited keys")
    fix_ckpts=list(newdir.rglob("checkpoint_0.pth"))
    if len(fix_ckpts)!=1:
        raise RuntimeError(f"Expected exactly one fixed smoke checkpoint: {fix_ckpts}")
    base,old,fix=load(champion),load(old_ckpt),load(fix_ckpts[0])
    keys=set(base)&set(old)&set(fix)
    # Every original and updated LoRA parameter should be present.
    key_counts=[
        {k for k in d if LORA_RE.search(k)}
        for d in (base,old,fix)
    ]
    if any(ks!=key_counts[0] for ks in key_counts) or not key_counts[0]:
        raise RuntimeError("LoRA tensor coverage mismatch across checkpoint states")
    result={}
    for key in sorted(key_counts[0]):
        match=LAYER_RE.search(key)
        if not match:
            raise RuntimeError(f"Unparseable LoRA layer key: {key}")
        layer=int(match.group(1))
        b,o,f=(d[key] for d in (base,old,fix))
        if b.shape!=o.shape or b.shape!=f.shape:
            raise RuntimeError(f"Shape mismatch in {key}")
        if not (torch.isfinite(b).all() and torch.isfinite(o).all()
                and torch.isfinite(f).all()):
            raise RuntimeError(f"Non-finite LoRA parameter {key}")
        # CPU only, just per tensor FP32 vector norms.
        dc=(o.float()-b.float()).norm().item()
        df=(f.float()-b.float()).norm().item()
        gap=(f.float()-o.float()).norm().item()
        row=result.setdefault(layer,{"tensors":0,"old_changed":0,
                     "fixed_changed":0,"old_sq":0.0,"fixed_sq":0.0,"gap_sq":0.0})
        row["tensors"]+=1
        row["old_changed"]+=int(dc!=0)
        row["fixed_changed"]+=int(df!=0)
        row["old_sq"]+=dc**2
        row["fixed_sq"]+=df**2
        row["gap_sq"]+=gap**2
    if sorted(result)!=list(range(32)):
        raise RuntimeError(f"Expected L0-L31, got {sorted(result)}")
    layers={}
    for layer,row in sorted(result.items()):
        layers[f"L{layer:02d}"]={
            "tensors":row["tensors"],
            "old_changed":row["old_changed"],
            "fixed_changed":row["fixed_changed"],
            "old_norm":math.sqrt(row["old_sq"]),
            "fixed_norm":math.sqrt(row["fixed_sq"]),
            "fixed_minus_old_norm":math.sqrt(row["gap_sq"]),
        }
    print(f"[P2A-CHECK] same seed={state['seed']} "
          f"smoke_steps={state['smoke_steps']} "
          "base=S2; original=fix_autocast_cache:false; fixed=true",flush=True)
    for layer,row in layers.items():
        print(f"[P2A-LAYER] {layer} before={row['old_changed']}/{row['tensors']} "
              f"after={row['fixed_changed']}/{row['tensors']} "
              f"old_norm={row['old_norm']:.8g} "
              f"fixed_norm={row['fixed_norm']:.8g} "
              f"gap={row['fixed_minus_old_norm']:.8g}",flush=True)
    first_eight=[layers[f"L{x:02d}"] for x in range(8)]
    recovered=all(x["fixed_changed"]>0 for x in first_eight)
    original_zero=all(x["old_changed"]==0 for x in first_eight)
    print(f"[P2A-EARLY] original_all_zero={original_zero} "
          f"fixed_every_layer_updated={recovered}",flush=True)
    report={
        "source_s2":str(champion),"original_smoke":str(old_ckpt),
        "fixed_smoke":str(fix_ckpts[0]),
        "fixed_config":str(cfg),"same_seed":state["seed"],
        "same_smoke_steps":state["smoke_steps"],
        "original_first_eight_all_zero":original_zero,
        "fixed_first_eight_all_updated":recovered,
        "layers":layers,
        "note":"Parameter restoration supports the cache hypothesis; "
               "it is NOT equivalent to a direct gradient check. "
               "No Valid or Test metrics were measured.",
    }
    dest=run/"P2A_AUTOCAST_CACHE_PROBE.json"
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),
                    encoding="utf-8")
    print(f"[P2A-COMPLETE] {dest}",flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=("prepare","compare"),required=True)
    p.add_argument("--run-dir",default=str(RUN_DEFAULT))
    a=p.parse_args()
    run=Path(a.run_dir).expanduser().resolve()
    if a.stage=="prepare":
        prepare(run)
    else:
        compare(run)


if __name__=="__main__":
    main()
