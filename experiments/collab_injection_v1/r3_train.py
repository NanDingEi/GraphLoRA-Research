#!/usr/bin/env python3
"""R3-CFLoRA / CoRA inspired L31 graph-conditional adapter trials.

Exactly the original S2 checkpoint and P2B 1-epoch continuation recipe.
Each method: independent 24-step smoke -> LoRA/new adapter change checks
-> one real epoch -> full Valid Small AUC/UAUC. GPU0 foreground only.
No Test, no output-head/logit modification, no source overwrite.
"""
import argparse
import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))

from r2_train import (
    locate_p2b_source,preflight,control_config,sha,repo_head,
    run_to_log,eval_result,check_smoke_lora_update,S2,
)

ARM_MODES={"cflora":"R3_CFLORA","cora":"R3_CORA"}


def smoke_new_params(path,mode):
    import torch
    try:
        ck=torch.load(path,map_location="cpu",mmap=True,weights_only=False)
    except TypeError:
        ck=torch.load(path,map_location="cpu")
    state=ck["model"]
    keys={k:v for k,v in state.items() if k.startswith("r3_adapter.")}
    if not keys:
        raise RuntimeError("R3 adapter parameters missing from smoke checkpoint")
    # Initial CFLoRA output matrix and CoRA B_c are exactly zero.
    changed=(
        [k for k in keys if ("encoder.2.weight" in k or
                            "encoder.2.bias" in k)]
        if mode=="cflora" else
        [k for k in keys if k.endswith("q_up.weight") or
                            k.endswith("v_up.weight")]
    )
    if len(changed)<2 or not all(bool(torch.isfinite(keys[k]).all()) and
                                 bool(torch.count_nonzero(keys[k])>0)
                                 for k in changed):
        raise RuntimeError(f"R3 new adapter failed to update from zero: {changed}")
    print(f"[R3-UPDATE] method={mode} changed_from_zero={changed}",flush=True)
    return {k:tuple(v.shape) for k,v in keys.items()}


def make_config(source,mode,arm,root,steps):
    cfg=control_config(source,mode,arm,root,steps)
    m=cfg["model"]
    m.pop("r2_mode",None)
    m["r3_mode"]=mode
    if m.get("trainable_policy",None) not in (None,"",False):
        raise RuntimeError(
            "R3 matched P2B cannot use an incompatible trainable_policy")
    if m.get("freeze_lora") is not False:
        raise RuntimeError("R3 training must keep original S2 LLM LoRA trainable")
    # New adapter is ALWAYS part of the end-to-end native Yes BCE, never
    # a teacher/logit loss. Original S2 semantic alignment remains as in P2B.
    return cfg


def source_info(args):
    source=locate_p2b_source(args.p2b_source)
    preflight(source)
    y=yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(y,dict):
        raise RuntimeError("Invalid P2B YAML")
    root=Path(args.root).expanduser().resolve()
    make_config(y,args.method,ARM_MODES[args.method],root,None)
    return source,y,root


def anchor_gate(mode):
    path=ROOT/"logs/collab_v1"/f"R3_{mode.upper()}_ANCHOR.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Must complete 32-example zero-init S2 anchor before train: {path}")
    j=json.loads(path.read_text(encoding="utf-8"))
    if j.get("method")!=mode or j.get("status")!="complete" or (
            j.get("samples")!=32):
        raise RuntimeError(f"Invalid R3 {mode} anchor")
    if (float(j.get("max_abs_native_vs_R1",1))>2e-4 or
            float(j.get("max_abs_zero_init_vs_S2",1))>2e-4):
        raise RuntimeError("R3 S2 anchor not preserved")
    if j.get("checkpoint_sha256")!=sha(S2):
        raise RuntimeError("R3 probe used a different S2 checkpoint")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--method",required=True,choices=tuple(ARM_MODES))
    p.add_argument("--root",default="logs/collab_v1/R3_COMPARE_20261022")
    p.add_argument("--p2b-source",default=None)
    p.add_argument("--check-source-only",action="store_true")
    p.add_argument("--prepare-only",action="store_true")
    args=p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
        raise RuntimeError("R3 experiment protocol requires CUDA_VISIBLE_DEVICES=0")
    source,source_yaml,root=source_info(args)
    print(f"[R3-SOURCE-OK] {source} {sha(source)} "
          f"S2_sha256={sha(S2)}",flush=True)
    if args.check_source_only:
        return

    anchor_gate(args.method)
    method=args.method
    arm=ARM_MODES[method]
    folder=root/arm
    if folder.exists():
        raise FileExistsError(f"R3 refuses to overwrite existing arm: {folder}")
    folder.mkdir(parents=True)
    info={
        "status":"prepared","method":method,
        "source_git_commit":repo_head(),
        "original_S2":str(S2),"original_S2_sha256":sha(S2),
        "matched_P2B_yaml":str(source),
        "matched_P2B_sha256":sha(source),
        "epochs":1,"smoke_steps":24,"seed":42,
        "reference_P2B_auc":0.7214924624586214,
        "reference_P2B_uauc":0.6647772423708141,
        "note":"Not full-paper replication; parameter-fusion-only S2 adaptations",
    }
    manifest=folder/"manifest.json"
    def save():
        manifest.write_text(json.dumps(info,indent=2,ensure_ascii=False),
                            encoding="utf-8")

    for label,steps in (("SMOKE",24),("FULL_E1",None)):
        cfg=make_config(source_yaml,method,arm,root,steps)
        yaml_path=folder/f"{label}.yaml"
        yaml_path.write_text(yaml.safe_dump(cfg,sort_keys=False),
                             encoding="utf-8")
        info[f"{label}_sha256"]=sha(yaml_path)
    save()
    print(f"[R3-{method.upper()}-PREPARED] {folder}",flush=True)
    if args.prepare_only:
        return

    try:
        info["status"]="smoke_running";save()
        log=folder/"SMOKE.log"
        run_to_log([sys.executable,"-u",
            "experiments/collab_injection_v1/r3_entry.py",
            "--cfg-path",folder/"SMOKE.yaml","--method",method],log)
        text=log.read_text(encoding="utf-8",errors="replace")
        match=re.search(r"\[R3-COUNTERS\]\s*(\{[^\n]+\})",text)
        if not match:
            raise RuntimeError("R3 smoke did not report module counters")
        counters=json.loads(match.group(1))
        for proj in ("q","v"):
            key=f"{proj}_{'cf_forward' if method=='cflora' else 'cora_forward'}"
            if counters.get(key,0)<1:
                raise RuntimeError(f"R3 new path never executed {key}")
        ckpts=list((ROOT/"output"/root.name/arm/"SMOKE").rglob(
            "checkpoint_0.pth"))
        if len(ckpts)!=1:
            raise RuntimeError(f"Expected exactly one R3 smoke checkpoint: {ckpts}")
        check_smoke_lora_update(ckpts[0])
        info["smoke_adapter_params"]=smoke_new_params(ckpts[0],method)
        info["smoke_checkpoint"]=str(ckpts[0])
        info["smoke_counters"]=counters

        info["status"]="full_running";save()
        log=folder/"FULL_E1.log"
        run_to_log([sys.executable,"-u",
            "experiments/collab_injection_v1/r3_entry.py",
            "--cfg-path",folder/"FULL_E1.yaml","--method",method],log)
        auc,uauc=eval_result(log)
        ckpts=list((ROOT/"output"/root.name/arm/"FULL_E1").rglob(
            "checkpoint_best.pth"))
        if len(ckpts)!=1:
            raise RuntimeError(f"Expected one R3 Valid best checkpoint: {ckpts}")
        info.update({
            "status":"complete",
            "valid_auc":auc,"valid_uauc":uauc,
            "delta_auc_P2B":auc-info["reference_P2B_auc"],
            "delta_uauc_P2B":uauc-info["reference_P2B_uauc"],
            "best_checkpoint":str(ckpts[0]),
        })
        save()
        print(f"[R3-{method.upper()}-RESULT] AUC={auc:.9f} "
              f"UAUC={uauc:.9f} dAUC_P2B={info['delta_auc_P2B']:+.9f} "
              f"dUAUC_P2B={info['delta_uauc_P2B']:+.9f}",flush=True)
    except BaseException as e:
        info.update({"status":"failed","error":repr(e)})
        save()
        print(f"[R3-{method.upper()}-STOP] {e}",flush=True)
        raise


if __name__=="__main__":
    main()
