#!/usr/bin/env python3
"""R4: isolate CoRA/CFLoRA add-on effects from legacy S2 optimization drift.

R4-01: CFLoRA new adapter only.
R4-02: CoRA new adapter only.
R4-03: CoRA new adapter + L31 V LoRA-B with 0.1x LR.
All three start from the ORIGINAL S2 champion (not each other).
The native BCE drives the new collaborative adapters; no logits edits.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
import torch
import yaml

ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))

from r4_freeze import MODES,V31_NAME
from r2_train import (locate_p2b_source,preflight,control_config,
                       sha,repo_head,run_to_log,eval_result,S2)
from r3_train import smoke_new_params

ARMS={"01":"R4_01_CFLORA_ONLY",
      "02":"R4_02_CORA_ONLY",
      "03":"R4_03_CORA_V31B"}


def make_config(source,round_id,root,steps):
    mode,local_v=MODES[round_id]
    arm=ARMS[round_id]
    cfg=control_config(source,mode,arm,root,steps)
    m,r=cfg["model"],cfg["run"]
    m.pop("r2_mode",None)
    if m.get("trainable_policy") not in (None,"",False):
        raise RuntimeError("R4 source must not activate previous frozen/fusion policy")
    # Keep original S2 LoRA forward active. Control requires_grad after
    # from_config loads its source checkpoint (r4_entry.py).
    if m.get("freeze_lora") is not False:
        raise RuntimeError("R4 requires S2 forward LoRA enabled")
    m["r3_mode"]=mode
    m["r4_round"]=round_id
    # RunnerBase groups r3_adapter.* into OTHER and LoRA-B into LORA,
    # with each group preserving its own lr_scale throughout the scheduler.
    r["lr_scale_other"]=1.0
    r["lr_scale_lora"]=0.1 if local_v else 1.0
    r["lr_scale_rec"]=r.get("lr_scale_rec",1.0)
    r["lr_scale_gnn"]=r.get("lr_scale_gnn",1.0)
    r["lr_scale_proj"]=r.get("lr_scale_proj",1.0)
    # Other training losses/optimizer/scheduler/dataset/seed are unchanged.
    return cfg


def check_anchor(round_id):
    file=ROOT/"logs/collab_v1"/f"R4_{round_id}_FROZEN_ANCHOR.json"
    if not file.is_file():
        raise RuntimeError(
            f"R4 frozen-model 32-example probe is required before training: {file}")
    data=json.loads(file.read_text(encoding="utf-8"))
    if data.get("status")!="complete" or data.get("round")!=round_id or (
        data.get("original_s2_sha256")!=sha(S2)):
        raise RuntimeError("R4 anchor does not match original S2/round")
    if (data.get("max_abs_S2_R1",1)>2e-4 or
            data.get("max_abs_zero_init",1)>2e-4):
        raise RuntimeError("R4 initial S2 equivalence failed")
    return data


def check_v31_updated(smoke_ckpt):
    """For R4-03 only: V31 LoRA-B must update; all other LoRA stays frozen."""
    try:
        state=torch.load(smoke_ckpt,map_location="cpu",mmap=True,
                         weights_only=False)["model"]
        original=torch.load(S2,map_location="cpu",mmap=True,
                            weights_only=False)["model"]
    except TypeError:
        state=torch.load(smoke_ckpt,map_location="cpu")["model"]
        original=torch.load(S2,map_location="cpu")["model"]
    matched=[n for n in state if V31_NAME.search(n)]
    if len(matched)!=1 or matched[0] not in original:
        raise RuntimeError(f"R4-03 invalid L31 V-B trainable checkpoint: {matched}")
    name=matched[0]
    if torch.equal(state[name],original[name]):
        raise RuntimeError("R4-03 L31 V-B did not update")
    print(f"[R4-03-V-B-UPDATE-OK] {name}",flush=True)
    del state,original
    return name


def verify_smoke(log,ckpt,round_id,mode):
    txt=log.read_text(encoding="utf-8",errors="replace")
    if f"[R4-FREEZE] " not in txt or "[R4-FROZEN-HASH-OK]" not in txt:
        raise RuntimeError("R4 smoke missing freeze or frozen-weight integrity check")
    match=re.search(r"\[R4-FREEZE\]\s*(\{[^\n]+\})",txt)
    if not match:
        raise RuntimeError("R4 frozen parameter manifest missing")
    actual=json.loads(match.group(1))
    if actual.get("round")!=round_id:
        raise RuntimeError("R4 wrong frozen round loaded")
    allow=actual.get("trainable_names",[])
    # Never permit anything outside adapter plus optional exactly-one V31-B.
    for name in allow:
        if not name.startswith("r3_adapter.") and not (
                round_id=="03" and V31_NAME.search(name)):
            raise RuntimeError(f"R4 nonisolated trainable parameter: {name}")
    local=[n for n in allow if V31_NAME.search(n)]
    if (round_id=="03") != (len(local)==1):
        raise RuntimeError("R4 local V31-B incorrectly frozen/unfrozen")
    if not allow or actual.get("trainable_params",0)<1:
        raise RuntimeError("R4 no trainable parameters")
    cm=re.search(r"\[R4-COUNTERS\]\s*(\{[^\n]+\})",txt)
    if not cm:
        raise RuntimeError("R4 runtime counters missing")
    counters=json.loads(cm.group(1))
    for tag in ("q","v"):
        key=f"{tag}_{'cf_forward' if mode=='cflora' else 'cora_forward'}"
        if counters.get(key,0)<=0:
            raise RuntimeError(f"R4 mechanism did not execute {key}")
    tensors=smoke_new_params(ckpt,mode)
    if round_id=="03":
        check_v31_updated(ckpt)
    return actual,counters,tensors


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--round",required=True,choices=tuple(MODES))
    p.add_argument("--root",default="logs/collab_v1/R4_ISOLATION_20261022")
    p.add_argument("--p2b-source",default=None)
    p.add_argument("--check-source-only",action="store_true")
    p.add_argument("--prepare-only",action="store_true")
    args=p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
        raise RuntimeError("R4 requires CUDA_VISIBLE_DEVICES=0")
    source_path=locate_p2b_source(args.p2b_source)
    preflight(source_path)
    source_yaml=yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(source_yaml,dict):
        raise RuntimeError("Invalid reference P2B config")
    root=Path(args.root).expanduser().resolve()
    make_config(source_yaml,args.round,root,None)
    print(f"[R4-SOURCE-OK] S2={sha(S2)} P2B={sha(source_path)} "
          f"reference={source_path}",flush=True)
    if args.check_source_only:
        return

    anchor=check_anchor(args.round)
    arm=ARMS[args.round]
    folder=root/arm
    if folder.exists():
        raise FileExistsError(f"R4 refuses overwrite of {folder}")
    folder.mkdir(parents=True)
    mode,local=MODES[args.round]
    status={
        "status":"prepared","round":args.round,"method":mode,
        "l31_v_lora_B_unfrozen":bool(local),
        "source_commit":repo_head(),
        "s2_checkpoint":str(S2),"s2_sha256":sha(S2),
        "p2b_config":str(source_path),"p2b_sha256":sha(source_path),
        "seed":42,"epochs":1,"batch_size":3,
        "new_adapter_lr":2e-5,
        "l31_v_B_lr":2e-6 if local else 0.0,
        "baseline_s2_auc":0.7296752596287361,
        "baseline_s2_uauc":0.6841670211619721,
        "p2b_auc":0.7214924624586214,
        "p2b_uauc":0.6647772423708141,
        "anchor":anchor,
    }
    manifest=folder/"manifest.json"
    def save():
        manifest.write_text(json.dumps(status,indent=2,ensure_ascii=False),
                            encoding="utf-8")
    for label,steps in (("SMOKE",24),("FULL_E1",None)):
        cfg=make_config(source_yaml,args.round,root,steps)
        filename=folder/(label+".yaml")
        filename.write_text(yaml.safe_dump(cfg,sort_keys=False),
                            encoding="utf-8")
        status[label+"_yaml_sha256"]=sha(filename)
    save()
    if args.prepare_only:
        print(f"[R4-{args.round}-PREPARED] {folder}",flush=True)
        return

    try:
        status["status"]="smoke_running";save()
        smoke=folder/"SMOKE.log"
        run_to_log([sys.executable,"-u",
            "experiments/collab_injection_v1/r4_entry.py",
            "--cfg-path",folder/"SMOKE.yaml","--round",args.round],smoke)
        checkpoints=list((ROOT/"output"/root.name/arm/"SMOKE").rglob(
            "checkpoint_0.pth"))
        if len(checkpoints)!=1:
            raise RuntimeError(f"R4 smoke checkpoint missing/ambiguous {checkpoints}")
        trainable,counters,shapes=verify_smoke(
            smoke,checkpoints[0],args.round,mode)
        if tuple(sorted(trainable["trainable_names"]))!=tuple(
                sorted(anchor["trainable_tensors"])):
            raise RuntimeError("R4 smoke trainable-name whitelist differs from anchor")
        status.update({
            "smoke_checkpoint":str(checkpoints[0]),
            "smoke_trainable":trainable,
            "smoke_adapter_shapes":shapes,
            "smoke_forward_counters":counters,
            "status":"full_running"})
        save()

        log=folder/"FULL_E1.log"
        run_to_log([sys.executable,"-u",
            "experiments/collab_injection_v1/r4_entry.py",
            "--cfg-path",folder/"FULL_E1.yaml","--round",args.round],log)
        if "[R4-FROZEN-HASH-OK]" not in log.read_text(encoding="utf-8",
                                                    errors="replace"):
            raise RuntimeError("R4 full run failed frozen S2 integrity check")
        auc,uauc=eval_result(log)
        checkpoints=list((ROOT/"output"/root.name/arm/"FULL_E1").rglob(
            "checkpoint_best.pth"))
        if len(checkpoints)!=1:
            raise RuntimeError(f"R4 best Valid checkpoint missing: {checkpoints}")
        status.update({
            "status":"complete",
            "valid_auc":auc,"valid_uauc":uauc,
            "delta_auc_original_s2":auc-status["baseline_s2_auc"],
            "delta_uauc_original_s2":uauc-status["baseline_s2_uauc"],
            "delta_auc_P2B":auc-status["p2b_auc"],
            "delta_uauc_P2B":uauc-status["p2b_uauc"],
            "best_checkpoint":str(checkpoints[0]),
        })
        save()
        print(f"[R4-{args.round}-RESULT] method={mode} "
              f"AUC={auc:.9f} UAUC={uauc:.9f} "
              f"dAUC_S2={status['delta_auc_original_s2']:+.9f} "
              f"dUAUC_S2={status['delta_uauc_original_s2']:+.9f}",
              flush=True)
    except BaseException as exc:
        status.update({"status":"failed","error":repr(exc)})
        save()
        print(f"[R4-{args.round}-STOP] {exc}",flush=True)
        raise


if __name__=="__main__":
    main()
