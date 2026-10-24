#!/usr/bin/env python3
"""Train four CoRA configurations for two total epochs on the existing NO31 worktree.

N1-N3 resume the corresponding epoch-0 checkpoint for epoch 1.
N4 is a fresh V/O, alpha=48 run for epochs 0 and 1.
GPU0 only, foreground and serial; Valid only; no Test evaluation.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime

import torch
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.no31_cora_original import night8 as core

ALL="q_proj,k_proj,v_proj,o_proj"
R2_CKPT=ROOT/"output/NO31_CORA_NIGHT8_20261024_012519_960783/R2/20261024024/checkpoint_0.pth"
R7_CKPT=ROOT/"output/NO31_CORA_NIGHT8_20261024_012519_960783/R7/20261024092/checkpoint_0.pth"
F4_CKPT=ROOT/"output/NO31_CORA_FOLLOWUP8_20261024_130326_269104/F4/20261024170/checkpoint_0.pth"

EXPERIMENTS=[
    dict(id="N1",title="Resume R2 V/O alpha32 to total 2 epochs",
         projections="v_proj,o_proj",layers="0-31",cf_input="concat",
         alpha=32,p_lr=1.0,seed=42,mode="resume",
         checkpoint=str(R2_CKPT),
         source_meta="logs/NO31_CORA_NIGHT8_20261024_012519_960783/R2/experiment.json",
         source_id="R2",start_auc=0.7257211273845012,
         start_uauc=0.6829168466943410),
    dict(id="N2",title="Resume R7 QKVO alpha64 to total 2 epochs",
         projections=ALL,layers="0-31",cf_input="concat",
         alpha=64,p_lr=1.0,seed=42,mode="resume",
         checkpoint=str(R7_CKPT),
         source_meta="logs/NO31_CORA_NIGHT8_20261024_012519_960783/R7/experiment.json",
         source_id="R7",start_auc=0.7288642146203805,
         start_uauc=0.6855345718193175),
    dict(id="N3",title="Resume F4 QKVO alpha48 to total 2 epochs",
         projections=ALL,layers="0-31",cf_input="concat",
         alpha=48,p_lr=1.0,seed=42,mode="resume",
         checkpoint=str(F4_CKPT),
         source_meta="logs/NO31_CORA_FOLLOWUP8_20261024_130326_269104/F4/experiment.json",
         source_id="F4",start_auc=0.7301218951846701,
         start_uauc=0.6880410112996300),
    dict(id="N4",title="Fresh V/O alpha48 for 2 epochs",
         projections="v_proj,o_proj",layers="0-31",cf_input="concat",
         alpha=48,p_lr=1.0,seed=42,mode="fresh",
         checkpoint=None,source_meta=None,source_id=None,
         start_auc=None,start_uauc=None),
]
BY_ID={r["id"]:r for r in EXPERIMENTS}
AUC_PATTERN=re.compile(r"\*\*\*auc:\s*([0-9.]+).*?\*\*\*uauc:\s*\(\s*([0-9.]+)")

def prepared_config():
    cfg=core.prepared_config()
    if cfg["model"].get("fix_autocast_cache") is not True:
        raise RuntimeError("fix_autocast_cache must remain enabled")
    if int(cfg["run"].get("max_epoch",-1))!=3:
        raise RuntimeError("Keep max_epoch=3 as the original LR schedule horizon")
    if int(cfg["run"].get("seed",-1))!=42:
        raise RuntimeError("Do not change the requested random seed")
    return cfg

def validate_metadata(row):
    if row["mode"]=="fresh":
        return
    meta_path=ROOT/row["source_meta"]
    if not meta_path.is_file():
        raise FileNotFoundError(f"Original experiment metadata missing: {meta_path}")
    meta=json.loads(meta_path.read_text())
    for key in ("projections","layers","cf_input","alpha","p_lr"):
        if meta.get(key)!=row[key]:
            raise RuntimeError(
                f"{row['id']} source metadata mismatch for {key}: "
                f"expected {row[key]!r}, got {meta.get(key)!r}"
            )
    if meta.get("id")!=row["source_id"]:
        raise RuntimeError(f"Wrong source experiment metadata: {meta_path}")

def validate_checkpoint(row,verbose=True):
    if row["mode"]!="resume":
        return
    path=Path(row["checkpoint"])
    if not path.is_file():
        raise FileNotFoundError(f"{row['id']} source checkpoint missing: {path}")
    validate_metadata(row)
    ck=torch.load(str(path),map_location="cpu",weights_only=False)
    if int(ck.get("epoch",-1))!=0:
        raise RuntimeError(f"{row['id']} must resume from epoch 0, got {ck.get('epoch')}")
    if not isinstance(ck.get("model"),dict):
        raise RuntimeError(f"{row['id']} checkpoint has no model state")
    if not isinstance(ck.get("optimizer"),dict) or not ck["optimizer"].get("state"):
        raise RuntimeError(f"{row['id']} checkpoint has no AdamW state; refusing weights-only resume")
    if ck.get("scaler") is None:
        raise RuntimeError(f"{row['id']} checkpoint has no AMP scaler state")
    saved_cfg=ck.get("config",{})
    mc=saved_cfg.get("model",{})
    rc=saved_cfg.get("run",{})
    expected_cfg=[
        (float(mc.get("gnn_scale",-1))==0.0,"gnn_scale=0"),
        (float(mc.get("gnn_conf_loss_weight",-1))==0.05,"CTR=.05"),
        (float(mc.get("s2_small_kd",-1))==0.05,"KD=.05"),
        (mc.get("gnn_semantic_align") is True,"semantic alignment ON"),
        (float(mc.get("gnn_semantic_weight",-1))==1e-4,"semantic weight=1e-4"),
        (float(mc.get("proj_reg_weight",-1))==5e-4,"projector reg=5e-4"),
        (mc.get("rec_model")=="MF","MF CF model"),
        (int(rc.get("seed",-1))==42,"seed42"),
        (int(rc.get("max_epoch",-1))==3,"3-epoch LR horizon"),
    ]
    for ok,label in expected_cfg:
        if not ok:
            raise RuntimeError(f"{row['id']} checkpoint config mismatch: {label}")
    sd=ck["model"]
    lo,hi=map(int,row["layers"].split("-"))
    expected_keys=[]
    for layer in range(lo,hi+1):
        for proj in row["projections"].split(","):
            suffix=f"layers.{layer}.self_attn.{proj}.cora_lora_P.weight"
            matches=[k for k in sd if k.endswith(suffix)]
            if len(matches)!=1:
                raise RuntimeError(f"{row['id']} expected one P tensor for {suffix}; got {matches}")
            key=matches[0]
            if tuple(sd[key].shape)!=(4096,16):
                raise RuntimeError(f"{row['id']} bad P shape: {key} {tuple(sd[key].shape)}")
            if float(sd[key].float().norm())<=0:
                raise RuntimeError(f"{row['id']} untrained/zero P in source checkpoint: {key}")
            expected_keys.append(key)
    actual=[k for k in sd if k.endswith(".cora_lora_P.weight")]
    if len(actual)!=len(expected_keys):
        raise RuntimeError(
            f"{row['id']} active CoRA target count mismatch: "
            f"expected {len(expected_keys)}, got {len(actual)}"
        )
    if verbose:
        print(f"[CONTINUE4-CKPT-OK] {row['id']} epoch=0 "
              f"optimizer_state={len(ck['optimizer']['state'])} "
              f"AMP_scaler=present P_tensors={len(expected_keys)} "
              f"alpha={row['alpha']} projections={row['projections']}",flush=True)
    del ck

def plan(rows):
    prepared_config()
    print("[CONTINUE4-PLAN] two total epochs per arm; seed=42 for all; "
          "GPU0 serial foreground; Valid only; no Test",flush=True)
    print("[CONTINUE4-LR] max_epoch=3 retained for the original LR schedule; "
          "S2_NIGHT_EPOCHS=2 controls the number of completed epochs",flush=True)
    for row in rows:
        if row["mode"]=="resume":
            print(f"[CONTINUE4-PLAN] {row['id']} RESUME source_epoch0={row['checkpoint']} "
                  f"-> train epoch1; alpha={row['alpha']} "
                  f"targets={row['projections']}; start_auc={row['start_auc']:.9f} "
                  f"start_uauc={row['start_uauc']:.9f}",flush=True)
        else:
            print(f"[CONTINUE4-PLAN] {row['id']} FRESH epochs0+1 "
                  f"alpha={row['alpha']} targets={row['projections']}",flush=True)

def run_one(row,common_cfg,root):
    ident=row["id"]
    exp_dir=root/ident
    exp_dir.mkdir(parents=False,exist_ok=False)
    cfg=yaml.safe_load(yaml.safe_dump(common_cfg))
    cfg["run"]["seed"]=42
    cfg["run"]["max_epoch"]=3
    cfg["run"]["lr_scale_cora_p"]=float(row["p_lr"])
    cfg["run"]["resume_ckpt_path"]=row["checkpoint"] if row["mode"]=="resume" else None
    cfg["run"]["output_dir"]=str(ROOT/"output"/root.name/ident)
    cfg["run"]["valid_splits"]=["valid"]
    cfg["run"]["test_splits"]=[]
    cfg["run"]["evaluate"]=False
    cfg["run"]["save_checkpoint_every_n_epochs"]=1
    cfg_path=exp_dir/"train.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    (exp_dir/"experiment.json").write_text(json.dumps(row,indent=2))
    command=[sys.executable,"-u","experiments/no31_cora_original/entry.py",
             "--cfg-path",str(cfg_path)]
    env=core.build_environment(row)
    env["S2_NIGHT_EPOCHS"]="2"
    env["PYTHONUNBUFFERED"]="1"
    log_file=exp_dir/"train.log"
    print(f"[CONTINUE4-START] {ident} mode={row['mode']} alpha={row['alpha']} "
          f"targets={row['projections']} seed=42 log={log_file}",flush=True)
    with log_file.open("x",encoding="utf-8") as saved:
        child=subprocess.Popen(command,cwd=str(ROOT),env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            encoding="utf-8",errors="replace",bufsize=1)
        try:
            for line in child.stdout:
                saved.write(line);saved.flush()
                if any(word in line for word in (
                    "[CORA-ORIGINAL","[NIGHT8-","[CONTINUE4","[FX]",
                    "[lr-group]","Averaged stats:","OutOfMemory","Traceback",
                    "RuntimeError","Resume checkpoint")):
                    print(f"[{ident}] {line.rstrip()[:700]}",flush=True)
            code=child.wait()
        except BaseException:
            child.terminate();child.wait();raise
    if code:
        raise RuntimeError(f"{ident} training failed exit={code}; log preserved: {log_file}")
    text=log_file.read_text(errors="replace")
    required=[
        "[CORA-ORIGINAL-PROBE-OK]",
        "[CORA-ORIGINAL-COUNTERS]",
        "[NIGHT8-STOP] trained_epochs=2 max_epoch=3",
        "[FX] autocast cache fix ON",
        f"static_P={core.selected_targets(row)} x [4096,16]",
        f"alpha={row['alpha']}",
        f"projections={row['projections']}",
        f"layers={row['layers']}",
        f"input={row['cf_input']}",
    ]
    if row["mode"]=="resume":
        required.append(f"Resume checkpoint from {row['checkpoint']}")
    missing=[marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(f"{ident} missing verification markers {missing}; log={log_file}")
    lr_lines=[line for line in text.splitlines() if "[lr-group]" in line and "cora_p_wd" in line]
    if not lr_lines or not any("scale=1" in line or "scale=1.0" in line for line in lr_lines):
        raise RuntimeError(f"{ident} CoRA-P LR scale was not verified: {lr_lines}")
    metrics=[]
    for line in text.splitlines():
        if "Averaged stats:" in line:
            match=AUC_PATTERN.search(line)
            if match:metrics.append((float(match.group(1)),float(match.group(2))))
    expected_metrics=1 if row["mode"]=="resume" else 2
    if len(metrics)!=expected_metrics:
        raise RuntimeError(f"{ident} expected {expected_metrics} new Valid metrics, got {metrics}")
    if row["mode"]=="resume":
        epoch0_auc,epoch0_uauc=row["start_auc"],row["start_uauc"]
        epoch1_auc,epoch1_uauc=metrics[-1]
    else:
        (epoch0_auc,epoch0_uauc),(epoch1_auc,epoch1_uauc)=metrics
    outputs=Path(cfg["run"]["output_dir"])
    saved_ckpts=list(outputs.rglob("checkpoint_1.pth"))
    if len(saved_ckpts)!=1:
        raise RuntimeError(f"{ident} expected one epoch1 checkpoint, got {saved_ckpts}")
    ck=torch.load(str(saved_ckpts[0]),map_location="cpu",weights_only=False)
    if int(ck.get("epoch",-1))!=1 or not ck.get("optimizer") or ck.get("scaler") is None:
        raise RuntimeError(f"{ident} final checkpoint does not preserve epoch1 optimizer/scaler")
    del ck
    result=dict(
        id=ident,mode=row["mode"],alpha=row["alpha"],projections=row["projections"],
        seed=42,epoch0_auc=epoch0_auc,epoch0_uauc=epoch0_uauc,
        epoch1_auc=epoch1_auc,epoch1_uauc=epoch1_uauc,
        delta_auc=epoch1_auc-epoch0_auc,delta_uauc=epoch1_uauc-epoch0_uauc,
        checkpoint=str(saved_ckpts[0]),log=str(log_file),
        source_checkpoint=row["checkpoint"],
    )
    (exp_dir/"result.json").write_text(json.dumps(result,indent=2))
    print(f"[CONTINUE4-RESULT] {ident} epoch0={epoch0_auc:.9f}/{epoch0_uauc:.9f} "
          f"epoch1={epoch1_auc:.9f}/{epoch1_uauc:.9f} "
          f"delta={result['delta_auc']:+.9f}/{result['delta_uauc']:+.9f}",flush=True)
    return result

def run(rows):
    common_cfg=prepared_config()
    subprocess.run([sys.executable,"experiments/no31_cora_original/test_night8_cpu.py"],
                   cwd=str(ROOT),check=True)
    subprocess.run([sys.executable,"experiments/no31_cora_original/test_continue4_cpu.py"],
                   cwd=str(ROOT),check=True)
    for row in rows:
        validate_checkpoint(row)
    if core.gpu_is_in_use():
        raise RuntimeError("GPU0 already has a compute process; refusing overlapping training")
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root=ROOT/"logs"/f"NO31_CORA_CONTINUE4_TWOEPOCHS_{stamp}"
    root.mkdir(parents=True,exist_ok=False)
    (root/"plan.json").write_text(json.dumps({
        "epochs_total":2,"seed":42,"max_epoch_schedule":3,
        "GPU":"0","valid_only":True,"test_splits":[],
        "experiments":rows,
    },indent=2))
    print(f"[CONTINUE4-ROOT] {root}",flush=True)
    records=[]
    for row in rows:
        try:
            result=run_one(row,common_cfg,root)
        except BaseException as exc:
            (root/"failed.json").write_text(json.dumps(
                {"id":row["id"],"reason":str(exc)},indent=2))
            print(f"[CONTINUE4-ABORT] stopped at {row['id']}; previous results preserved in {root}",flush=True)
            raise
        records.append(result)
        with (root/"results.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(result)+"\n")
        with (root/"summary.csv").open("w",encoding="utf-8",newline="") as f:
            fields=("id","mode","alpha","projections","seed","epoch0_auc","epoch0_uauc",
                    "epoch1_auc","epoch1_uauc","delta_auc","delta_uauc","checkpoint","log")
            writer=csv.DictWriter(f,fieldnames=fields)
            writer.writeheader()
            for record in records:writer.writerow({k:record[k] for k in fields})
    print(f"[CONTINUE4-COMPLETE] 4 experiments complete; summary={root/'summary.csv'}",flush=True)

def main():
    p=argparse.ArgumentParser(description="Two-total-epoch continuation for R2/R7/F4 plus fresh V/O alpha48")
    action=p.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan",action="store_true")
    action.add_argument("--check-checkpoints",action="store_true")
    action.add_argument("--run",action="store_true")
    p.add_argument("--only",choices=list(BY_ID))
    p.add_argument("--start",choices=list(BY_ID))
    args=p.parse_args()
    if args.only and args.start:
        p.error("Use --only or --start, not both")
    rows=EXPERIMENTS
    if args.only:rows=[BY_ID[args.only]]
    if args.start:rows=EXPERIMENTS[list(BY_ID).index(args.start):]
    plan(rows)
    if args.check_checkpoints:
        for row in rows:validate_checkpoint(row)
        print("[CONTINUE4-CHECKPOINTS-PASS]",flush=True)
        return
    if args.run:run(rows)

if __name__=="__main__":
    main()
