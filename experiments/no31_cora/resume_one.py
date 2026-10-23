#!/usr/bin/env python3
"""Resume ONLY archived NO31+CTR/KD+CoRA from epoch0 for epoch1 (1 epoch).

This is a genuine optimizer/scaler/model resume, not a new two-epoch training.
It never overwrites previous checkpoints, log files or YAMLs. Both original
three-epoch schedule and model/CTR/KD/CoRA knobs stay unchanged.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
ORIG_LOG = ROOT / "logs" / "NO31_CTRKD_CORA_20261023"
ORIG_OUTPUT = ROOT / "output" / "NO31_CTRKD_CORA_20261023" / "cora"
AUC_RE = re.compile(r"\*\*\*auc:\s*([0-9.]+).*?\*\*\*uauc:\s*\(\s*([0-9.]+)")
BASELINE_EPOCH0_AUC = 0.635651530081458
BASELINE_EPOCH0_UAUC = 0.5958768400456956


def read_ckpt(path):
    # Only read checkpoints produced by your own training process.
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def check_checkpoint(checkpoint_path):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    ckpt = read_ckpt(checkpoint_path)
    if not isinstance(ckpt, dict):
        raise RuntimeError("Invalid checkpoint format")
    needed = {"model", "optimizer", "scaler", "epoch", "config"}
    if needed - set(ckpt):
        raise RuntimeError(f"Checkpoint lacks full resume state: {needed - set(ckpt)}")
    if int(ckpt["epoch"]) != 0:
        raise RuntimeError(f"Need CoRA epoch0, found epoch={ckpt['epoch']}")
    cfg = ckpt["config"]
    if int(cfg["run"]["seed"]) != 42 or int(cfg["run"]["max_epoch"]) != 3:
        raise RuntimeError("Historical seed or LR scheduler differs")
    if float(cfg["model"]["gnn_scale"]) != 0:
        raise RuntimeError("NO31 feature injection is enabled in checkpoint")
    if float(cfg["model"]["gnn_conf_loss_weight"]) != 0.05:
        raise RuntimeError("CTR weight differs")
    if cfg["model"]["gnn_semantic_align"] is not True:
        raise RuntimeError("GNN semantic supervision disabled")
    if cfg["model"].get("ckpt") or cfg["run"].get("resume_ckpt_path"):
        raise RuntimeError("Original source training unexpectedly used prior resume")
    names = ckpt["model"]
    for key in ("r3_adapter.q_up.weight", "r3_adapter.v_up.weight"):
        if key not in names:
            raise RuntimeError(f"CoRA weight missing: {key}")
        tensor = names[key]
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"Non-finite CoRA weight: {key}")
    if not ckpt["optimizer"].get("state"):
        raise RuntimeError("Missing AdamW optimizer moments")
    groups = ckpt["optimizer"].get("param_groups", [])
    cora_groups = [g for g in groups if g.get("group_name", "").startswith("cora_")]
    if not cora_groups:
        raise RuntimeError("Optimizer does not contain trained CoRA param group")
    if any(float(g.get("lr_scale", -1)) != 0.25 for g in cora_groups):
        raise RuntimeError("Original CoRA LR multiplier differs")
    if ckpt["scaler"] is None:
        raise RuntimeError("Original AMP GradScaler state missing")
    print("[NO31-RESUME-CKPT] epoch0 model, AdamW, AMP scaler, CTR/KD, CoRA verified",flush=True)
    print("[NO31-RESUME-CORA-NORM] q=%.8g v=%.8g" %
          (names["r3_adapter.q_up.weight"].float().norm().item(),
           names["r3_adapter.v_up.weight"].float().norm().item()),flush=True)
    del ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,default=None,
                   help="Optional exact CoRA checkpoint_0.pth path")
    p.add_argument("--dry-run",action="store_true",
                   help="Validate source checkpoint and print plan, without starting training")
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Use CUDA_VISIBLE_DEVICES=0 for GPU0 only")
    os.chdir(ROOT)
    source_yaml = ORIG_LOG / "cora.yaml"
    if not source_yaml.is_file():
        raise FileNotFoundError(f"Original CoRA training recipe not found: {source_yaml}")
    cfg = yaml.safe_load(source_yaml.read_text())
    if float(cfg["model"]["gnn_scale"]) != 0 or float(cfg["model"]["gnn_conf_loss_weight"]) != 0.05:
        raise RuntimeError("Not the original NO31/CTR+KD CoRA recipe")
    if int(cfg["run"]["seed"]) != 42 or int(cfg["run"]["max_epoch"]) != 3:
        raise RuntimeError("Training seed/scheduler configuration changed")
    if cfg["run"].get("resume_ckpt_path") or cfg["model"].get("ckpt"):
        raise RuntimeError("Original CoRA recipe must start fresh")

    if args.checkpoint is None:
        paths = sorted(ORIG_OUTPUT.rglob("checkpoint_0.pth"))
        if len(paths) != 1:
            raise RuntimeError(f"Expected ONE original CoRA checkpoint_0.pth, found {paths}")
        ckpt = paths[0].resolve()
    else:
        ckpt = args.checkpoint.expanduser().resolve()
    if ckpt.name != "checkpoint_0.pth":
        raise RuntimeError("Only epoch0 checkpoint supported")
    check_checkpoint(ckpt)

    print("[NO31-RESUME-PLAN] original=%s -> only epoch1 (S2_NIGHT_EPOCHS=2)" % ckpt,flush=True)
    print("[NO31-RESUME-PLAN] CoRA=Q+V ID only, gain=.1, detach_gnn=True; "
          "CTR=.05 KD=.05 seed42 max_epoch=3",flush=True)
    if args.dry_run:
        print("[NO31-RESUME-DRY-RUN-OK] No files or GPU training created",flush=True)
        return

    # New distinct output tree, never touch the completed original run.
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root_log=ROOT / "logs" / ("NO31_CTRKD_CORA_RESUME_E1_" + stamp)
    root_out=ROOT / "output" / ("NO31_CTRKD_CORA_RESUME_E1_" + stamp)
    root_log.mkdir(parents=True,exist_ok=False)
    config=copy.deepcopy(cfg)
    config["run"]["resume_ckpt_path"]=str(ckpt)
    config["run"]["output_dir"]=str(root_out)
    config["run"]["max_epoch"]=3
    config["run"]["save_checkpoint_every_n_epochs"]=1
    config["run"]["evaluate"]=False
    config["run"]["test_splits"]=[]
    config["run"]["valid_splits"]=["valid"]
    config["model"]["ckpt"]=None

    recipe=root_log/"resume_epoch1.yaml"
    recipe.write_text(yaml.safe_dump(config,sort_keys=False),encoding="utf-8")
    init_hash=root_log/"resume_fresh_init.json"
    log=root_log/"resume_epoch1.log"
    cmd=[
        sys.executable,"-u","experiments/no31_cora/entry.py",
        "--cfg-path",str(recipe),"--mode","cora",
        "--init-hash",str(init_hash),
        "--cora-active","both","--cora-gain","0.1",
        "--cora-detach-gnn","true",
    ]
    env={**os.environ,"CUDA_VISIBLE_DEVICES":"0","S2_NIGHT_EPOCHS":"2",
         "S2_INPUT_ALPHA":"0.00","S2_SMALL_KD":"0.05","PYTHONUNBUFFERED":"1"}
    print("[NO31-RESUME-START] epoch=1, new_log=%s" % log,flush=True)
    with log.open("x",encoding="utf-8") as dest:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,
                   stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                   text=True,encoding="utf-8",errors="replace",bufsize=1)
        try:
            for line in proc.stdout:
                dest.write(line)
                dest.flush()
                if ("[NO31" in line or "[NIGHT8-" in line or
                    "Resume checkpoint from" in line or "Averaged stats:" in line or
                    "Traceback" in line or "Error:" in line):
                    print(line.strip()[:550],flush=True)
            code=proc.wait()
        except BaseException:
            proc.terminate();proc.wait();raise
    if code:
        raise RuntimeError(f"Resume failed with exit={code}. Inspect {log}")
    text=log.read_text(errors="replace")
    if "Resume checkpoint from" not in text:
        raise RuntimeError("Runner did not confirm restoration of optimizer/scaler/model")
    if "[NIGHT8-STOP] trained_epochs=2" not in text:
        raise RuntimeError("Epoch1 was not completed")
    if "Train: data epoch: [1]" not in text or "Train: data epoch: [0]" in text:
        raise RuntimeError("Unexpected training epoch indices")
    if "[NIGHT8-LOSS]" not in text:
        raise RuntimeError("CTR/KD losses did not execute")
    if "[NO31-CORA-COUNTERS]" not in text:
        raise RuntimeError("CoRA forward counters missing")
    matches=[]
    for line in text.splitlines():
        if "Averaged stats:" in line:
            m=AUC_RE.search(line)
            if m:matches.append((float(m.group(1)),float(m.group(2))))
    if len(matches)!=1:
        raise RuntimeError(f"Expected one Valid AUC/UAUC for resumed epoch1: {matches}")
    scores=(auc,uauc)=matches[0]
    checkpoints=list(root_out.rglob("checkpoint_1.pth"))
    if len(checkpoints)!=1:
        raise RuntimeError(f"Expected one checkpoint_1.pth under {root_out}: {checkpoints}")
    if list(root_out.rglob("checkpoint_0.pth")):
        raise RuntimeError("Resume unexpectedly restarted epoch0")
    summary={
        "source_checkpoint":str(ckpt),
        "resumed_epoch":1,
        "valid_auc":auc,
        "valid_uauc":uauc,
        "valid_auc_delta_vs_original_epoch0":auc-BASELINE_EPOCH0_AUC,
        "valid_uauc_delta_vs_original_epoch0":uauc-BASELINE_EPOCH0_UAUC,
        "new_checkpoint":str(checkpoints[0]),
        "training_log":str(log),
    }
    (root_log/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("[NO31-RESUME-COMPLETE] epoch1 ValidAUC=%.9f ValidUAUC=%.9f "
          "delta_vs_epoch0_auc=%+.9f delta_vs_epoch0_uauc=%+.9f" %
          (auc,uauc,auc-BASELINE_EPOCH0_AUC,uauc-BASELINE_EPOCH0_UAUC),flush=True)
    print("[NO31-RESUME-CHECKPOINT] "+str(checkpoints[0]),flush=True)
    print("[NO31-RESUME-LOG] "+str(log),flush=True)


if __name__=="__main__":
    main()
