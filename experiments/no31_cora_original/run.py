#!/usr/bin/env python3
"""Prepare and run *one* exact CoRA-mechanism experiment on original NO31.

Does not modify historical NO31-CTR/KD model/loss code. Foreground GPU0 only.
New output dirs, Valid AUC/UAUC and checkpoints; never writes Test results.
"""
import argparse
import copy
import hashlib
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"tran_configs/S2_groups_acc3_recovered.yaml"
STAGING=ROOT/"logs/NO31_CORA_ORIGINAL_READY"
RUNNAME="NO31_CORA_ORIGINAL"
AUC_RE=re.compile(r"\*\*\*auc:\s*([0-9.]+).*?\*\*\*uauc:\s*\(\s*([0-9.]+)")


def prepare():
    if STAGING.exists():
        raise FileExistsError(f"Existing configs preserved: {STAGING}; call run instead")
    cfg=yaml.safe_load(SOURCE.read_text())
    if (cfg["model"]["lora_gnn"]!="ngcf"
            or cfg["model"]["rec_model"]!="MF"
            or cfg["model"]["gnn_semantic_align"] is not True
            or cfg["model"]["gnn_semantic_weight"]!=1e-4
            or cfg["model"]["proj_reg_weight"]!=5e-4
            or cfg["model"]["rec_config"]["embedding_size"]!=256
            or cfg["run"]["seed"]!=42
            or cfg["run"]["max_epoch"]!=3):
        raise RuntimeError("Original NO31 baseline model and recipe changed")
    if cfg["model"]["ckpt"] or cfg["run"].get("resume_ckpt_path"):
        raise RuntimeError("Do not initialize CoRA from a trained GraphLoRA checkpoint")
    if not Path(cfg["model"]["rec_config"]["pretrained_path"]).is_file():
        raise FileNotFoundError("Pretrained NO31 MF checkpoint missing")
    if not Path(cfg["model"]["llama_model"]).is_dir():
        raise FileNotFoundError("Vicuna-7B directory missing")
    # The original GraphLoRA NO31 training code uses graph NPZ files
    # addressed relative to the worktree, not the external dataset YAML path.
    # Link and SHA256-verify the historical files, without copying them.
    src_graph=Path("/home/bbx/user_lx/GraphLoRA/dataset/ml-1m")
    dst_graph=ROOT/"dataset/ml-1m"
    if not src_graph.is_dir():
        raise FileNotFoundError(src_graph)
    dst_graph.mkdir(parents=True,exist_ok=True)
    def checksum(path):
        h=hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda:stream.read(1048576),b""):
                h.update(chunk)
        return h.hexdigest()
    for split in ("train","valid","test"):
        original=src_graph/f"movie_graph_{split}.npz"
        target=dst_graph/f"movie_graph_{split}.npz"
        if not original.is_file():
            raise FileNotFoundError(original)
        if not target.exists() and not target.is_symlink():
            target.symlink_to(original)
        if not target.is_file() or checksum(original)!=checksum(target):
            raise RuntimeError(f"Historical graph NPZ mismatch: {target}")
    # Preserve NO31 semantic alignment, projector regularization,
    # MF/NGCF finetuning, LR schedule, data splits, batching and seed.
    cfg["model"]["gnn_scale"]=0.0
    cfg["model"]["gnn_conf_loss_weight"]=0.05
    cfg["model"]["ckpt"]=None
    cfg["run"]["resume_ckpt_path"]=None
    cfg["run"]["valid_splits"]=["valid"]
    cfg["run"]["test_splits"]=[]
    cfg["run"]["evaluate"]=False
    cfg["run"]["save_checkpoint_every_n_epochs"]=1
    cfg["run"]["output_dir"]=str(ROOT/"output"/RUNNAME)
    STAGING.mkdir(parents=True,exist_ok=False)
    (STAGING/"train.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False))
    (STAGING/"manifest.json").write_text(json.dumps({
        "source":"NO31+CTR/KD restored from original historical Night8 patch",
        "source_config":str(SOURCE),
        "original_CoRA":"VanillaCreamer/CoRA main",
        "original_modules":"QFormer8x8,4 queries; full dynamic rank16 A; static zero-P; 32xQKVO",
        "batch_query_expansion_fix":True,
        "cf_condition":"frozen original pretrained MF u and i, 256+256",
        "gnn_scale":0.0,"CTR":0.05,"KD":0.05,"sem":1e-4,"proj_reg":5e-4,
        "init_lr":cfg["run"]["init_lr"],"max_epoch":3,"train_epochs":1,"seed":42,
        "dataset":"ML1M OOD2","evaluation":"Valid only"
    },indent=2))
    print("[CORA-ORIGINAL-PREPARED] original full-matrix A+P; NO31 CTR/KD unchanged; 1 epoch",flush=True)


def run(probe_only=False):
    if not (STAGING/"train.yaml").is_file():
        raise FileNotFoundError("Run prepare first; original NO31 model config absent")
    cfg=yaml.safe_load((STAGING/"train.yaml").read_text())
    if (float(cfg["model"]["gnn_scale"])!=0
        or float(cfg["model"]["gnn_conf_loss_weight"])!=0.05
        or cfg["run"]["test_splits"]):
        raise RuntimeError("NO31 constraints changed")
    if probe_only:
        cmd=[sys.executable,"-u","experiments/no31_cora_original/entry.py",
             "--cfg-path",str(STAGING/"train.yaml"),"--probe-only"]
        env={**os.environ,"CUDA_VISIBLE_DEVICES":"0","S2_INPUT_ALPHA":"0",
             "S2_SMALL_KD":"0.05","S2_NIGHT_EPOCHS":"1",
             "PYTHONUNBUFFERED":"1"}
        code=subprocess.call(cmd,cwd=ROOT,env=env)
        if code:raise RuntimeError(f"CoRA model init probe failed exit={code}")
        return
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_log=ROOT/"logs"/f"{RUNNAME}_{stamp}"
    if run_log.exists():
        raise FileExistsError(run_log)
    run_log.mkdir(parents=True,exist_ok=False)
    config=copy.deepcopy(cfg)
    config["run"]["output_dir"]=str(ROOT/"output"/f"{RUNNAME}_{stamp}")
    recipe=run_log/"train.yaml"
    recipe.write_text(yaml.safe_dump(config,sort_keys=False))
    logfile=run_log/"train.log"
    cmd=[sys.executable,"-u","experiments/no31_cora_original/entry.py",
         "--cfg-path",str(recipe)]
    env={**os.environ,"CUDA_VISIBLE_DEVICES":"0","S2_INPUT_ALPHA":"0",
         "S2_SMALL_KD":"0.05","S2_NIGHT_EPOCHS":"1","PYTHONUNBUFFERED":"1"}
    print("[CORA-ORIGINAL-TRAIN] full Q/K/V/O 32 layers, 1 epoch, seed42, output="+str(run_log),flush=True)
    with logfile.open("x") as out:
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,
             stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
             text=True,encoding="utf8",errors="replace",bufsize=1)
        try:
            for line in p.stdout:
                out.write(line)
                out.flush()
                if any(w in line for w in ("[CORA-ORIGINAL","[NIGHT8-",
                        "Averaged stats:","Traceback","Error:","OutOfMemory")):
                    print(line.rstrip()[:650],flush=True)
            exitcode=p.wait()
        except BaseException:
            p.terminate();p.wait();raise
    if exitcode:
        raise RuntimeError(f"Training failed exit={exitcode}; log preserved: {logfile}")
    text=logfile.read_text(errors="replace")
    if ("[CORA-ORIGINAL-PROBE-OK]" not in text or
        "[NIGHT8-STOP] trained_epochs=1" not in text or
        "[NIGHT8-LOSS]" not in text or
        "[CORA-ORIGINAL-COUNTERS]" not in text):
        raise RuntimeError("Training log missing CoRA/NO31/epoch checks")
    record=re.findall(r"\[CORA-ORIGINAL-COUNTERS\]\s*(\{[^\n]+\})",text)
    counts=json.loads(record[-1])
    if counts.get("conditioned_batches",0)<1 or counts.get("projection_forwards",0)<128:
        raise RuntimeError(f"Original CoRA weights were not used correctly: {counts}")
    metrics=[]
    for line in text.splitlines():
        if "Averaged stats:" in line:
            m=AUC_RE.search(line)
            if m:metrics.append((float(m.group(1)),float(m.group(2))))
    if len(metrics)!=1:
        raise RuntimeError(f"Expected a single epoch0 Valid AUC/UAUC; observed {metrics}")
    checkpoints=list(Path(config["run"]["output_dir"]).rglob("checkpoint_0.pth"))
    if len(checkpoints)!=1:
        raise RuntimeError(f"Expected one final checkpoint_0.pth: {checkpoints}")
    auc,uauc=metrics[0]
    result={"valid_auc":auc,"valid_uauc":uauc,"epoch":0,
            "no31_baseline_auc":0.719760366,
            "delta_auc_vs_no31":auc-0.719760366,
            "checkpoint":str(checkpoints[0]),"log":str(logfile),"counters":counts}
    (run_log/"result.json").write_text(json.dumps(result,indent=2))
    print("[CORA-ORIGINAL-COMPLETE] ValidAUC=%.9f ValidUAUC=%.9f deltaNO31=%+.9f" %
          (auc,uauc,auc-0.719760366),flush=True)
    print("[CORA-ORIGINAL-CKPT] "+str(checkpoints[0]),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("prepare","probe","run"))
    args=p.parse_args()
    os.chdir(ROOT)
    if args.action=="prepare":
        prepare()
    elif args.action=="probe":
        run(probe_only=True)
    else:
        if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
            raise RuntimeError("GPU0 required, CUDA_VISIBLE_DEVICES=0")
        run(probe_only=False)


if __name__=="__main__":
    main()
