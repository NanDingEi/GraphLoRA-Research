#!/usr/bin/env python3
"""Nine controlled S2 vs CoRA experiments, all from the original S2 recipe.

Each experiment starts a FRESH model: pretrained historical MF and Vicuna,
fresh GNN/Projector/LoRA, optional fresh CoRA. No GraphLoRA/S2 checkpoints.
Two complete train+Valid epochs per experiment, retaining 3-epoch cosine horizon.
Runs strictly serially on CUDA_VISIBLE_DEVICES=0, foreground only.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import yaml

ROOT=Path(__file__).resolve().parents[2]
EXPERIMENTS=ROOT/"experiments/s2_cora"
OUT=ROOT/"output/S2_CORA_NINE_20261022"
LOG=ROOT/"logs/S2_CORA_NINE_20261022"
SOURCE=ROOT/"tran_configs/S2_groups_acc3_recovered.yaml"
ORIGINAL_GRAPH=Path("/home/bbx/user_lx/GraphLoRA/dataset/ml-1m")
EXPERIMENTS_9=[
    {"id":"01_S2_S42","mode":"baseline","seed":42,"branch":"both","gain":1.0,"feature":"both","attention":"full"},
    {"id":"02_CORA_QV","mode":"cora","seed":42,"branch":"both","gain":1.0,"feature":"both","attention":"full"},
    {"id":"03_CORA_Q","mode":"cora","seed":42,"branch":"q","gain":1.0,"feature":"both","attention":"full"},
    {"id":"04_CORA_V","mode":"cora","seed":42,"branch":"v","gain":1.0,"feature":"both","attention":"full"},
    {"id":"05_CORA_HALF","mode":"cora","seed":42,"branch":"both","gain":0.5,"feature":"both","attention":"full"},
    {"id":"06_CORA_USER","mode":"cora","seed":42,"branch":"both","gain":1.0,"feature":"user_only","attention":"full"},
    {"id":"07_CORA_ITEM","mode":"cora","seed":42,"branch":"both","gain":1.0,"feature":"item_only","attention":"full"},
    {"id":"08_CORA_INTERACT","mode":"cora","seed":42,"branch":"both","gain":1.0,"feature":"interaction","attention":"full"},
    {"id":"09_CORA_NO_CROSS","mode":"cora","seed":42,"branch":"both","gain":1.0,"feature":"both","attention":"no_cross"},
]
HASH_GROUPS=("MF","GNN","Projector","LoRA")
METRICS=re.compile(r"\*\*\*auc:\s*([0-9.]+).*?\*\*\*uauc:\s*\(\s*([0-9.]+)")
NEEDED={
    ("model","gnn_semantic_align"):True,
    ("model","gnn_semantic_weight"):1e-4,
    ("model","proj_reg_weight"):5e-4,
    ("model","proj_reg_items"):64,
    ("model","train_gnn_with_rec_loss"):True,
    ("model","gcn_layer"):31,
    ("model","gnn_scale"):0.1,
    ("run","accum_grad_iters"):3,
    ("run","batch_size_train"):3,
    ("run","batch_size_eval"):4,
    ("run","init_lr"):1e-4,
    ("run","min_lr"):5e-5,
    ("run","lr_scale_rec"):0.5,
    ("run","lr_scale_gnn"):0.25,
    ("run","lr_scale_proj"):0.5,
    ("run","lr_scale_lora"):1.0,
    ("run","max_epoch"):3,
}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def historical_reference():
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Missing historical S2 YAML: {SOURCE}")
    cfg=yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    for (section,key),expected in NEEDED.items():
        value=cfg[section].get(key)
        equal=(value==expected) if not isinstance(expected,float) else (
            value is not None and math.isclose(float(value),expected,abs_tol=1e-12,rel_tol=0))
        if not equal:
            raise RuntimeError(f"Historical S2 recipe changed {section}.{key}: {value} != {expected}")
    m,r=cfg["model"],cfg["run"]
    if m.get("freeze_rec") is not False or m.get("freeze_lora") is not False or m.get("freeze_proj") is not False:
        raise RuntimeError("S2 must train MF, Projector and LoRA")
    if m.get("lora_gnn")!="ngcf" or m.get("rec_model")!="MF" or m.get("fix_autocast_cache",False):
        raise RuntimeError("S2 NGCF/MF model and original FX setting not preserved")
    if m.get("ckpt") not in (None,"") or r.get("resume_ckpt_path") not in (None,""):
        raise RuntimeError("Jointly trained GraphLoRA/S2 weights are forbidden")
    mf=Path(m["rec_config"]["pretrained_path"])
    if not mf.is_file():
        raise FileNotFoundError(f"Historical pretrained MF missing: {mf}")
    if not Path(m["llama_model"]).is_dir():
        raise FileNotFoundError(f"Vicuna pretrained model missing: {m['llama_model']}")
    for split in ("train","valid","test"):
        src=ORIGINAL_GRAPH/f"movie_graph_{split}.npz"
        dst=ROOT/"dataset/ml-1m"/f"movie_graph_{split}.npz"
        if not src.is_file():
            raise FileNotFoundError(f"Original S2 adjacency matrix missing: {src}")
        if not dst.exists():
            dst.parent.mkdir(parents=True,exist_ok=True)
            dst.symlink_to(src)
        if not dst.is_file() or digest(src)!=digest(dst):
            raise RuntimeError(f"Graph data mismatch: {src} vs {dst}")
    print("[NINE-S2-PARITY] original MF, NGCF graphs, S2 recipe verified",flush=True)
    return cfg

def safe_write(path,text):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        f.write(text)

def config_for(cfg,run):
    x=copy.deepcopy(cfg)
    x["model"]["ckpt"]=None
    x["run"]["resume_ckpt_path"]=None
    x["run"]["output_dir"]=str(OUT/run["id"])
    x["run"]["seed"]=run["seed"]
    x["run"]["max_epoch"]=3
    x["run"]["stop_after_n_epochs"]=2
    x["run"]["save_checkpoint_every_n_epochs"]=1
    x["run"]["evaluate"]=False
    x["run"]["valid_splits"]=["valid"]
    x["run"]["test_splits"]=[]
    for key in ("max_train_steps","mid_epoch_eval_iters","iters_per_epoch","max_iters"):
        x["run"].pop(key,None)
    return x

def prep():
    cfg=historical_reference()
    if LOG.exists() or OUT.exists():
        raise FileExistsError(f"Nine-run output/log already exists; refuse overwrite: {LOG} / {OUT}")
    LOG.mkdir(parents=True,exist_ok=False)
    mf=Path(cfg["model"]["rec_config"]["pretrained_path"])
    manifest={"status":"prepared","source_commit":"d818fa2943ede4e9a810b032a3422e94d0c77ed0",
              "recipe_sha256":digest(SOURCE),"pretrained_mf_sha256":digest(mf),
              "s2_reference_valid_ep0_auc":0.7296752596287361,
              "s2_reference_valid_ep1_auc":0.7328835632321707,
              "num_experiments":len(EXPERIMENTS_9),"epochs_each":2,
              "scheduler_horizon":3,"GPU":"0","runs":EXPERIMENTS_9}
    safe_write(LOG/"manifest.json",json.dumps(manifest,indent=2)+"\n")
    for plan in EXPERIMENTS_9:
        path=LOG/(plan["id"]+".yaml")
        safe_write(path,yaml.safe_dump(config_for(cfg,plan),sort_keys=False))
    print("[NINE-PREPARED] 9 independent TWO-epoch experiments; 3-epoch LR horizon",flush=True)
    for plan in EXPERIMENTS_9:
        print(f"[NINE-PLAN] {plan['id']} seed={plan['seed']} mode={plan['mode']} "
              f"branch={plan['branch']} gain={plan['gain']} "
              f"feature={plan['feature']} attention={plan['attention']}",flush=True)

def read_metrics(logpath):
    """Require BOTH full train epochs to have a Valid AUC/UAUC."""
    content=Path(logpath).read_text(errors="replace")
    pairs=[]
    for line in content.splitlines():
        if "Averaged stats:" not in line:
            continue
        match=METRICS.search(line)
        if match:
            pairs.append(tuple(map(float,match.groups())))
    if len(pairs)!=2:
        raise RuntimeError(f"{logpath}: expected two valid epochs, got {pairs}")
    for auc,uauc in pairs:
        if not (0<=auc<=1 and 0<=uauc<=1 and math.isfinite(auc) and math.isfinite(uauc)):
            raise RuntimeError(f"Invalid Valid AUC/UAUC: {(auc,uauc)}")
    if "[S2-EPOCH-LIMIT]" not in content or "completed=2 scheduler_max_epoch=3" not in content:
        raise RuntimeError(f"Two-epoch stop under 3-epoch LR horizon unverified: {logpath}")
    for ep in (0,1):
        if f"Train: data epoch: [{ep}] Total time:" not in content:
            raise RuntimeError(f"{logpath}: missing FULL training epoch {ep}")
    return pairs


def ensure_checkpoints(id_):
    base=OUT/id_
    found=[]
    for ep in (0,1):
        paths=list(base.rglob(f"checkpoint_{ep}.pth"))
        if len(paths)!=1:
            raise RuntimeError(f"{id_} missing or duplicate checkpoint epoch {ep}: {paths}")
        found.append(str(paths[0]))
    best=list(base.rglob("checkpoint_best.pth"))
    if len(best)!=1:
        raise RuntimeError(f"{id_} missing best checkpoint: {best}")
    return found, str(best)


def check_shared(run_id, fp_kind="init"):
    control=LOG/"01_S2_S42.init.json"
    variant=LOG/(run_id+"."+fp_kind+".json")
    if not control.is_file() or not variant.is_file():
        raise RuntimeError(f"{run_id}: missing shared initialization fingerprint")
    baseline=json.loads(control.read_text())["groups"]
    other=json.loads(variant.read_text())["groups"]
    mismatch=[name for name in HASH_GROUPS
              if baseline[name]["sha256"] != other[name]["sha256"]]
    if mismatch:
        raise RuntimeError(f"{run_id}: unequal S2 initial weights: {mismatch}")
    print(f"[NINE-INIT-IDENTICAL] {run_id} {fp_kind} "
          "MF/GNN/Projector/LoRA SHA256 equal to seed-42 S2",flush=True)


def preflight_cora(plan):
    """Fail BEFORE training when shared initialization differs from S2."""
    id_=plan["id"]
    fp=LOG/(id_+".preflight.json")
    logfile=LOG/(id_+".preflight.log")
    if fp.exists() and logfile.exists():
        check_shared(id_, "preflight")
        return
    if fp.exists() or logfile.exists():
        raise RuntimeError(f"{id_}: incomplete CoRA preflight; stop to avoid overwriting")
    cmd=[sys.executable,"-u","experiments/s2_cora/s2_entry.py",
         "--cfg-path",str(LOG/(id_+".yaml")),
         "--fingerprint",str(fp),"--mode","cora",
         "--cora-active",plan["branch"],"--cora-gain",str(plan["gain"]),
         "--cora-feature",plan["feature"],"--cora-attention",plan["attention"],
         "--probe-only"]
    print(f"[NINE-PREFLIGHT] {id_} comparing S2 initialization",flush=True)
    with logfile.open("x",encoding="utf-8") as dest:
        proc=subprocess.Popen(cmd,cwd=ROOT,
                              env={**os.environ,"CUDA_VISIBLE_DEVICES":"0","PYTHONUNBUFFERED":"1"},
                              stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,encoding="utf-8",errors="replace",bufsize=1)
        for line in proc.stdout:
            dest.write(line)
            if "[S2-" in line or "Traceback" in line or "Error:" in line:
                print(f"[{id_}-PREFLIGHT] {line.strip()[:500]}",flush=True)
        rc=proc.wait()
    if rc:
        raise RuntimeError(f"CoRA preflight failed for {id_}: {logfile}")
    if "[S2-PROBE-OK]" not in logfile.read_text(errors="replace"):
        raise RuntimeError(f"CoRA model construction not verified for {id_}")
    check_shared(id_,"preflight")

def run_one(plan):
    id_=plan["id"]
    conf=LOG/(id_+".yaml")
    fp=LOG/(id_+".init.json")
    logfile=LOG/(id_+".log")
    final=LOG/(id_+".result.json")
    if final.exists():
        saved=json.loads(final.read_text())
        if not (conf.is_file() and fp.is_file() and logfile.is_file()):
            raise RuntimeError(f"Partial complete record: {id_}")
        scores=read_metrics(logfile)
        if scores != [tuple(row) for row in saved["valid"]]:
            raise RuntimeError(f"Recorded Valid metric mismatch for {id_}")
        ensure_checkpoints(id_)
        if plan["mode"]=="cora":
            check_shared(id_)
        print(f"[NINE-ALREADY-COMPLETE] {id_} best_auc={saved['auc']:.9f}",flush=True)
        return saved
    if fp.exists() or logfile.exists() or (OUT/id_).exists():
        raise RuntimeError(f"Partial/failed run found {id_}; refuse overwrite")
    if plan["mode"]=="cora":
        preflight_cora(plan)
    cmd=[sys.executable,"-u","experiments/s2_cora/s2_entry.py",
         "--cfg-path",str(conf),"--fingerprint",str(fp),"--mode",plan["mode"]]
    if plan["mode"]=="cora":
        cmd += ["--cora-active",plan["branch"],"--cora-gain",str(plan["gain"]),
                "--cora-feature",plan["feature"],"--cora-attention",plan["attention"]]
    print(f"[NINE-START] {id_} 2epochs {time.strftime('%F %T')}",flush=True)
    with logfile.open("x",encoding="utf-8") as file:
        process=subprocess.Popen(cmd,cwd=ROOT,
                env={**os.environ,"CUDA_VISIBLE_DEVICES":"0","PYTHONUNBUFFERED":"1"},
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,encoding="utf-8",errors="replace",bufsize=1)
        try:
            for line in process.stdout:
                file.write(line)
                file.flush()
                if ("Averaged stats:" in line or "[S2-" in line or
                    "Traceback" in line or "RuntimeError" in line or
                    ("Train: data epoch:" in line and "[11296/" in line)):
                    print(f"[{id_}] {line.strip()[:500]}",flush=True)
            exit_code=process.wait()
        except BaseException:
            process.terminate();process.wait();raise
    if exit_code:
        raise RuntimeError(f"{id_}: process exited {exit_code}, see {logfile}")
    scores=read_metrics(logfile)
    checkpoints,best=ensure_checkpoints(id_)
    if plan["mode"]=="cora":
        check_shared(id_)
        content=logfile.read_text(errors="replace")
        counter=re.search(r"\[S2-CORA-COUNTERS\]\s*(\{[^\n]+\})",content)
        if not counter:
            raise RuntimeError(f"{id_}: CoRA counters missing")
        counts=json.loads(counter.group(1))
        active=("q","v") if plan["branch"]=="both" else (plan["branch"],)
        for tag in active:
            if counts.get(tag+"_cora_forward",0)<=0:
                raise RuntimeError(f"{id_}: CoRA {tag.upper()} never executed")
        for tag in set(("q","v"))-set(active):
            if counts.get(tag+"_cora_forward",0)!=0:
                raise RuntimeError(f"{id_}: supposedly disabled CoRA {tag.upper()} executed")
    best_ep=max(range(2),key=lambda ep:scores[ep][0])
    auc,uauc=scores[best_ep]
    result={**plan,"auc":auc,"uauc":uauc,
            "best_epoch":best_ep,"valid":[list(x) for x in scores],
            "checkpoints":checkpoints,"best_checkpoint":best}
    safe_write(final,json.dumps(result,indent=2)+"\n")
    print(f"[NINE-COMPLETE] {id_} "
          f"E0 AUC={scores[0][0]:.9f} UAUC={scores[0][1]:.9f} "
          f"E1 AUC={scores[1][0]:.9f} UAUC={scores[1][1]:.9f} "
          f"BEST E{best_ep} AUC={auc:.9f}",flush=True)
    return result

def verify_repo():
    orig="d818fa2943ede4e9a810b032a3422e94d0c77ed0"
    # The archived model has only a gated CoRA NGCF view; the trainer adds a post-epoch stop.
    src=ROOT/"minigpt4/models/minigpt4rec_v2.py"
    if "_s2_cora_need_view" not in src.read_text(errors="replace"):
        raise RuntimeError("Wrong base model; expected minimal archived S2 CoRA branch")
    trainer=(ROOT/"minigpt4/runners/runner_base.py").read_text()
    if "[S2-EPOCH-LIMIT]" not in trainer:
        raise RuntimeError("S2 two-epoch cap missing; never shorten max_epoch")
    if not (ROOT/"experiments/s2_cora/s2_entry.py").is_file():
        raise RuntimeError("S2 CoRA entrypoint missing")
    print(f"[NINE-SOURCE-OK] historical_base={orig} S2 3-epoch scheduler retained",flush=True)

def make_summary():
    results=[]
    for plan in EXPERIMENTS_9:
        file=LOG/(plan["id"]+".result.json")
        if file.is_file():
            results.append(json.loads(file.read_text()))
    if not results:
        return
    control=next((x for x in results if x["id"]=="01_S2_S42"),None)
    path=LOG/"summary.csv.tmp"
    with path.open("w",newline="",encoding="utf-8") as out:
        writer=csv.writer(out)
        writer.writerow(("experiment","seed","mode","cora_active","gain","feature","attention",
                         "epoch0_auc","epoch0_uauc","epoch1_auc","epoch1_uauc",
                         "best_epoch","best_auc","best_uauc",
                         "delta_best_auc_vs_S2","delta_best_uauc_vs_S2"))
        for row in results:
            (a0,u0),(a1,u1)=row["valid"]
            da=(row["auc"]-control["auc"]) if control and row["mode"]=="cora" else None
            du=(row["uauc"]-control["uauc"]) if control and row["mode"]=="cora" else None
            writer.writerow((row["id"],row["seed"],row["mode"],row["branch"],row["gain"],
                             row["feature"],row["attention"],
                             f"{a0:.9f}",f"{u0:.9f}",f"{a1:.9f}",f"{u1:.9f}",
                             row["best_epoch"],f'{row["auc"]:.9f}',f'{row["uauc"]:.9f}',
                             f"{da:+.9f}" if da is not None else "",
                             f"{du:+.9f}" if du is not None else ""))
    path.replace(LOG/"summary.csv")
    print(f"[NINE-SUMMARY] completed={len(results)}/9 path={LOG/'summary.csv'}",flush=True)

def execute():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must equal 0")
    verify_repo()
    historical_reference()
    if not LOG.is_dir():
        raise FileNotFoundError("Run prepare first; YAMLs missing")
    approved=historical_reference_silent()
    for p in EXPERIMENTS_9:
        yamlfile=LOG/(p["id"]+".yaml")
        if not yamlfile.is_file():
            raise FileNotFoundError(yamlfile)
        cfg=yaml.safe_load(yamlfile.read_text())
        if cfg!=config_for(approved,p):
            raise RuntimeError(f"{p['id']} config differs from historical S2 recipe")
        if p["mode"]=="cora" and not (LOG/"01_S2_S42.result.json").is_file():
            raise RuntimeError("S2 control must finish before all CoRA arms")
        result=run_one(p)
        if p["id"]=="01_S2_S42":
            if result["auc"]<0.72:
                make_summary()
                raise RuntimeError(
                    f"Historical S2 best of 2 AUC {result['auc']:.6f}<0.72; STOP all CoRA runs")
            print(f"[NINE-S2-REPRODUCED] best_auc={result['auc']:.9f} "
                  f"historical_ep1=0.732883563",flush=True)
        make_summary()
    print("[NINE-ALL-COMPLETE] 9 variants, fixed seed=42, two full epochs each, Valid only",flush=True)

_source_cache=None
def historical_reference_silent():
    global _source_cache
    if _source_cache is None:
        _source_cache=yaml.safe_load(SOURCE.read_text())
    return _source_cache

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("prepare","run","summary"))
    args=parser.parse_args()
    os.chdir(ROOT)
    if args.action=="prepare":
        verify_repo()
        prep()
    elif args.action=="run":
        execute()
    else:
        make_summary()
