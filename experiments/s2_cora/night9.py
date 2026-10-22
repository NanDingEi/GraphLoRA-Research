#!/usr/bin/env python3
"""Nine controlled S2 vs CoRA experiments, all from the original S2 recipe.

Each experiment starts a FRESH model: pretrained historical MF and Vicuna,
fresh GNN/Projector/LoRA, optional fresh CoRA. No GraphLoRA/S2 checkpoints.
One complete train+Valid epoch per experiment, retaining 3-epoch cosine horizon.
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
    {"id":"01_S2_S42","mode":"baseline","seed":42,"branch":"both","gain":1.0},
    {"id":"02_CORA_S42","mode":"cora","seed":42,"branch":"both","gain":1.0},
    {"id":"03_CORA_Q_S42","mode":"cora","seed":42,"branch":"q","gain":1.0},
    {"id":"04_CORA_V_S42","mode":"cora","seed":42,"branch":"v","gain":1.0},
    {"id":"05_CORA_HALF_S42","mode":"cora","seed":42,"branch":"both","gain":0.5},
    {"id":"06_S2_S43","mode":"baseline","seed":43,"branch":"both","gain":1.0},
    {"id":"07_CORA_S43","mode":"cora","seed":43,"branch":"both","gain":1.0},
    {"id":"08_S2_S44","mode":"baseline","seed":44,"branch":"both","gain":1.0},
    {"id":"09_CORA_S44","mode":"cora","seed":44,"branch":"both","gain":1.0},
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
    x["run"]["stop_after_n_epochs"]=1
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
              "num_experiments":len(EXPERIMENTS_9),"epochs_each":1,
              "scheduler_horizon":3,"GPU":"0","runs":EXPERIMENTS_9}
    safe_write(LOG/"manifest.json",json.dumps(manifest,indent=2)+"\n")
    for plan in EXPERIMENTS_9:
        path=LOG/(plan["id"]+".yaml")
        safe_write(path,yaml.safe_dump(config_for(cfg,plan),sort_keys=False))
    print("[NINE-PREPARED] 9 independent one-epoch experiments; 3-epoch LR horizon",flush=True)
    for plan in EXPERIMENTS_9:
        print(f"[NINE-PLAN] {plan['id']} seed={plan['seed']} mode={plan['mode']} "
              f"branch={plan['branch']} gain={plan['gain']}",flush=True)

def read_metrics(logpath):
    text=Path(logpath).read_text(errors="replace")
    pairs=[]
    for line in text.splitlines():
        match=METRICS.search(line)
        if match:
            pairs.append(tuple(map(float,match.groups())))
    if len(pairs)!=1:
        raise RuntimeError(f"{logpath}: expected exactly one Valid AUC/UAUC, got {pairs}")
    a,u=pairs[0]
    if not (0<=a<=1 and 0<=u<=1 and math.isfinite(a) and math.isfinite(u)):
        raise RuntimeError("Nonfinite or invalid Valid metric")
    if "[S2-EPOCH-LIMIT]" not in text:
        raise RuntimeError(f"Runner didn't preserve S2 3-epoch horizon with 1-epoch stop: {logpath}")
    return a,u

def run_one(plan):
    id_=plan["id"]
    conf=LOG/(id_+".yaml")
    fp=LOG/(id_+".init.json")
    logfile=LOG/(id_+".log")
    final=LOG/(id_+".result.json")
    if final.exists():
        saved=json.loads(final.read_text())
        if not (conf.is_file() and fp.is_file() and logfile.is_file()):
            raise RuntimeError(f"Partial result for {id_}; do not silently skip")
        auc,uauc=read_metrics(logfile)
        if abs(auc-saved["auc"])>1e-10 or abs(uauc-saved["uauc"])>1e-10:
            raise RuntimeError(f"Result is corrupted: {id_}")
        print(f"[NINE-ALREADY-COMPLETE] {id_} auc={auc:.9f} uauc={uauc:.9f}",flush=True)
        return saved
    if fp.exists() or logfile.exists() or (OUT/id_).exists():
        raise RuntimeError(f"Partial/failed run found {id_}; fail closed to avoid losing data")
    cmd=[sys.executable,"-u","experiments/s2_cora/s2_entry.py",
         "--cfg-path",str(conf),"--fingerprint",str(fp),"--mode",plan["mode"]]
    if plan["mode"]=="cora":
        cmd += ["--cora-active",plan["branch"],"--cora-gain",str(plan["gain"])]
    print(f"[NINE-START] {id_} {time.strftime('%F %T')}",flush=True)
    with logfile.open("x",encoding="utf-8") as file:
        process=subprocess.Popen(cmd,cwd=ROOT,
                env={**os.environ,"CUDA_VISIBLE_DEVICES":"0","PYTHONUNBUFFERED":"1"},
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,encoding="utf-8",errors="replace",bufsize=1)
        try:
            for line in process.stdout:
                file.write(line);file.flush()
                if ("Averaged stats:" in line or "[S2-" in line or
                    "Traceback" in line or "RuntimeError" in line or
                    ("Train: data epoch:" in line and "[11296/" in line)):
                    print(f"[{id_}] {line.strip()[:500]}",flush=True)
            exit_code=process.wait()
        except BaseException:
            process.terminate();process.wait();raise
    if exit_code:
        raise RuntimeError(f"{id_}: process exited {exit_code}, see {logfile}")
    auc,uauc=read_metrics(logfile)
    checkpoints=list((OUT/id_).rglob("checkpoint_0.pth"))
    if len(checkpoints)!=1:
        raise RuntimeError(f"{id_}: expected 1 epoch-0 checkpoint, got {checkpoints}")
    if plan["mode"]=="cora":
        content=logfile.read_text(errors="replace")
        counter=re.search(r"\[S2-CORA-COUNTERS\]\s*(\{[^\n]+\})",content)
        if not counter:
            raise RuntimeError(f"{id_}: CoRA forward counter missing")
        counts=json.loads(counter.group(1))
        active=("q","v") if plan["branch"]=="both" else (plan["branch"],)
        for tag in active:
            if counts.get(tag+"_cora_forward",0)<=0:
                raise RuntimeError(f"{id_}: {tag.upper()} CoRA branch not executed")
    result={**plan,"auc":auc,"uauc":uauc,"checkpoint":str(checkpoints[0])}
    safe_write(final,json.dumps(result,indent=2)+"\n")
    print(f"[NINE-COMPLETE] {id_} AUC={auc:.9f} UAUC={uauc:.9f}",flush=True)
    return result

def check_shared(seed, run_id):
    control=LOG/(f"{'01_S2_S42' if seed==42 else '06_S2_S43' if seed==43 else '08_S2_S44'}.init.json")
    variant=LOG/(run_id+".init.json")
    if not control.exists() or not variant.exists():
        raise RuntimeError(f"Seed {seed}: missing baseline/CoRA initial fingerprint")
    a=json.loads(control.read_text())["groups"]
    b=json.loads(variant.read_text())["groups"]
    mismatch=[name for name in HASH_GROUPS if a[name]["sha256"]!=b[name]["sha256"]]
    if mismatch:
        raise RuntimeError(f"{run_id}: shared initial weights differ for {mismatch}")
    print(f"[NINE-SHARED-INIT-PASS] {run_id} seed={seed} "
          "MF/GNN/Projector/LoRA hashes identical",flush=True)

def verify_repo():
    orig="d818fa2943ede4e9a810b032a3422e94d0c77ed0"
    # At the time these experiments were authored the only model modification
    # was guarded CoRA GNN penult exposure. No need to reset user's worktree.
    src=ROOT/"minigpt4/models/minigpt4rec_v2.py"
    if "_s2_cora_need_view" not in src.read_text(errors="replace"):
        raise RuntimeError("Wrong base model; expected minimal archived S2 CoRA branch")
    trainer=(ROOT/"minigpt4/runners/runner_base.py").read_text()
    if "[S2-EPOCH-LIMIT]" not in trainer:
        raise RuntimeError("S2 one-epoch cap not installed; never shorten max_epoch")
    if not (ROOT/"experiments/s2_cora/s2_entry.py").is_file():
        raise RuntimeError("S2 CoRA entrypoint missing")
    print(f"[NINE-SOURCE-OK] historical_base={orig} S2 3-epoch scheduler retained",flush=True)

def make_summary():
    table=[]
    lookup={}
    for plan in EXPERIMENTS_9:
        path=LOG/(plan["id"]+".result.json")
        if path.exists():
            row=json.loads(path.read_text())
            table.append(row)
            lookup[plan["id"]]=row
    if not table:
        return
    tmp=LOG/"summary.csv.tmp"
    with tmp.open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(("experiment","seed","mode","cora_active","gain","valid_auc",
                         "valid_uauc","delta_auc_vs_same_seed_s2","delta_uauc_vs_same_seed_s2"))
        for row in table:
            ctrl=lookup.get(
                {42:"01_S2_S42",43:"06_S2_S43",44:"08_S2_S44"}[row["seed"]])
            da=row["auc"]-ctrl["auc"] if ctrl and row["mode"]=="cora" else None
            du=row["uauc"]-ctrl["uauc"] if ctrl and row["mode"]=="cora" else None
            writer.writerow((row["id"],row["seed"],row["mode"],row["branch"],row["gain"],
                             f'{row["auc"]:.9f}',f'{row["uauc"]:.9f}',
                             f'{da:+.9f}' if da is not None else "",
                             f'{du:+.9f}' if du is not None else ""))
    tmp.replace(LOG/"summary.csv")
    print("[NINE-SUMMARY] completed="+str(len(table))+"/9 path="+str(LOG/"summary.csv"),flush=True)

def execute():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must equal 0")
    verify_repo()
    historical_reference()
    if not LOG.is_dir():
        raise FileNotFoundError("Run prepare first; YAMLs missing")
    seed_results={}
    for p in EXPERIMENTS_9:
        yamlfile=LOG/(p["id"]+".yaml")
        cfg=yaml.safe_load(yamlfile.read_text())
        if cfg!=config_for(historical_reference_silent(),p):
            raise RuntimeError(f"{p['id']} config differs from approved S2 recipe")
        if p["mode"]=="cora":
            # Fresh CoRA training cannot proceed until same-seed S2 is complete.
            control_id={42:"01_S2_S42",43:"06_S2_S43",44:"08_S2_S44"}[p["seed"]]
            if control_id not in seed_results:
                raise RuntimeError(f"{p['id']} preceding control missing")
        result=run_one(p)
        seed_results[p["id"]]=result
        if p["id"]=="01_S2_S42" and result["auc"]<0.72:
            make_summary()
            raise RuntimeError(f"Historical S2 ep0 AUC {result['auc']:.6f}<0.72; STOP all CoRA runs")
        if p["mode"]=="cora":
            check_shared(p["seed"],p["id"])
        make_summary()
    print("[NINE-ALL-COMPLETE] 9 experiments each one real epoch, Valid only",flush=True)

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
