#!/usr/bin/env python3
"""One-click historical NO31 CTR+KD / NO31 CTR+KD + CoRA comparison.

Defaults: seed42, 1 complete epoch each, Valid only, GPU0, foreground,
original three-epoch cosine LR horizon. Original MF checkpoint retained.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"tran_configs/S2_groups_acc3_recovered.yaml"
LOG=ROOT/"logs/NO31_CTRKD_CORA_20261023"
OUT=ROOT/"output/NO31_CTRKD_CORA_20261023"
RECIPE_CHECK={
    ("model","gnn_semantic_align"):True,
    ("model","gnn_semantic_weight"):1e-4,
    ("model","proj_reg_weight"):5e-4,
    ("model","gcn_layer"):31,
    ("model","lora_gnn"):"ngcf",
    ("model","rec_model"):"MF",
    ("run","seed"):42,
    ("run","batch_size_train"):3,
    ("run","accum_grad_iters"):3,
    ("run","max_epoch"):3,
    ("run","lr_scale_rec"):0.5,
    ("run","lr_scale_gnn"):0.25,
    ("run","lr_scale_proj"):0.5,
    ("run","lr_scale_lora"):1.0,
}
ARMS=("no31","cora")
METRICS=re.compile(r"\*\*\*auc:\s*([0-9.]+).*?\*\*\*uauc:\s*\(\s*([0-9.]+)")


def prepare(epoch_count):
    if LOG.exists() or OUT.exists():
        raise FileExistsError(f"Refusing to overwrite previous NO31 runs: {LOG}, {OUT}")
    base=yaml.safe_load(SOURCE.read_text())
    for (section,k),expected in RECIPE_CHECK.items():
        got=base[section].get(k)
        if got != expected:
            raise RuntimeError(f"Original S2 recipe mismatch {section}.{k}: {got!r} != {expected!r}")
    m,r=base["model"],base["run"]
    if m.get("ckpt") or r.get("resume_ckpt_path"):
        raise RuntimeError("Do not start from trained S2 checkpoint")
    if any(m.get(k) is not False for k in ("freeze_rec","freeze_lora","freeze_proj")):
        raise RuntimeError("Original S2 training freeze flags changed")
    if not Path(m["rec_config"]["pretrained_path"]).is_file():
        raise FileNotFoundError("Historical pretrained MF checkpoint not found")
    if not Path(m["llama_model"]).is_dir():
        raise FileNotFoundError("Original Vicuna checkpoint not found")
    src=Path("/home/bbx/user_lx/GraphLoRA/dataset/ml-1m")
    if not src.is_dir():
        raise FileNotFoundError(src)
    dst=ROOT/"dataset/ml-1m"
    for split in ("train","valid","test"):
        old=src/f"movie_graph_{split}.npz"
        new=dst/f"movie_graph_{split}.npz"
        if not old.is_file(): raise FileNotFoundError(old)
        if not new.exists():
            new.parent.mkdir(parents=True,exist_ok=True)
            new.symlink_to(old)
        if not new.is_file():raise FileNotFoundError(new)
        def checksum(path):
            h=hashlib.sha256()
            with path.open("rb") as f:
                for ch in iter(lambda:f.read(1048576),b""):
                    h.update(ch)
            return h.hexdigest()
        if checksum(old)!=checksum(new):
            raise RuntimeError(f"Graph NPZ differs from historical S2: {new}")
    LOG.mkdir(parents=True,exist_ok=False)
    for arm in ARMS:
        cfg=copy.deepcopy(base)
        cfg["model"]["gnn_scale"]=0.0
        cfg["model"]["gnn_conf_loss_weight"]=0.05
        cfg["model"]["ckpt"]=None
        cfg["run"]["resume_ckpt_path"]=None
        cfg["run"]["max_epoch"]=3
        cfg["run"]["seed"]=42
        cfg["run"]["evaluate"]=False
        cfg["run"]["valid_splits"]=["valid"]
        cfg["run"]["test_splits"]=[]
        cfg["run"]["save_checkpoint_every_n_epochs"]=1
        cfg["run"]["output_dir"]=str(OUT/arm)
        (LOG/f"{arm}.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False))
    a=yaml.safe_load((LOG/"no31.yaml").read_text())
    b=yaml.safe_load((LOG/"cora.yaml").read_text())
    a["run"].pop("output_dir")
    b["run"].pop("output_dir")
    if a!=b:raise RuntimeError("Control and CoRA YAMLs differ beyond output_dir")
    (LOG/"manifest.json").write_text(json.dumps({
        "status":"prepared","source":"archived NO31 Night8 training patch",
        "historic_commit":"d818fa2943ede4e9a810b032a3422e94d0c77ed0",
        "gnn_scale":0,"input_alpha":0,"ctr":0.05,"kd":0.05,
        "seed":42,"epochs":epoch_count,"scheduler_max_epoch":3,
        "cora_default":"Q+V ID-only, gain=.1, detach GNN",
        "no_graphlora_warmstart":True,"mf_pretrained":m["rec_config"]["pretrained_path"]
    },indent=2))
    print(f"[NO31-CORA-PREPARED] 2 arms, {epoch_count} epoch(s) each, "
          "gnn_scale=0 CTR=.05 KD=.05 input_alpha=0 seed42",flush=True)


def launch(arm,epoch_count,probe=False):
    suffix="_probe" if probe else "_train"
    fp=LOG/(arm+suffix+"_init.json")
    log=LOG/(arm+suffix+".log")
    if probe and fp.is_file() and log.is_file():
        # Init probes have no optimizer state, forward passes or checkpoints.
        # Completed probes can be reused without modifying any existing files.
        # An older CoRA probe was incorrectly marked failed AFTER successful
        # model construction by checking training-only Q/V forward counters.
        # Accept ONLY that exact known post-probe error, never other failures.
        log_text=log.read_text(errors="replace")
        marker=f"[NO31-PROBE-OK] mode={arm}"
        old_counter_bug=(
            arm == "cora"
            and "RuntimeError: CoRA branch q never executed" in log_text
            and '[NO31-CORA-COUNTERS] {"initialized": 1}' in log_text
        )
        clean=(
            "Traceback (most recent call last):" not in log_text
            and "[NO31-CORA-COUNTERS]" not in log_text
        )
        if marker not in log_text or not (clean or old_counter_bug):
            raise RuntimeError(f"Unsafe existing probe; inspect before reuse: {log}")
        if "Train: data epoch:" in log_text or "[NIGHT8-STOP]" in log_text:
            raise RuntimeError(f"Probe unexpectedly contains training: {log}")
        saved=json.loads(fp.read_text())
        if saved.get("mode") != arm or set(saved.get("groups",{})) != {
            "MF","GNN","Projector","LoRA"
        }:
            raise RuntimeError(f"Invalid probe initial fingerprint: {fp}")
        if any(not saved["groups"][g].get("sha256") for g in
               ("MF","GNN","Projector","LoRA")):
            raise RuntimeError(f"Incomplete probe fingerprint: {fp}")
        print(f"[NO31-PROBE-REUSED] {arm} "
              f"status={'recovered-old-counter-bug' if old_counter_bug else 'clean'} "
              f"hashes_verified", flush=True)
        return None
    if fp.exists() or log.exists():
        raise FileExistsError(f"Previous NO31 {arm} run/probe exists: {fp}, {log}")
    cmd=[sys.executable,"-u","experiments/no31_cora/entry.py",
         "--cfg-path",str(LOG/(arm+".yaml")),"--mode",arm,
         "--init-hash",str(fp)]
    if probe:cmd+=["--probe-only"]
    if arm=="cora":
        cmd+=["--cora-gain","0.1","--cora-detach-gnn","true",
              "--cora-active","both"]
    env={**os.environ,"CUDA_VISIBLE_DEVICES":"0","S2_INPUT_ALPHA":"0.00",
         "S2_SMALL_KD":"0.05","S2_NIGHT_EPOCHS":str(epoch_count),
         "PYTHONUNBUFFERED":"1"}
    print(f"[NO31-CORA-RUN] {arm} phase={suffix} epochs={epoch_count}",flush=True)
    with log.open("x",encoding="utf-8") as dest:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,
                 stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                 text=True,encoding="utf-8",errors="replace",bufsize=1)
        try:
            for line in proc.stdout:
                dest.write(line);dest.flush()
                if ("[NO31" in line or "[NIGHT8-" in line or
                    "Averaged stats:" in line or "Traceback" in line or
                    "Error:" in line):
                    print(f"[{arm}] {line.strip()[:550]}",flush=True)
            code=proc.wait()
        except BaseException:
            proc.terminate();proc.wait();raise
    if code:
        raise RuntimeError(f"{arm} process failed code={code}: {log}")
    if probe:
        if "[NO31-PROBE-OK]" not in log.read_text(errors="replace"):
            raise RuntimeError(f"{arm} model initialization probe missing")
        return None
    data=log.read_text(errors="replace")
    if f"[NIGHT8-STOP] trained_epochs={epoch_count}" not in data:
        raise RuntimeError(f"Historical epoch stop not confirmed: {arm}")
    if "[NIGHT8-MODEL] gnn_scale=0 alpha=0.0" not in data:
        raise RuntimeError(f"Original NO31 recipe not confirmed: {arm}")
    if "[NIGHT8-LOSS]" not in data:
        raise RuntimeError(f"Original CTR/KD losses not executed: {arm}")
    scores=[]
    for line in data.splitlines():
        if "Averaged stats:" in line:
            match=METRICS.search(line)
            if match:scores.append(tuple(map(float,match.groups())))
    if len(scores)!=epoch_count:
        raise RuntimeError(f"{arm}: expected {epoch_count} Valid rows: {scores}")
    if arm=="cora":
        if "[NO31-CORA-COUNTERS]" not in data:
            raise RuntimeError("CoRA runtime did not report activity")
        counters=re.findall(r"\[NO31-CORA-COUNTERS\]\s*(\{[^\n]+\})",data)
        parsed=json.loads(counters[-1])
        if any(parsed.get(key,0)<=0 for key in ("q_id_forward","v_id_forward")):
            raise RuntimeError(f"Q/V CoRA not exercised: {parsed}")
    checkpoints={}
    for ep in range(epoch_count):
        paths=list((OUT/arm).rglob(f"checkpoint_{ep}.pth"))
        if len(paths)!=1: raise RuntimeError(f"{arm} incomplete checkpoint {ep}: {paths}")
        checkpoints[ep]=str(paths[0])
    best_ckpts=list((OUT/arm).rglob("checkpoint_best.pth"))
    if len(best_ckpts)!=1:raise RuntimeError(f"{arm} best checkpoint missing")
    result={"arm":arm,"epochs":epoch_count,
            "valid":[{"epoch":i,"auc":a,"uauc":u} for i,(a,u) in enumerate(scores)],
            "best_epoch":max(range(epoch_count),key=lambda i:scores[i][0]),
            "checkpoints":checkpoints,"best_ckpt":str(best_ckpts[0])}
    (LOG/(arm+"_result.json")).write_text(json.dumps(result,indent=2))
    return result


def check_hashes(suffix):
    control=json.loads((LOG/("no31_"+suffix+"_init.json")).read_text())["groups"]
    cora=json.loads((LOG/("cora_"+suffix+"_init.json")).read_text())["groups"]
    bad=[g for g in ("MF","GNN","Projector","LoRA")
         if control[g]["sha256"]!=cora[g]["sha256"]]
    if bad:raise RuntimeError(f"CoRA unfair initial shared weights: {bad}")
    print("[NO31-INIT-EQUAL] "+suffix+" MF GNN Projector LoRA SHA256 identical",flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("prepare","run"))
    p.add_argument("--epochs",type=int,default=1)
    args=p.parse_args()
    if args.epochs not in (1,2):raise RuntimeError("Exactly one or two full epochs")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
        raise RuntimeError("Export CUDA_VISIBLE_DEVICES=0")
    os.chdir(ROOT)
    if args.action=="prepare":
        prepare(args.epochs)
        return
    if not (LOG/"no31.yaml").is_file() or not (LOG/"cora.yaml").is_file():
        raise RuntimeError("First run: python experiments/no31_cora/run_pair.py prepare")
    meta=json.loads((LOG/"manifest.json").read_text())
    if int(meta["epochs"])!=args.epochs:raise RuntimeError("Requested epochs don't match prepared manifest")
    launch("no31",args.epochs,probe=True)
    launch("cora",args.epochs,probe=True)
    check_hashes("probe")
    a=launch("no31",args.epochs,probe=False)
    if a["valid"][0]["auc"]<0.69:
        raise RuntimeError(f"Historical NO31 control collapsed: {a['valid'][0]['auc']:.6f}; do not run CoRA")
    b=launch("cora",args.epochs,probe=False)
    check_hashes("train")
    print("[NO31-CORA-COMPLETE] S2-no31 control vs CoRA",flush=True)
    for i in range(args.epochs):
        print(f"epoch={i} baselineAUC={a['valid'][i]['auc']:.9f} "
              f"CoRA_AUC={b['valid'][i]['auc']:.9f} "
              f"delta={b['valid'][i]['auc']-a['valid'][i]['auc']:+.9f} "
              f"baselineUAUC={a['valid'][i]['uauc']:.9f} "
              f"CoRA_UAUC={b['valid'][i]['uauc']:.9f}",flush=True)


if __name__=="__main__":
    main()
