#!/usr/bin/env python3
"""R2-06/07/08: matched S2 one-epoch continuation with new internal injection.

Foreground subprocess, GPU0 only. No automatic multi-arm loops. Each
--round is independently warm-started from the EXACT original S2
checkpoint, same P2B source config/scheduler/optimizer. Unique outputs.
"""
import argparse
import copy
import datetime
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT)
SOURCE=ROOT/"logs/collab_v1/S2_L30_AUX_P0_20261021_165700/P2B_S2_CACHE_FIXED_FULL.yaml"
S2=Path("/home/bbx/user_lx/GraphLoRA/logs/S2_STRICT_20261017_143611/"
        "L31_output/20261017143/checkpoint_best.pth")
MODES={"06":"relation_v","08":"relation_shuffle"}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda:f.read(1024*1024),b""):
            h.update(data)
    return h.hexdigest()


def repo_head():
    return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()


def probe_gate(arm):
    probe02=ROOT/"logs/collab_v1/R2_02_PROBE.json"
    probe05=ROOT/"logs/collab_v1/R2_05_PROBE.json"
    for path in (probe02,probe05):
        if not path.is_file():
            raise FileNotFoundError(f"Complete R2 probes before training: {path}")
    a=json.loads(probe02.read_text())
    b=json.loads(probe05.read_text())
    if a.get("status")!="complete" or b.get("status")!="complete":
        raise RuntimeError("Incomplete R2 probe results")
    if a["results"]["audit"]["counters"].get("audited_tokens",0)<=0:
        raise RuntimeError("R2-02 saw no V graph injections")
    for k in ("relation_v","relation_shuffle"):
        row=b["results"].get(k,{})
        if row.get("counters",{}).get("history_tokens",0)<=0:
            raise RuntimeError(f"R2-05 saw no valid historical tokens for {k}")
    c=b["score_differences_vs_native"]["relation_v"]
    if c["fraction_nonzero"]<=0:
        raise RuntimeError("R2-05 candidate-aware V has zero native logit impact")
    if arm=="07":
        probe03=ROOT/"logs/collab_v1/R2_03_PROBE.json"
        if not probe03.is_file():
            raise FileNotFoundError(probe03)
        d=json.loads(probe03.read_text())
        if d.get("status")!="complete":
            raise RuntimeError("R2-03 incomplete")
        q=d["score_differences_vs_native"]["decision_q"]
        # Branch selection predetermined and reproducible; not chosen from AUC.
        mode="relation_v_q" if q["mean_abs"]>1e-6 else "relation_v_raw"
        return mode,{"q_mean_abs":q["mean_abs"],
                     "rule":"mean_abs > 1e-6 => relation_v_q else relation_v_raw"}
    return MODES[arm],{}


def control_config(source,mode,arm,root,steps):
    cfg=copy.deepcopy(source)
    m,r=cfg["model"],cfg["run"]
    if Path(m["ckpt"]).resolve()!=S2.resolve():
        raise RuntimeError("P2B source must warm-start original S2")
    if (str(m.get("collab_mode"))!="legacy" or int(m.get("gcn_layer",0))!=31 or
        abs(float(m.get("gnn_scale",0))-.1)>1e-10 or
        float(m.get("l30_aux_weight",0))!=0 or
        not bool(m.get("fix_autocast_cache",False)) or
        bool(m.get("freeze_lora",False))):
        raise RuntimeError("P2B source is not controlled, FX-fixed S2 legacy")
    if abs(float(r.get("init_lr",0))-2e-5)>1e-10:
        raise RuntimeError("Expected exact P2B LR=2e-5")
    if int(r.get("seed",-1))!=42:
        raise RuntimeError("Expected exact P2B seed=42")
    if int(r.get("batch_size_train",0))!=3:
        raise RuntimeError("Expected P2B bs=3")
    if int(r.get("accum_grad_iters",0))!=3:
        raise RuntimeError("Expected P2B gradient accumulation=3")
    m["r2_mode"]=mode
    r["output_dir"]=str((root/"output"/arm/("SMOKE" if steps else "FULL_E1")).resolve())
    r["evaluate"]=False
    r["resume_ckpt_path"]=None
    r["skip_validation"]=bool(steps)
    r["diagnostic_stop_after_epoch0"]=True
    r["stop_after_n_epochs"]=1
    r["max_epoch"]=3
    if steps:
        r["max_train_steps"]=int(steps)
    else:
        r.pop("max_train_steps",None)
    for stale in ("iters_per_epoch","max_iters","mid_epoch_eval_iters"):
        r.pop(stale,None)
    return cfg


def run_to_log(cmd,path):
    with path.open("w",encoding="utf-8") as f:
        with subprocess.Popen([str(v) for v in cmd],cwd=ROOT,
                              stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              bufsize=1,text=True,errors="replace") as process:
            try:
                for ln in process.stdout:
                    f.write(ln)
                    if any(x in ln for x in (
                        "[R2-","Averaged stats:","[AMP]",
                        "Train: data epoch:", "Evaluation Total time:",
                        "Traceback","Error:","Saving checkpoint")):
                        print(ln[:700].strip(),flush=True)
                rc=process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
    if rc:
        raise RuntimeError(f"R2 process failed rc={rc} log={path}")


def eval_result(log):
    regex=re.compile(
        r"\*\*\*auc:\s*([0-9.eE+-]+).*?\*\*\*uauc:\(\s*([0-9.eE+-]+)")
    rows=[]
    for line in log.read_text(encoding="utf-8",errors="replace").splitlines():
        if "Averaged stats:" in line:
            m=regex.search(line)
            if m:
                rows.append((float(m.group(1)),float(m.group(2))))
    if len(rows)!=1 or not all(math.isfinite(x) for x in rows[0]):
        raise RuntimeError(f"Expected exactly one FULL Valid AUC/UAUC: {rows}")
    return rows[0]


def preflight():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0 before R2 training")
    if not S2.is_file() or not SOURCE.is_file():
        raise FileNotFoundError("Missing original S2 checkpoint or P2B reference YAML")
    p=subprocess.run(["pgrep","-af","[t]rain_collm_mf_din.py|[r]2_runtime_entry.py"],
                     capture_output=True,text=True)
    if p.returncode==0 and p.stdout.strip():
        raise RuntimeError("Another GraphLoRA train may be active:\n"+p.stdout)
    diff=subprocess.run(["git","diff","--quiet","HEAD","--",
                          "minigpt4/models/minigpt4rec_v2.py",
                          "minigpt4/runners/runner_base.py",
                          "experiments/collab_injection_v1/r2_collab_runtime.py"],
                        cwd=ROOT)
    if diff.returncode:
        raise RuntimeError("Core/experiment source modified since commit; refuse training")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--round",required=True,choices=("06","07","08"))
    p.add_argument("--root",default="logs/collab_v1/R2_EIGHT_20261022")
    p.add_argument("--smoke-steps",type=int,default=24)
    p.add_argument("--prepare-only",action="store_true")
    args=p.parse_args()
    if args.smoke_steps!=24:
        raise ValueError("Only the predefined 24-step smoke is allowed")
    preflight()
    mode,branch=probe_gate(args.round)
    source=yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    base=Path(args.root).resolve()
    arm="R2_"+args.round
    folder=base/arm
    if folder.exists():
        raise FileExistsError(f"R2 arm folder exists, refusing overwrite: {folder}")
    folder.mkdir(parents=True)
    info={"round":args.round,"mode":mode,"gate":branch,
          "source_commit":repo_head(),"original_s2":str(S2),
          "original_s2_sha256":sha(S2),
          "source_p2b":str(SOURCE),"source_p2b_sha256":sha(SOURCE),
          "status":"prepared"}
    smoke=control_config(source,mode,arm,base,24)
    full=control_config(source,mode,arm,base,None)
    for label,cfg in (("SMOKE",smoke),("FULL_E1",full)):
        path=folder/(label+".yaml")
        if path.exists():
            raise FileExistsError(path)
        path.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
        info[label+"_yaml_sha256"]=sha(path)
    manifest=folder/"manifest.json"
    def write():
        manifest.write_text(json.dumps(info,indent=2,ensure_ascii=False))
    write()
    print(f"[R2-{args.round}-PREPARED] mode={mode} "
          f"config={folder/'FULL_E1.yaml'}",flush=True)
    if args.prepare_only:
        return
    if not (ROOT/"output").exists():
        (ROOT/"output").mkdir()
    try:
        info["status"]="smoke_running"
        write()
        run_to_log([sys.executable,"-u",
            "experiments/collab_injection_v1/r2_runtime_entry.py",
            "--cfg-path",folder/"SMOKE.yaml","--mode",mode],
            folder/"SMOKE.log")
        data=(folder/"SMOKE.log").read_text(encoding="utf-8",errors="replace")
        if "[R2-RUNTIME]" not in data or "[R2-COUNTERS]" not in data:
            raise RuntimeError("R2 runtime was not attached")
        m=re.search(r"\[R2-COUNTERS\]\s*(\{[^\n]+\})",data)
        if not m:
            raise RuntimeError("R2 smoke runtime summary missing")
        counters=json.loads(m.group(1))["counters"]
        if args.round in ("06","07","08") and counters.get("weighted_v_tokens",0)<=0:
            raise RuntimeError("R2 smoke never activated V weighting")
        if mode=="relation_v_q" and counters.get("decision_q_hook",0)<=0:
            raise RuntimeError("R2 smoke never activated decision-Q hook")
        info["smoke_runtime_counters"]=counters
        info["status"]="full_running"
        write()
        run_to_log([sys.executable,"-u",
            "experiments/collab_injection_v1/r2_runtime_entry.py",
            "--cfg-path",folder/"FULL_E1.yaml","--mode",mode],
            folder/"FULL_E1.log")
        auc,uauc=eval_result(folder/"FULL_E1.log")
        matches=list((base/"output"/arm/"FULL_E1").rglob("checkpoint_best.pth"))
        if len(matches)!=1:
            raise RuntimeError(f"Expected exactly one R2 best checkpoint: {matches}")
        info.update({"status":"complete","auc":auc,"uauc":uauc,
                     "delta_auc_vs_P2B":auc-.7214924624586214,
                     "delta_uauc_vs_P2B":uauc-.6647772423708141,
                     "best_checkpoint":str(matches[0])})
        write()
        print(f"[R2-{args.round}-RESULT] mode={mode} AUC={auc:.9f} "
              f"UAUC={uauc:.9f} dAUC_P2B={info['delta_auc_vs_P2B']:+.9f} "
              f"dUAUC_P2B={info['delta_uauc_vs_P2B']:+.9f}",
              flush=True)
    except BaseException as e:
        info.update({"status":"failed","error":repr(e)})
        write()
        print(f"[R2-{args.round}-STOP] {e}",flush=True)
        raise


if __name__=="__main__":
    main()
