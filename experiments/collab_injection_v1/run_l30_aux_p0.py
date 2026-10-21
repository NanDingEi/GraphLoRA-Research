#!/usr/bin/env python3
"""P0 serial controlled L30 auxiliary BCE experiment on GraphLoRA S2.

Two independently S2-warm-started arms, GPU0 foreground:
 CONTROL (aux BCE weight=0) vs AUX002 (weight=0.02).
Use the SAME initial optimizer, LR schedule, seed, training set, and 1-epoch
stop. Final metrics always use native L31 Yes prediction, NOT voting.

No Test evaluation, no auto GPU background jobs, no trained model overwrite.
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
CHECKPOINT=Path(
    "/home/bbx/user_lx/GraphLoRA/logs/S2_STRICT_20261017_143611/"
    "L31_output/20261017143/checkpoint_best.pth")
S2_AUC=0.7296752596287361
S2_UAUC=0.6841670211619721
# The original experiment is already on the server (one full CONTROL epoch
# finished). Only this exact code lineage is eligible for automatic recovery.
P0_PRE_GUARD_FIX_COMMIT="1a41ce8a2fa8fd18bf7a01e410fd62bec7858fa0"
OLD_GUARD='collab_mode != "legacy" or int(gnn_scale) != 0.1 or'
NEW_GUARD='collab_mode != "legacy" or abs(float(gnn_scale) - 0.1) > 1e-10 or'
ARMS=(("P0_CONTROL",0.0),("P0_L30_AUX002",0.02))

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def repo_head():
    return subprocess.check_output(
        ["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()

def subprocess_log(cmd, path):
    path.parent.mkdir(parents=True,exist_ok=True)
    print("[P0-CMD]", " ".join(map(str,cmd)),flush=True)
    with path.open("w",encoding="utf-8") as fh:
        with subprocess.Popen([str(x) for x in cmd],cwd=ROOT,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                bufsize=1,text=True,errors="replace") as p:
            try:
                for line in p.stdout:
                    fh.write(line)
                    fh.flush()
                    if (line.startswith(("[L30-AUX]","[L30-ANCHOR]","[P0-"))
                            or "Averaged stats:" in line
                            or "Traceback" in line
                            or "Error:" in line):
                        print(line[:450].rstrip(),flush=True)
                rc=p.wait()
            except KeyboardInterrupt:
                p.terminate()
                p.wait()
                raise
    if rc:
        raise RuntimeError(f"Child exited {rc}; see {path}")

def check_smoke_lora_update(champion, trained, arm):
    """Verify smoke actually modified the existing S2 LoRA parameters.

    Merely seeing a loss and saved file would not guarantee that AMP did
    not skip all optimizer steps. Compare exact existing LoRA tensor keys.
    """
    import torch
    def ckpt(path):
        try:
            return torch.load(str(path), map_location="cpu", mmap=True,
                              weights_only=False)["model"]
        except TypeError:
            return torch.load(str(path), map_location="cpu")["model"]
    before=ckpt(champion)
    after=ckpt(trained)
    keys=[k for k in after if ("lora_A." in k or "lora_B." in k)
          and k in before]
    if not keys:
        raise RuntimeError(f"{arm} smoke has no comparable S2 LoRA tensors")
    changed=sum(int(not torch.equal(before[k],after[k])) for k in keys)
    if changed==0:
        raise RuntimeError(f"{arm} smoke trained zero LoRA tensors: AMP/optimizer problem")
    print(f"[P0-UPDATE] {arm} smoke changed LoRA tensors={changed}/{len(keys)}",
          flush=True)


def get_val(log):
    pattern=re.compile(
        r"\*\*\*auc:\s*([0-9.eE+-]+).*?\*\*\*uauc:\(\s*([0-9.eE+-]+)")
    found=[]
    for line in log.read_text(encoding="utf-8",errors="replace").splitlines():
        if "Averaged stats:" in line:
            match=pattern.search(line)
            if match:
                found.append(tuple(float(v) for v in match.groups()))
    if len(found)!=1 or not all(math.isfinite(x) for x in found[0]):
        raise RuntimeError(f"Expected one final Valid AUC and UAUC, got {len(found)}: {log}")
    return found[0]

def preflight():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
        raise RuntimeError("Explicitly export CUDA_VISIBLE_DEVICES=0")
    p=subprocess.run(["pgrep","-af","[t]rain_collm_mf_din.py"],
                     capture_output=True,text=True)
    if p.returncode==0 and p.stdout.strip():
        raise RuntimeError("Another training process is active:\n"+p.stdout)
    p=subprocess.run(["git","diff","HEAD","--quiet","--",
        "minigpt4/models/minigpt4rec_v2.py",
        "experiments/collab_injection_v1/run_l30_aux_p0.py",
        "experiments/collab_injection_v1/test_l30_aux_cpu.py"],
        cwd=ROOT)
    if p.returncode:
        raise RuntimeError("Uncommitted P0 code; commit or stash changes")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Historical S2 checkpoint missing: {CHECKPOINT}")

def migrate_verified_guard_fix(state, logs):
    """Keep a completed CONTROL across ONLY the exact integer-cast guard fix.

    The CONTROL's model forward was identical: L30 auxiliary weight=0 means
    the erroneous condition was never evaluated. The original AUX arm was
    stopped at model construction and never trained. Require these facts and
    byte-for-byte model source equivalence apart from the ONE guard replacement.
    """
    if state.get("git_commit")!=P0_PRE_GUARD_FIX_COMMIT:
        raise RuntimeError("P0 resume source commit mismatch; unsafe migration")
    model_path="minigpt4/models/minigpt4rec_v2.py"
    before=subprocess.check_output([
        "git","show",f"{P0_PRE_GUARD_FIX_COMMIT}:{model_path}"
    ],cwd=ROOT,text=True)
    after=(ROOT/model_path).read_text(encoding="utf-8")
    if before.count(OLD_GUARD)!=1 or before.replace(
        OLD_GUARD,NEW_GUARD,1)!=after:
        raise RuntimeError("More than the one audited P0 guard changed; refuse resume")
    arms=state.get("arms",{})
    if set(arms)!={"P0_CONTROL","P0_L30_AUX002"} or (
        arms["P0_CONTROL"].get("status")!="complete" or
        arms["P0_L30_AUX002"].get("status")!="failed"
    ):
        raise RuntimeError("Expected exactly completed CONTROL + failed AUX")
    original_control=logs/"P0_CONTROL_FULL.yaml"
    control_ckpt=Path(arms["P0_CONTROL"].get("best_checkpoint",""))
    if not (original_control.is_file() and control_ckpt.is_file() and
            sha(original_control)==arms["P0_CONTROL"].get("config_sha")):
        raise RuntimeError("Completed CONTROL cannot be verified")
    control_cfg=yaml.safe_load(original_control.read_text(encoding="utf-8"))
    if (float(control_cfg["model"].get("l30_aux_weight",0))!=0 or
        Path(control_cfg["model"]["ckpt"]).resolve()!=CHECKPOINT.resolve() or
        control_cfg["model"]["collab_mode"]!="legacy"):
        raise RuntimeError("Previous CONTROL was not exact unmodified S2")
    failed_log=logs/"P0_AUX_ANCHOR.log"
    if not failed_log.is_file() or (
        "ValueError: L30 auxiliary BCE requires legacy S2 L31 LoRA injection"
        not in failed_log.read_text(encoding="utf-8",errors="replace")
    ):
        raise RuntimeError("Previous AUX error does not match repaired integer-cast guard")
    state["guard_fix_migration"]={
        "old_commit":P0_PRE_GUARD_FIX_COMMIT,
        "new_commit":repo_head(),
        "model_change":"single int(gnn_scale) to float(gnn_scale) comparison",
        "control_reused":True,
        "aux_retrained_from_original_S2":True,
    }
    state["git_commit"]=repo_head()
    print("[P0-GUARD-FIX] verified one-line model change; "
          "reusing completed CONTROL, restarting only AUX from S2",
          flush=True)
    write_state(logs,state)


def write_state(logdir,state):
    tmp=logdir/"state.tmp"
    tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
    tmp.replace(logdir/"state.json")

def create_config(source, name, weight, logdir, outdir, args):
    cfg=copy.deepcopy(source)
    m,r=cfg["model"],cfg["run"]
    m["ckpt"]=str(CHECKPOINT.resolve())
    m["collab_mode"]="legacy"
    m["gnn_scale"]=0.1
    m["gcn_layer"]=31
    m["l30_aux_weight"]=float(weight)
    m["l30_aux_layer"]=30
    m.pop("trainable_policy",None)
    # Preserve S2 original GNN/projector/LoRA trainability and teacher config.
    r.update({
        "seed":args.seed,
        "max_epoch":3,
        "diagnostic_stop_after_epoch0":True,
        "stop_after_n_epochs":1,
        "evaluate":False,
        "skip_validation":False,
        "resume_ckpt_path":None,
        "save_checkpoint_every_n_epochs":0,
        "accum_grad_iters":int(args.accum),
        "init_lr":float(args.lr),
        "min_lr":float(args.lr)*0.4,
        "warmup_lr":min(1e-6,float(args.lr)*0.05),
        "amp_init_scale":1024.0,
        "amp_growth_interval":2000,
        "mid_epoch_eval_iters":"",
        "output_dir":str((outdir/name/"FULL_E1").resolve()),
    })
    for stale in ("max_train_steps","iters_per_epoch","max_iters"):
        r.pop(stale,None)
    file=logdir/f"{name}_FULL.yaml"
    file.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
    smoke=copy.deepcopy(cfg)
    smoke["run"]["max_train_steps"]=args.smoke_steps
    smoke["run"]["skip_validation"]=True
    smoke["run"]["output_dir"]=str((outdir/name/"SMOKE").resolve())
    sm=logdir/f"{name}_SMOKE.yaml"
    sm.write_text(yaml.safe_dump(smoke,sort_keys=False),encoding="utf-8")
    return file,sm

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--lr",type=float,default=2e-5)
    p.add_argument("--accum",type=int,default=3)
    p.add_argument("--smoke-steps",type=int,default=24)
    p.add_argument("--resume-dir",default=None,
                   help="Resume only with SAME committed code/config and S2 checkpoint")
    args=p.parse_args()
    if (args.seed<0 or args.accum<1 or args.smoke_steps<1 or
        not math.isfinite(args.lr) or not 0<args.lr<=1e-4):
        raise ValueError("Invalid P0 hyperparameters")
    preflight()
    base=ROOT/"logs/collab_v1"
    base.mkdir(parents=True,exist_ok=True)
    subprocess_log([sys.executable,"-m","py_compile",
        "minigpt4/models/minigpt4rec_v2.py",
        "experiments/collab_injection_v1/run_l30_aux_p0.py"],
        base/"P0_SYNTAX.log")
    subprocess_log([sys.executable,
        "experiments/collab_injection_v1/test_l30_aux_cpu.py"],
        base/"P0_CPU.log")
    subprocess_log([sys.executable,
        "experiments/collab_injection_v1/prepare_e3_configs.py"],
        base/"P0_PREPARE.log")
    src=ROOT/"tran_configs/collab_v1/e3/S2_L31_CONTROL.yaml"
    source=yaml.safe_load(src.read_text(encoding="utf-8"))
    if source["model"]["collab_mode"]!="legacy" or (
        Path(source["model"]["ckpt"]).resolve()!=CHECKPOINT.resolve()
    ):
        raise RuntimeError("Generated control config is not historical S2")
    if args.resume_dir:
        logs=Path(args.resume_dir).expanduser().resolve()
        path=logs/"state.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        state=json.loads(path.read_text(encoding="utf-8"))
        if state["champion_sha256"]!=sha(CHECKPOINT):
            raise RuntimeError("Unsafe P0 resume: S2 checkpoint changed")
        if (state["seed"]!=args.seed or state["lr"]!=args.lr or
            state["accum"]!=args.accum or state["smoke_steps"]!=args.smoke_steps):
            raise RuntimeError("Unsafe P0 resume: hyperparameters changed")
        if state["git_commit"]!=repo_head():
            migrate_verified_guard_fix(state, logs)
        out=Path(state["output_dir"])
    else:
        tag="S2_L30_AUX_P0_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logs=base/tag
        out=ROOT/"output"/tag
        if logs.exists() or out.exists():
            raise RuntimeError("P0 output directory collision")
        logs.mkdir(parents=True)
        out.mkdir(parents=True)
        state={
            "git_commit":repo_head(),
            "champion":str(CHECKPOINT),
            "champion_sha256":sha(CHECKPOINT),
            "seed":args.seed,"lr":args.lr,
            "accum":args.accum,"smoke_steps":args.smoke_steps,
            "output_dir":str(out.resolve()),
            "arms":{},
        }
        write_state(logs,state)
    print(f"[P0-PLAN] GPU0 foreground control/aux weight=0/.02 "
          f"seed={args.seed} lr={args.lr} optimizer=fresh; 1 epoch each",
          flush=True)
    print(f"[P0-ROOT] logs={logs} outputs={out}",flush=True)
    for name,weight in ARMS:
        old=state["arms"].get(name)
        if old and old["status"]=="complete":
            cfg=logs/f"{name}_FULL.yaml"
            ck=Path(old["best_checkpoint"])
            if not (ck.is_file() and cfg.is_file() and sha(cfg)==old["config_sha"]):
                raise RuntimeError(f"Broken completed arm: {name}")
            print(f"[P0-SKIP] {name} verified complete",flush=True)
            continue
        # Failed/incomplete arm is restarted from its ORIGINAL S2 checkpoint.
        # Never reuse a partially trained model or stale checkpoint_best.
        if old is not None and (out/name).exists():
            import shutil
            shutil.rmtree(out/name)
            print(f"[P0-CLEAN] removed incomplete outputs for {name}",flush=True)
        full,smoke=create_config(source,name,weight,logs,out,args)
        cfg_hash=sha(full)
        state["arms"][name]={"status":"running","config_sha":cfg_hash}
        write_state(logs,state)
        print(f"[P0-START] {name} aux_weight={weight}",flush=True)
        try:
            # Audit evaluation parity for the enabled auxiliary hook.
            if weight:
                subprocess_log([sys.executable,"-u",
                    "experiments/collab_injection_v1/eval_l30_aux_anchor.py",
                    "--config",full,"--checkpoint",CHECKPOINT,
                    "--result",logs/"P0_AUX_ANCHOR.json"],
                    logs/"P0_AUX_ANCHOR.log")
            subprocess_log([sys.executable,"-u","train_collm_mf_din.py",
                "--cfg-path",smoke],logs/f"{name}_SMOKE.log")
            smoke_text=(logs/f"{name}_SMOKE.log").read_text(
                encoding="utf-8",errors="replace")
            if bool(weight) != ("[L30-AUX] step=" in smoke_text):
                raise RuntimeError("P0 smoke aux behavior differs from config")
            smoke_ckpts=list((out/name/"SMOKE").rglob("checkpoint_0.pth"))
            if len(smoke_ckpts)!=1:
                raise RuntimeError(f"{name} expected one smoke checkpoint: {smoke_ckpts}")
            check_smoke_lora_update(CHECKPOINT,smoke_ckpts[0],name)
            subprocess_log([sys.executable,"-u","train_collm_mf_din.py",
                "--cfg-path",full],logs/f"{name}_FULL.log")
            matches=list((out/name/"FULL_E1").rglob("checkpoint_best.pth"))
            if len(matches)!=1:
                raise RuntimeError(f"{name} expected one best checkpoint: {matches}")
            auc,uauc=get_val(logs/f"{name}_FULL.log")
            state["arms"][name]={
                "status":"complete","config_sha":cfg_hash,
                "aux_weight":weight,"auc":auc,"uauc":uauc,
                "delta_auc_vs_s2":auc-S2_AUC,
                "delta_uauc_vs_s2":uauc-S2_UAUC,
                "best_checkpoint":str(matches[0]),
            }
            write_state(logs,state)
            print(f"[P0-RESULT] {name} AUC={auc:.9f} UAUC={uauc:.9f} "
                  f"dAUC_S2={auc-S2_AUC:+.9f} dUAUC_S2={uauc-S2_UAUC:+.9f}",
                  flush=True)
        except BaseException as ex:
            state["arms"][name]={"status":"failed",
                                 "config_sha":cfg_hash,"error":str(ex)}
            write_state(logs,state)
            print(f"[P0-STOP] {name}: {ex}; no later training",flush=True)
            print(f"[P0-RESUME] python -u "
                  f"experiments/collab_injection_v1/run_l30_aux_p0.py "
                  f"--resume-dir {logs} --seed {args.seed} --lr {args.lr} "
                  f"--accum {args.accum} --smoke-steps {args.smoke_steps}",flush=True)
            raise
    ctrl=state["arms"]["P0_CONTROL"]
    aux=state["arms"]["P0_L30_AUX002"]
    delta_a=aux["auc"]-ctrl["auc"]
    delta_u=aux["uauc"]-ctrl["uauc"]
    report={
        "source_checkpoint":str(CHECKPOINT),
        "seed":args.seed,"lr":args.lr,"epochs":1,
        "control":ctrl,"aux002":aux,
        "delta_aux_vs_control_auc":delta_a,
        "delta_aux_vs_control_uauc":delta_u,
        "note":"single seed; requires replication/independent holdout",
    }
    (logs/"comparison.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"[P0-COMPLETE] AUX vs CONTROL: dAUC={delta_a:+.9f} "
          f"dUAUC={delta_u:+.9f}",flush=True)
    print(f"[P0-ROOT] logs={logs} outputs={out}",flush=True)

if __name__=="__main__":
    main()
