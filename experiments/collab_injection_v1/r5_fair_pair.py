#!/usr/bin/env python3
"""Fair R5: original GraphLoRA vs GraphLoRA+CoRA, each TWO full epochs.

Keep pretrained MF (0912 checkpoint) and the original 3-epoch training recipe.
Do NOT load S2/R4 or any jointly-trained GraphLoRA checkpoint. GNN/LoRA/
projector are fresh as in the original baseline, with MATCHING initial hashes.
Only methodological difference between paired arms: CoRA.
"""
import argparse, copy, json, math, os, re, subprocess, sys
from pathlib import Path
import torch, yaml

ROOT=Path("/home/bbx/user_lx/GraphLoRA_collab_v1")
EXP="experiments/collab_injection_v1"
NAME="R5_FAIR_MF_CORA_20261022"
LOG=ROOT/"logs/collab_v1"/NAME
OUT=ROOT/"output"/NAME
MF=ROOT.parent/"GraphLoRA/checkpoints/0912_ml1m_oodv2_best_model_d256lr-0.001wd0.0001.pth"
METRIC=re.compile(r"\*\*\*auc:\s*([0-9.eE+-]+).*?\*\*\*uauc:\s*\(\s*([0-9.eE+-]+)")

def load_torch(path):
    try: return torch.load(str(path),map_location="cpu",weights_only=False)
    except TypeError: return torch.load(str(path),map_location="cpu")

def reference(source):
    if not isinstance(source,dict) or not all(x in source for x in ("model","run","datasets")):
        raise RuntimeError("Original GraphLoRA YAML invalid")
    m,r=source["model"],source["run"]
    path=Path(m.get("rec_config",{}).get("pretrained_path",""))
    if not MF.is_file() or path.resolve()!=MF.resolve():
        raise RuntimeError(f"Must retain original pretrained MF: {MF}; YAML had {path}")
    if m.get("ckpt") not in (None,"") or r.get("resume_ckpt_path") not in (None,""):
        raise RuntimeError("Original GraphLoRA YAML has a jointly-trained model checkpoint")
    if m.get("rec_model")!="MF" or m.get("lora_gnn")!="ngcf":
        raise RuntimeError("Original must have MF + NGCF")
    for key in ("freeze_rec","freeze_lora","freeze_proj"):
        if m.get(key) is not False: raise RuntimeError(f"Original {key} is not False")
    if (m.get("collab_mode","legacy")!="legacy" or int(m.get("gcn_layer",-1))!=31
        or m.get("gnn_semantic_align") is not True):
        raise RuntimeError("Wrong original GraphLoRA L31/semantic settings")
    if (int(r.get("max_epoch",-1))!=3 or r.get("seed")!=42 or
            float(r.get("init_lr",0))!=1e-4 or
            int(r.get("batch_size_train",-1))!=3):
        raise RuntimeError("Unexpected original baseline scheduler/seed/LR/batch")
    return m,r

def make(source,arm,phase):
    cfg=copy.deepcopy(source)
    m,r=cfg["model"],cfg["run"]
    for name in ("r4_round","trainable_policy","r2_mode"):
        m.pop(name,None)
    if arm=="cora": m["r3_mode"]="cora"
    else: m.pop("r3_mode",None)
    m["ckpt"]=None
    # Same FX autocast cache fix for both arms (explicit shared correction).
    m["fix_autocast_cache"]=True
    r["output_dir"]=str(OUT/arm.upper()/phase)
    r["resume_ckpt_path"]=None
    r["evaluate"]=False
    r["distributed"]=False
    r["max_epoch"]=3                      # Original cosine LR horizon
    r["stop_after_n_epochs"]=1 if phase=="SMOKE" else 2
    r["diagnostic_stop_after_epoch0"]=False
    r["save_checkpoint_every_n_epochs"]=1
    r["skip_validation"]=phase=="SMOKE"
    r["valid_splits"]=["valid"]
    r["test_splits"]=[]
    for key in ("max_train_steps","iters_per_epoch","max_iters","mid_epoch_eval_iters"):
        r.pop(key,None)
    if phase=="SMOKE": r["max_train_steps"]=24
    if m["rec_config"]["pretrained_path"]!=source["model"]["rec_config"]["pretrained_path"]:
        raise RuntimeError("MF checkpoint modified")
    return cfg

def process(arm,phase):
    folder=LOG/arm.upper()
    fp=folder/f"{phase}_initial.json"
    logfile=folder/f"{phase}.log"
    cmd=[sys.executable,"-u",f"{EXP}/r5_fair_entry.py",
         "--cfg-path",str(folder/f"{phase}.yaml"),"--arm",arm,
         "--fingerprint",str(fp)]
    with logfile.open("x",encoding="utf-8") as dest:
        sub=subprocess.Popen(cmd,cwd=ROOT,
              env={**os.environ,"CUDA_VISIBLE_DEVICES":"0"},
              stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
              text=True,errors="replace",bufsize=1)
        try:
            for line in sub.stdout:
                dest.write(line)
                if any(tag in line for tag in (
                        "[R5-FAIR","[R3-","Train: data epoch:",
                        "Averaged stats:","Saving checkpoint","Traceback","Error:")):
                    print(f"[{arm}-{phase}] {line.strip()[:500]}",flush=True)
            code=sub.wait()
        except BaseException:
            sub.terminate();sub.wait();raise
    if code: raise RuntimeError(f"{arm}/{phase} failed code={code}: {logfile}")
    if not fp.exists(): raise RuntimeError(f"Missing initial weights hash: {fp}")
    text=logfile.read_text(errors="replace")
    if arm=="cora":
        match=re.search(r"\[R3-COUNTERS\]\s*(\{[^\n]+\})",text)
        if not match:raise RuntimeError("CoRA counters missing")
        counters=json.loads(match.group(1))
        if any(counters.get(k,0)<=0 for k in ("q_cora_forward","v_cora_forward")):
            raise RuntimeError(f"CoRA was inactive: {counters}")
    elif "[R3-MODEL]" in text: raise RuntimeError("Baseline activated CoRA")
    return json.loads(fp.read_text())["initial_shared"],text

def equal_weights(x,y,tag):
    groups=("MF","GNN","Projector","LoRA")
    bad=[key for key in groups if x[key]["sha256"]!=y[key]["sha256"]]
    if bad: raise RuntimeError(f"Unfair initial {tag} mismatch: {bad}")
    print(f"[R5-FAIR-INIT-MATCH] {tag} MF GNN Projector LoRA exact",flush=True)

def verify_weights(arm,phase,n):
    folder=OUT/arm.upper()/phase
    for epoch in range(n):
        matches=list(folder.rglob(f"checkpoint_{epoch}.pth"))
        if len(matches)!=1:raise RuntimeError(f"{arm}/{phase} checkpoint_{epoch} invalid {matches}")
        obj=load_torch(matches[0])
        if obj.get("epoch")!=epoch or not obj.get("optimizer"):
            raise RuntimeError(f"Checkpoint epoch or optimizer invalid: {matches[0]}")
        keys=obj["model"].keys()
        flags={"MF":any(k.startswith("rec_encoder.") for k in keys),
               "GNN":any(k.startswith("gcn.") for k in keys),
               "Projector":any(k.startswith("llama_proj.") for k in keys),
               "LoRA":any(".lora_B." in k for k in keys),
               "CoRA":any(k.startswith("r3_adapter.") for k in keys)}
        if not all(flags[k] for k in ("MF","GNN","Projector","LoRA")) or flags["CoRA"]!=(arm=="cora"):
            raise RuntimeError(f"Missing trainable modules in {matches[0]}: {flags}")
        del obj

def scores(text):
    ans=[]
    for ln in text.splitlines():
        if "Averaged stats:" in ln:
            m=METRIC.search(ln)
            if m:
                a,u=map(float,m.groups())
                if not all(math.isfinite(v) and 0<=v<=1 for v in (a,u)):
                    raise RuntimeError(f"Invalid validation metrics: {(a,u)}")
                ans.append((a,u))
    if len(ans)!=2: raise RuntimeError(f"Expected exactly 2 Valid evaluations, got {ans}")
    return ans

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--baseline-yaml",default=str(ROOT/"tran_configs/collm_pretrain_mf_ood_scratch_rlm.yaml"))
    p.add_argument("--prepare-only",action="store_true")
    args=p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0")
    source_path=Path(args.baseline_yaml).expanduser().resolve()
    if not source_path.is_file(): raise FileNotFoundError(source_path)
    raw=yaml.safe_load(source_path.read_text())
    reference(raw)
    if LOG.exists() or OUT.exists():
        raise FileExistsError(f"Would overwrite existing logs/output: {LOG} {OUT}")
    for arm in ("control","cora"):
        dest=LOG/arm.upper()
        dest.mkdir(parents=True,exist_ok=False)
        for phase in ("SMOKE","FULL_E2"):
            (dest/f"{phase}.yaml").write_text(yaml.safe_dump(make(raw,arm,phase),sort_keys=False))
    mf_hash=None
    manifest={"status":"prepared","reference_yaml":str(source_path),
              "MF_checkpoint":str(MF),"S2_checkpoint":None,"seed":42,
              "epochs":2,"arms":["control","cora"],
              "GNN_initialization":"original random NGCF, shared hashes verified",
              "fix_autocast_cache":True}
    status=LOG/"manifest.json"
    def save():status.write_text(json.dumps(manifest,indent=2))
    save()
    print(f"[R5-FAIR-START] MF={MF} S2=NONE control_vs_cora 2epochs",flush=True)
    if args.prepare_only:return
    try:
        manifest["status"]="smoke";save()
        sh={}
        for arm in ("control","cora"):
            sh[arm],_=process(arm,"SMOKE")
            verify_weights(arm,"SMOKE",1)
        equal_weights(sh["control"],sh["cora"],"SMOKE")
        manifest["status"]="full";save()
        fh={};res={}
        for arm in ("control","cora"):
            fh[arm],text=process(arm,"FULL_E2")
            res[arm]=scores(text)
            verify_weights(arm,"FULL_E2",2)
            equal_weights(sh[arm],fh[arm],f"{arm} smoke-vs-full")
        equal_weights(fh["control"],fh["cora"],"FULL")
        csv=["epoch,control_auc,cora_auc,delta_auc,control_uauc,cora_uauc,delta_uauc"]
        for ep in range(2):
            x,y=res["control"][ep],res["cora"][ep]
            csv.append(f"{ep},{x[0]:.9f},{y[0]:.9f},{y[0]-x[0]:+.9f},"
                       f"{x[1]:.9f},{y[1]:.9f},{y[1]-x[1]:+.9f}")
        (LOG/"fair_comparison.csv").write_text("\n".join(csv)+"\n")
        manifest["status"]="complete"
        manifest["results"]={a:[{"epoch":i,"auc":z[0],"uauc":z[1]}
                         for i,z in enumerate(res[a])] for a in res}
        save()
        print("[R5-FAIR-COMPLETE]\n"+"\n".join(csv),flush=True)
    except BaseException as e:
        manifest["status"]="failed";manifest["error"]=repr(e);save();raise

if __name__=="__main__":main()
