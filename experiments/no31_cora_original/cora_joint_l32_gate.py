#!/usr/bin/env python3
"""Single-model end-to-end CoRA RR1+RR5 unified continuation experiments.

J0: RR1 continuation: original L32 BCE plus original C6 CTR/KD.
J1: same, plus 0.25 * BCE(user-gated L31/L32 residual, label).
J2: J1 plus 0.02 * same-user positive/negative pairwise rank loss.

All three restart from the SAME RR1 epoch37 compact checkpoint, SAME deterministic
75% random / 25% same-user-pair sampler, same LR and seed42. RR5 checkpoint
is NEVER loaded; only its existing user-gate architecture is reused.
One Vicuna-7B on physical GPU0, D4_TOPK_RES025, C6_KD_BOTH.
Every epoch: full VALID AUC/UAUC on both L32 and fused score; no TEST.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent.parent
PROTOCOL="CORA_JOINT_L32_GATE_20261113"
SOURCE_CONFIG=ROOT/"local_configs/CORA_RESIDUAL_ROUTING6_20261103/RR1_L32_BASE.yaml"
RR1_CHECKPOINT=ROOT/"output/CORA_RESIDUAL_ROUTING6_20261103/RR1_L32_BASE/20261103092/checkpoint_best.pth"
SEED=42
BATCH_SIZE=2
ACCUM_GRAD=4
STEPS_PER_EPOCH=400
LR=3e-5
MF_LR_RATIO=.1
TRIALS={
    "J0_RR1_CONTINUE":dict(fused_bce=0.0,rank=0.0,gate=False),
    "J1_DUAL_BCE":dict(fused_bce=.25,rank=0.0,gate=True),
    "J2_DUAL_USER_RANK":dict(fused_bce=.25,rank=.02,gate=True),
}


class SameUserPairBatchSampler:
    """Deterministic identical sampler for J0/J1/J2; avoids empty pairwise batches."""
    def __init__(self,frame,steps=STEPS_PER_EPOCH,seed=SEED):
        if not {"UserID","TargetItemID","label"}.issubset(set(frame.columns)):
            raise RuntimeError("TRAIN DataFrame missing UserID/TargetItemID/label")
        user=frame["UserID"].to_numpy(dtype=np.int64)
        label=frame["label"].to_numpy(dtype=np.int64)
        if not np.isin(label,[0,1]).all():raise ValueError("TRAIN labels are not binary")
        groups={}
        for i,(u,y) in enumerate(zip(user,label)):
            groups.setdefault(int(u),[[],[]])[int(y)].append(i)
        self.pairs=[(np.asarray(g[1]),np.asarray(g[0])) for g in groups.values()
                    if g[0] and g[1]]
        if len(self.pairs)<2:raise RuntimeError("Insufficient same-user positive/negative pairs")
        self.size=len(frame)
        self.steps=steps
        self.seed=seed
        self.epoch=0
    def __len__(self):return self.steps
    def __iter__(self):
        rng=np.random.RandomState(self.seed+self.epoch)
        self.epoch+=1
        for i in range(self.steps):
            if i%4==0:
                positive,negative=self.pairs[int(rng.randint(len(self.pairs)))]
                yield [int(positive[rng.randint(len(positive))]),
                       int(negative[rng.randint(len(negative))])]
            else:
                yield [int(v) for v in rng.randint(self.size,size=2)]


def same_user_rank_loss(scores,labels,users):
    import torch.nn.functional as F
    if scores.numel()!=2 or labels.numel()!=2 or users.numel()!=2:
        raise RuntimeError("Paired training requires microbatch=2")
    if int(users[0])!=int(users[1]) or int(labels[0])==int(labels[1]):
        return scores.sum()*0.,False
    if set(labels.long().detach().cpu().tolist())!={0,1}:
        raise RuntimeError("Invalid same-user pair labels")
    pos=(labels>0.5).nonzero(as_tuple=False).reshape(-1)[0]
    neg=(labels<0.5).nonzero(as_tuple=False).reshape(-1)[0]
    return F.softplus(-(scores[pos]-scores[neg])),True


def restore_rr1_strict(model,checkpoint,allow_new_gate):
    """Restore trainable RR1 weights, except the J1/J2 newly initialized gate."""
    import torch
    try:ckpt=torch.load(str(checkpoint),map_location="cpu",weights_only=False)
    except TypeError:ckpt=torch.load(str(checkpoint),map_location="cpu")
    if not isinstance(ckpt,dict) or not isinstance(ckpt.get("model"),dict):
        raise RuntimeError("Expected trainable-parameter-only RR1 checkpoint")
    state=ckpt["model"]
    named=dict(model.named_parameters())
    trainable={name for name,p in named.items() if p.requires_grad}
    gate={name for name in trainable if name.startswith("cora_residual_fusion.")}
    if bool(gate)!=bool(allow_new_gate):
        raise RuntimeError(f"Gate keys mismatch expected_gate={allow_new_gate} actual={sorted(gate)[:6]}")
    missing=trainable-set(state)
    if missing!=gate:
        raise RuntimeError(f"Missing non-gate RR1 trainable checkpoint keys: {sorted(missing-gate)[:15]}; "
                           f"unexpected missing: {sorted(missing)[:15]}")
    unexpected=set(state)-set(model.state_dict())
    if unexpected:
        raise RuntimeError("Unknown checkpoint keys: "+repr(sorted(unexpected)[:15]))
    for key,value in state.items():
        target=model.state_dict()[key]
        if target.shape!=value.shape:raise RuntimeError(f"RR1 tensor shape mismatch {key}")
    incompatible=model.load_state_dict(state,strict=False)
    if set(incompatible.missing_keys)&(trainable-gate) or incompatible.unexpected_keys:
        raise RuntimeError("RR1 checkpoint restore mismatch after load_state_dict")
    if ckpt.get("epoch") is not None and abs(int(ckpt["epoch"])-37)>1:
        raise RuntimeError(f"Expected RR1 epoch37 checkpoint, got {ckpt['epoch']}")
    print(f"[RR1 WEIGHTS RESTORED] epoch={ckpt.get('epoch')} "
          f"loaded={len(state)} newly_initialized_gate_keys={len(gate)} "
          "fresh_optimizer=True teacher_model=False",flush=True)


def install_sampler(trial):
    from torch.utils.data import DataLoader
    from minigpt4.runners.runner_base import RunnerBase
    from minigpt4.datasets.datasets.dataloader_utils import IterLoader,PrefetchLoader
    orig=RunnerBase.train_loader.fget
    def replacement(self):
        if hasattr(self,"_joint_pair_loader"):return self._joint_pair_loader
        orig(self) # initialize self.datasets and VALID/TEST loaders in the official runner
        collection=self.datasets.get("train")
        if not isinstance(collection,(list,tuple)) or len(collection)!=1:
            raise RuntimeError("Expected one single original TRAIN dataset")
        ds=collection[0]
        frame=getattr(ds,"annotation",None)
        if frame is None or len(frame)<1000:raise RuntimeError("Expected full TRAIN DataFrame")
        sampler=SameUserPairBatchSampler(frame)
        dl=DataLoader(ds,batch_sampler=sampler,num_workers=0,pin_memory=True,
                      collate_fn=getattr(ds,"collater",None))
        self._joint_pair_loader=IterLoader(PrefetchLoader(dl),use_distributed=False)
        print(f"[SAME SAMPLER] {trial} train_n={len(ds)} "
              f"pair_users={len(sampler.pairs)} pairing=1/4 "
              f"random=3/4 batch=2 steps={len(sampler)} seed={SEED}",flush=True)
        return self._joint_pair_loader
    RunnerBase.train_loader=property(replacement)


def install_joint_loss(trial,config):
    """Use original C6 loss as-is, add fused BCE (and optionally pair-rank)."""
    import torch
    import torch.nn.functional as F
    import cora_residual_routing6 as r6
    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx

    original_train=MiniGPT4Rec_vx.forward_v2
    original_eval=MiniGPT4Rec_vx.generate_for_samples_v2
    uses_gate=bool(config["gate"])
    captures={"user":[],"label":[],"raw":[],"fused":[]}
    status={"step":0,"pairs":0}

    def train(self,samples,*args,**kwargs):
        if not uses_gate:
            result=original_train(self,samples,*args,**kwargs)
            return result
        result,final,layers,ui,error=r6._capture_live(
            self,samples,original_train,True,*args,**kwargs)
        if not isinstance(result,dict) or "loss" not in result:
            raise RuntimeError("C6 train loss missing; original C6 must remain intact")
        labels=samples["label"].float().to(final.device).view(-1)
        user_ids=samples["UserID"].long().to(final.device).view(-1)
        fused,strength,weights,residuals,gate_prior=self.cora_residual_fusion(
            ui,layers,final)
        raw_bce=F.binary_cross_entropy_with_logits(final.float(),labels)
        fused_bce=F.binary_cross_entropy_with_logits(fused.float(),labels)
        # Original C6 result['loss'] already includes L32 BCE, CTR=.005 and
        # gated MF KD=.01; do not replace or double-count that main loss.
        rank=final.sum()*0.
        pair=False
        if config["rank"]:
            rank,pair=same_user_rank_loss(fused,labels,user_ids)
        total=result["loss"]+config["fused_bce"]*fused_bce
        total=total+config["rank"]*rank+gate_prior
        if not torch.isfinite(total).all():
            raise FloatingPointError("Nonfinite joint CoRA loss")
        result["loss"]=total
        result["logits"]=fused
        status["step"]+=1
        if pair:status["pairs"]+=1
        if status["step"]>100 and config["rank"] and status["pairs"]==0:
            raise RuntimeError("No positive/negative user pairs; rank branch not trained")
        if status["step"]==1 or status["step"]%100==0:
            print(f"[JOINT LOSS] {trial} microstep={status['step']} "
                  f"base_l32_bce={float(raw_bce.detach()):.6f} "
                  f"fused_bce={float(fused_bce.detach()):.6f} "
                  f"user_rank={float(rank.detach()):.6f} "
                  f"rank_pairs={status['pairs']} "
                  f"gate_mean={float(strength.detach().mean()):.6f} "
                  f"gate_std={float(strength.detach().std(unbiased=False)):.6f} "
                  f"prior={float(gate_prior.detach()):.6g} "
                  f"loss={float(total.detach()):.6f} "
                  f"l32_reconstruction_err={error:.6g}",flush=True)
        return result

    def evaluate(self,samples,*args,**kwargs):
        if not uses_gate:
            result=original_eval(self,samples,*args,**kwargs)
            if not isinstance(result,dict) or "logits" not in result:
                raise RuntimeError("Expected valid logits in dict")
            raw=result["logits"].float().detach()
            fused=raw
        else:
            result,final,layers,ui,error=r6._capture_live(
                self,samples,original_eval,True,*args,**kwargs)
            if not isinstance(result,dict) or "logits" not in result:
                raise RuntimeError("Expected valid logits in dict")
            fused=self.cora_residual_fusion(ui,layers,final)[0].float()
            labels=samples["label"].float().to(fused.device)
            result["logits"]=fused
            result["loss"]=F.binary_cross_entropy_with_logits(fused,labels)
            raw=final.float().detach()
        if not torch.isfinite(fused).all():
            raise FloatingPointError("Nonfinite joint fused VALID logits")
        if not self.training:
            for field,tensor in [("user",samples["UserID"]),
                                 ("label",samples["label"]),
                                 ("raw",raw),("fused",fused)]:
                captures[field].append(tensor.detach().float().cpu().numpy().reshape(-1))
        return result

    MiniGPT4Rec_vx.forward_v2=train
    MiniGPT4Rec_vx.generate_for_samples_v2=evaluate
    print(f"[JOINT MODEL] {trial} only_one_vicuna7b=True "
          f"online_dual_BCE={config['fused_bce']} rank={config['rank']} "
          "teacher=False D4+C6 unchanged",flush=True)
    return captures


def install_per_epoch_auc_uauc(trial,captures):
    from sklearn.metrics import roc_auc_score
    from minigpt4.runners.runner_base_rec import RecRunnerBase
    original=RecRunnerBase.eval_epoch

    def score(y,users,pred):
        y=np.asarray(y,dtype=np.int64)
        users=np.asarray(users,dtype=np.int64)
        pred=np.asarray(pred,dtype=np.float64)
        a=float(roc_auc_score(y,pred))
        us=[]
        for u in np.unique(users):
            mask=users==u
            if np.unique(y[mask]).size==2:
                us.append(float(roc_auc_score(y[mask],pred[mask])))
        if not us:raise RuntimeError("No valid users with both classes")
        return a,float(np.mean(us)),len(us)

    def eval_epoch(self,split_name,cur_epoch,skip_reload=False):
        for vals in captures.values():vals.clear()
        result=original(self,split_name,cur_epoch,skip_reload)
        if split_name!="valid" or result is None:return result
        if not captures["label"]:
            raise RuntimeError("VALID produced no logits for AUC/UAUC")
        y=np.concatenate(captures["label"]).astype(np.int64)
        users=np.concatenate(captures["user"]).astype(np.int64)
        raw=np.concatenate(captures["raw"])
        fused=np.concatenate(captures["fused"])
        if not(len(y)==len(users)==len(raw)==len(fused)):
            raise RuntimeError("VALID prediction/count mismatch")
        if np.unique(y).size!=2:raise RuntimeError("VALID contains one class")
        ar,ur,nu=score(y,users,raw)
        af,uf,_=score(y,users,fused)
        official=float(result["agg_metrics"])
        if abs(af-official)>.005:
            raise RuntimeError(f"VALID official AUC mismatch fused={af} reported={official}")
        print(f"[VALID JOINT] {trial} epoch={cur_epoch} "
              f"L32_AUC={ar:.9f} L32_UAUC={ur:.9f} "
              f"fused_AUC={af:.9f} fused_UAUC={uf:.9f} "
              f"uauc_users={nu} n={len(y)}",flush=True)
        result["auc_l32"]=ar
        result["uauc_l32"]=ur
        result["auc_fused"]=af
        result["uauc_fused"]=uf
        root=ROOT/"out"/PROTOCOL/trial
        root.mkdir(parents=True,exist_ok=True)
        path=root/"valid_every_epoch.csv"
        with path.open("a",newline="",encoding="utf-8") as f:
            writer=csv.writer(f)
            if f.tell()==0:writer.writerow(
                ["epoch","l32_auc","l32_uauc","fused_auc","fused_uauc",
                 "uauc_users","n"])
            writer.writerow([cur_epoch,ar,ur,af,uf,nu,len(y)])
        # Save every epoch, including those not selected as best by AUC.
        self._save_checkpoint(cur_epoch,is_best=False)
        return result

    RecRunnerBase.eval_epoch=eval_epoch


def install_rr1_initialization(trial,use_gate):
    from minigpt4.runners.runner_base import RunnerBase
    original=RunnerBase.train
    def train(self):
        if self.resume_ckpt_path is not None:
            raise RuntimeError("RR1 compact model-only init requires FRESH AdamW; no optimizer resume")
        model=self.unwrap_dist_model(self.model)
        restore_rr1_strict(model,RR1_CHECKPOINT,use_gate)
        if self._optimizer is not None:
            raise RuntimeError("Optimizer already initialized before RR1 restore")
        return original(self)
    RunnerBase.train=train


def validate_config(path):
    import yaml
    c=yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    r=c["run"]
    checks={
        "seed42":int(r["seed"])==SEED,
        "batch2":int(r["batch_size_train"])==BATCH_SIZE,
        "accum4":int(r["accum_grad_iters"])==ACCUM_GRAD,
        "steps400":int(r["iters_per_epoch"])==STEPS_PER_EPOCH,
        "LR3e-5":abs(float(r["init_lr"])-LR)<1e-12,
        "valid_only":r.get("valid_splits")==["valid"] and not r.get("test_splits"),
        "MF_trainable":c["model"]["freeze_rec"] is False,
        "no_auto_resume":not r.get("resume_ckpt_path") and not c["model"].get("ckpt"),
    }
    bad=[k for k,v in checks.items() if not v]
    if bad:raise RuntimeError("Invalid joint config: "+repr(bad))
    return c


def prepare():
    import yaml
    if not SOURCE_CONFIG.is_file() or not RR1_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Need RR1 source config/checkpoint: {SOURCE_CONFIG}, {RR1_CHECKPOINT}")
    config_dir=ROOT/"local_configs"/PROTOCOL
    config_dir.mkdir(parents=True,exist_ok=True)
    (ROOT/"log"/PROTOCOL).mkdir(parents=True,exist_ok=True)
    for trial,settings in TRIALS.items():
        path=config_dir/(trial+".yaml")
        if path.exists():
            validate_config(path)
        else:
            cfg=yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))
            cfg["run"].update({
                "init_lr":LR,
                "warmup_lr":min(float(cfg["run"].get("warmup_lr",LR/10)),LR/10),
                "min_lr":min(float(cfg["run"].get("min_lr",1e-6)),1e-6),
                "seed":SEED,"batch_size_train":BATCH,
                "accum_grad_iters":ACCUM_GRAD,"iters_per_epoch":STEPS_PER_EPOCH,
                "max_epoch":1000,"valid_splits":["valid"],"test_splits":[],
                "evaluate":False,"resume_ckpt_path":None,
                "output_dir":str(ROOT/"output"/PROTOCOL/trial),
            })
            cfg["model"]["freeze_rec"]=False
            cfg["model"]["ckpt"]=None
            path.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
        validate_config(path)
        print(f"[JOINT PLAN] trial={trial} settings={settings} cfg={path}",flush=True)
    print(f"[JOINT PLAN READY] seed={SEED} LR={LR} MF LR={LR*MF_LR_RATIO:g} "
          f"batch={BATCH_SIZE} accum={ACCUM_GRAD} steps={STEPS_PER_EPOCH} "
          "RR1 epoch37 start, separate checkpoints, VALID-only, no teachers",flush=True)


def worker(trial,epochs,patience):
    import cora_residual_routing6 as r6
    use_gate=TRIALS[trial]["gate"]
    r6.configure()
    r6.gpu0_preflight("joint-"+trial)
    config=ROOT/"local_configs"/PROTOCOL/(trial+".yaml")
    validate_config(config)
    base_trial=r6.TRIAL_BY_NAME["RR5_USER_GATE_R31" if use_gate else "RR1_L32_BASE"]
    r6.late.base.core.patch_runner(
        screen_epochs=epochs,mf_lr_ratio=MF_LR_RATIO,patience=patience)
    r6.late.base.core.install_compact_atomic_best_checkpoint()
    r6.late.base.combo.install_combo_model(r6.late.base.combo.NAMES["C6_KD_BOTH"])
    r6.late.base._patch_runtime_classes(base_trial)
    if use_gate:
        r6.install_trial_module(base_trial)
    r6.install_optimizer_audit(base_trial)
    install_sampler(trial)
    captures=install_joint_loss(trial,TRIALS[trial])
    install_per_epoch_auc_uauc(trial,captures)
    install_rr1_initialization(trial,use_gate)
    print(f"[JOINT WORKER READY] trial={trial} epochs={epochs} "
          f"seed={SEED} single7B=True valid_only=True "
          f"source_rr1={RR1_CHECKPOINT}",flush=True)
    sys.argv=["train_collm_mf_din.py","--cfg-path",str(config)]
    runpy.run_path(str(PROJECT/"train_collm_mf_din.py"),run_name="__main__")


def cpu_test():
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(SEED)
    rng=np.random.RandomState(SEED)
    frame=pd.DataFrame({"UserID":np.repeat(np.arange(1,13),4),
                        "TargetItemID":np.arange(48)+1,
                        "label":np.tile([0,1,0,1],12)})
    sampler=SameUserPairBatchSampler(frame,steps=32)
    batches=list(iter(sampler))
    assert len(batches)==32
    assert len(set(tuple(x) for x in batches))>1
    for pair in batches[::4]:
        assert frame.iloc[pair[0]].UserID==frame.iloc[pair[1]].UserID
        assert frame.iloc[pair[0]].label!=frame.iloc[pair[1]].label
    s=torch.tensor([.2,-.3],requires_grad=True)
    labels=torch.tensor([1.,0.])
    users=torch.tensor([1,1])
    rank,valid=same_user_rank_loss(s,labels,users)
    assert valid and torch.allclose(rank,F.softplus(torch.tensor(-.5)))
    gate=nn.Sequential(nn.LayerNorm(8),nn.Linear(8,64),nn.GELU(),nn.Linear(64,1))
    ui=torch.randn(2,8)
    z31=torch.randn(2,requires_grad=True)
    z32=torch.randn(2,requires_grad=True)
    fused=z32+torch.sigmoid(gate(ui)).view(-1)*(z31-z32)
    base=F.binary_cross_entropy_with_logits(z32,labels)
    joint=base+.25*F.binary_cross_entropy_with_logits(fused,labels)
    loss=joint+.02*same_user_rank_loss(fused,labels,users)[0]
    loss.backward()
    assert z32.grad is not None and z31.grad is not None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in gate.parameters())
    assert all(torch.isfinite(p.grad).all() for p in (z31,z32))
    assert not TRIALS["J0_RR1_CONTINUE"]["gate"]
    assert all(k["gate"] for name,k in TRIALS.items() if name!="J0_RR1_CONTINUE")
    print("[JOINT CPU PASS] deterministic paired sampling, dual BCE, "
          "single-model gate and ranking gradients, seed42; no GPU inference",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-test",action="store_true")
    parser.add_argument("--plan",action="store_true")
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--only",default=",".join(TRIALS))
    parser.add_argument("--epochs",type=int,default=6)
    parser.add_argument("--patience",type=int,default=6)
    parser.add_argument("--_worker",default="",help=argparse.SUPPRESS)
    args=parser.parse_args()
    os.chdir(PROJECT)
    if args.cpu_test:
        cpu_test()
        return
    if args._worker:
        if args._worker not in TRIALS:raise ValueError("Unknown worker")
        worker(args._worker,args.epochs,args.patience)
        return
    if args.plan==args.run:
        parser.error("Specify exactly one of --plan/--run")
    if args.epochs<=0 or args.patience<=0:
        parser.error("epochs/patience must be positive")
    if args.plan:
        prepare();return
    names=[x.strip() for x in args.only.split(",") if x.strip()]
    if not names or len(names)!=len(set(names)) or any(n not in TRIALS for n in names):
        parser.error("Invalid --only list")
    prepare()
    for trial in names:
        log=ROOT/"log"/PROTOCOL/(trial+".log")
        output=ROOT/"out"/PROTOCOL/trial
        saved=ROOT/"output"/PROTOCOL/trial
        if log.exists() or output.exists() or saved.exists():
            raise RuntimeError(f"Trial exists; refuse overwrite: {trial}; "
                               f"log={log}, out={output}, ckpt={saved}")
    for trial in names:
        import cora_residual_routing6 as r6
        r6.gpu0_preflight("launch-"+trial)
        log=ROOT/"log"/PROTOCOL/(trial+".log")
        cmd=[sys.executable,"-u",str(Path(__file__).resolve()),
             "--_worker",trial,"--epochs",str(args.epochs),
             "--patience",str(args.patience)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="0")
        print(f"[JOINT START] trial={trial} serial GPU0 foreground log={log}",flush=True)
        with log.open("x",encoding="utf-8") as f:
            p=subprocess.Popen(cmd,cwd=PROJECT,env=env,
                              stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,encoding="utf-8",errors="replace",bufsize=1)
            for line in p.stdout:
                print(line,end="",flush=True)
                f.write(line)
            rc=p.wait()
        if rc:
            raise SystemExit(f"[JOINT STOP ON FAIL] trial={trial} exit={rc}; see {log}")
    print("[JOINT ALL COMPLETE] J0/J1/J2 VALID only, no TEST",flush=True)


if __name__=="__main__":
    main()
