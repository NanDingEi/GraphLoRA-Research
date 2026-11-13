#!/usr/bin/env python3
"""RR1 compact-checkpoint Student continuation: T0/T1/T2/T3.

One GPU0 Vicuna-7B + CoRA student at a time. Original D4 multi-TopK inputs,
C6 CTR=.005, reliability-gated MF KD=.01, seed42 are retained.
No Test during training; every VALID epoch records both AUC and UAUC.
T2/T3 use offline RR1/RR5 TRAIN-only teacher logits, never two live 7B teachers.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent.parent
PROTOCOL="CORA_STUDENT_JOINT_20261113"
RR1=ROOT/"output/CORA_RESIDUAL_ROUTING6_20261103/RR1_L32_BASE/20261103092/checkpoint_best.pth"
SOURCE=ROOT/"local_configs/CORA_RESIDUAL_ROUTING6_20261103/RR1_L32_BASE.yaml"
TEACHER_DIR=ROOT/"out"/PROTOCOL
SEED=42
EPOCH_STEPS=400
BATCH=2
ACCUM=4
LR=3e-5
MF_LR_RATIO=0.1
TEMP=2.0

TRIALS={
    "T0_CONTINUE": {"rank":0.,   "ens":0.,   "mf_anchor":0.},
    "T1_USER_RANK": {"rank":.02,  "ens":0.,   "mf_anchor":0.},
    "T2_ENSEMBLE_KD":{"rank":0., "ens":.05,  "mf_anchor":0.},
    "T3_JOINT":     {"rank":.02,  "ens":.05,  "mf_anchor":.0005},
}


class IndexedDataset:
    """Preserve original sample/prompt; attach exact TRAIN row identity."""
    def __init__(self, dataset):
        self.base=dataset
        self.annotation=dataset.annotation
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        record=dict(self.base[int(i)])
        record["sample_index"]=np.int64(i)
        return record
    def collater(self, values):
        return self.base.collater(values)


class MixedPairBatchSampler:
    """Same sampling for all four trials; every fourth microbatch is a user pair."""
    def __init__(self,annotation,steps=EPOCH_STEPS,seed=SEED):
        if len(annotation)<8:
            raise ValueError("Train dataset too small for paired sampling")
        required={"UserID","label","TargetItemID"}
        if not required.issubset(annotation.columns):
            raise ValueError("Expected TRAIN annotation UserID/TargetItemID/label")
        y=annotation["label"].to_numpy(dtype=np.int64)
        u=annotation["UserID"].to_numpy(dtype=np.int64)
        groups={}
        for ix, (uid, label) in enumerate(zip(u,y)):
            if label not in (0,1):
                raise ValueError("Pair sampler requires binary labels")
            groups.setdefault(int(uid),[[],[]])[int(label)].append(ix)
        self.pairs=[(np.asarray(parts[1]),np.asarray(parts[0]))
                    for parts in groups.values() if parts[0] and parts[1]]
        if len(self.pairs)<2:
            raise RuntimeError("TRAIN has insufficient users with both positive and negative labels")
        self.n=len(y)
        self.steps=int(steps)
        self.seed=int(seed)
        self.epoch=0
    def __len__(self):
        return self.steps
    def __iter__(self):
        rng=np.random.RandomState(self.seed+self.epoch)
        self.epoch+=1
        for step in range(self.steps):
            if step % 4 == 0:
                positive,negative=self.pairs[int(rng.randint(len(self.pairs)))]
                yield [int(positive[rng.randint(len(positive))]),
                       int(negative[rng.randint(len(negative))])]
            else:
                x=rng.randint(self.n,size=2)
                yield [int(x[0]),int(x[1])]


def _annotation(dataset):
    frame=getattr(dataset,"annotation",None)
    if frame is None or not all(x in frame.columns for x in ("UserID","TargetItemID","label")):
        raise RuntimeError("Expected original Movielens TRAIN DataFrame; teacher identity audit unavailable")
    return frame


class TeacherTargets:
    def __init__(self,frame,root=TEACHER_DIR):
        sources={}
        columns={"labels":"label","users":"UserID","items":"TargetItemID"}
        n=len(frame)
        for name in ("RR1","RR5"):
            path=Path(root)/f"teacher_{name}_train.npz"
            if not path.is_file():
                raise FileNotFoundError("Missing TRAIN teacher scores: "+str(path))
            with np.load(path,allow_pickle=False) as f:
                expected={"sample_index","labels","users","items","scores"}
                if not expected.issubset(set(f.files)):
                    raise RuntimeError(f"{path} missing teacher NPZ keys")
                if not np.array_equal(f["sample_index"],np.arange(n,dtype=np.int64)):
                    raise RuntimeError(f"{name} teacher sample_index/order mismatch")
                for saved,original in columns.items():
                    if not np.array_equal(f[saved],frame[original].to_numpy(dtype=np.int64)):
                        raise RuntimeError(f"{name} teacher {saved} identity mismatch")
                z=np.asarray(f["scores"],dtype=np.float64).reshape(-1)
            if z.shape!=(n,) or not np.isfinite(z).all():
                raise RuntimeError(f"Invalid {name} TRAIN teacher scores")
            mu=float(z.mean()); sd=float(z.std())
            if sd<1e-8: raise RuntimeError(f"Teacher {name} scores have near-zero variance")
            sources[name]=np.clip((z-mu)/sd,-15,15)
            print(f"[TRAIN TEACHER AUDIT] {name}: n={n} mean={mu:.7g} std={sd:.7g}",flush=True)
        combined=.5*sources["RR1"]+.5*sources["RR5"]
        import torch
        self.targets=torch.from_numpy(combined.astype(np.float32))
        self.length=n
    def lookup(self,indices,device):
        ids=indices.detach().cpu().long().view(-1)
        if ids.numel()==0 or ids.min()<0 or ids.max()>=self.length:
            raise RuntimeError("Student TRAIN sample_index out of teacher bounds")
        return self.targets[ids].to(device=device,non_blocking=True)


def _compact_load(model,path):
    import torch
    try:
        ck=torch.load(str(path),map_location="cpu",weights_only=False)
    except TypeError:
        ck=torch.load(str(path),map_location="cpu")
    if not isinstance(ck,dict) or not isinstance(ck.get("model"),dict):
        raise RuntimeError("RR1 is not a compact trainable-only checkpoint")
    trained={name for name,p in model.named_parameters() if p.requires_grad}
    saved=ck["model"]
    available=set(model.state_dict())
    if trained-set(saved) or set(saved)-available:
        raise RuntimeError(
            f"RR1 restore key mismatch: trainable_missing={sorted(trained-set(saved))[:12]}, "
            f"unexpected={sorted(set(saved)-available)[:12]}")
    incompat=model.load_state_dict(saved,strict=False)
    if set(incompat.missing_keys)&trained or incompat.unexpected_keys:
        raise RuntimeError("RR1 compact model.load_state_dict mismatch")
    print(f"[STUDENT RESTORE PASS] RR1 epoch={ck.get('epoch')} "
          f"trained_tensor_keys={len(trained)} optimizer=FRESH "
          f"source={path}",flush=True)
    return ck


def _pairwise_logistic(scores,labels,users):
    import torch
    import torch.nn.functional as F
    if scores.numel()!=2 or labels.numel()!=2:
        raise RuntimeError("Paired training requires microbatch=2")
    if int(users[0]) != int(users[1]) or int(labels[0])==int(labels[1]):
        return scores.sum()*0.,False
    pos=torch.where(labels>0.5)[0]
    neg=torch.where(labels<0.5)[0]
    if pos.numel()!=1 or neg.numel()!=1:
        raise RuntimeError("Pair sampler produced invalid labels")
    return F.softplus(-(scores[pos[0]]-scores[neg[0]])),True


def _install_sampler_and_losses(trial_name,spec):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from minigpt4.runners.runner_base import RunnerBase
    from minigpt4.datasets.datasets.dataloader_utils import PrefetchLoader,IterLoader
    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx

    state={"teacher":None,"pair_steps":0,"seen":0}
    original_train_loader=RunnerBase.train_loader.fget

    def paired_train_loader(runner):
        existing=getattr(runner,"_student_pair_loader",None)
        if existing is not None: return existing
        # Trigger official dataset reorganization once, preserving VALID loaders.
        original_train_loader(runner)
        train_set=runner.datasets.get("train")
        if isinstance(train_set,(tuple,list)):
            if len(train_set)!=1:
                raise RuntimeError("Expected one TRAIN dataset, not mixed datasets")
            train_set=train_set[0]
        frame=_annotation(train_set)
        if len(frame)<1000:
            raise RuntimeError("Expected full TRAIN dataset; abort tiny subset")
        if spec["ens"]:
            state["teacher"]=TeacherTargets(frame)
        mixed=MixedPairBatchSampler(frame,steps=EPOCH_STEPS,seed=SEED)
        wrapped=IndexedDataset(train_set)
        loader=DataLoader(wrapped,batch_sampler=mixed,
                          num_workers=0,pin_memory=True,collate_fn=wrapped.collater)
        runner._student_pair_loader=IterLoader(
            PrefetchLoader(loader),use_distributed=False)
        print(f"[MIXED PAIR SAMPLER] trial={trial_name} n={len(wrapped)} "
              f"eligible_users={len(mixed.pairs)} every_fourth_batch=pos_neg "
              f"micro_steps={EPOCH_STEPS} seed={SEED}",flush=True)
        return runner._student_pair_loader

    RunnerBase.train_loader=property(paired_train_loader)

    original=MiniGPT4Rec_vx.forward_v2
    yes_id=None

    def student_forward(self,samples,*args,**kwargs):
        nonlocal yes_id
        if yes_id is None:
            yes_id=int(self.llama_tokenizer(
                self.pos_ans[0],add_special_tokens=False).input_ids[0])
        answer=[self.pos_ans[0] if int(y)==1 else self.neg_ans[0]
                for y in samples["label"]]
        tokens=self.llama_tokenizer(answer,return_tensors="pt",
                                   padding="longest",truncation=True,
                                   max_length=self.max_txt_len,
                                   add_special_tokens=False)
        offset=int(tokens.input_ids.shape[-1])+1
        caught={}
        def grab(_module,_inputs,output):
            caught["logits"]=output.logits[:,-offset,yes_id].float()
        handle=self.llama_model_lora.register_forward_hook(grab)
        try:
            result=original(self,samples,*args,**kwargs)
        finally:
            handle.remove()
        if not isinstance(result,dict) or "loss" not in result or "logits" not in caught:
            raise RuntimeError("C6 student main loss/logit capture failed")
        pred=caught["logits"]
        labels=samples["label"].float().to(pred.device).view(-1)
        userids=samples["UserID"].long().view(-1)
        if pred.ndim!=1 or pred.shape!=labels.shape:
            raise RuntimeError("Student predicted logits shape mismatch")
        extra=pred.sum()*0.
        rank_loss=pred.sum()*0.
        kd_loss=pred.sum()*0.
        anchor_loss=pred.sum()*0.
        pair=False
        if spec["rank"]:
            rank_loss,pair=_pairwise_logistic(pred,labels,userids)
            extra=extra+float(spec["rank"])*rank_loss
        elif userids.numel()==2 and userids[0]==userids[1] and labels[0]!=labels[1]:
            pair=True
        if spec["ens"]:
            teacher=state["teacher"]
            if teacher is None:
                raise RuntimeError("TRAIN teacher cache was not loaded")
            if "sample_index" not in samples:
                raise RuntimeError("Missing exact TRAIN sample index for KD")
            soft_logits=teacher.lookup(samples["sample_index"],pred.device)
            kd_loss=F.binary_cross_entropy_with_logits(
                pred/TEMP, torch.sigmoid(soft_logits/TEMP))*TEMP**2
            extra=extra+float(spec["ens"])*kd_loss
        if spec["mf_anchor"]:
            if not hasattr(self,"_student_anchor_user"):
                raise RuntimeError("MF anchor snapshot missing: must restore RR1 before training")
            items=samples["TargetItemID"].long().view(-1)
            user_table=self.rec_encoder.user_embedding.weight
            item_table=self.rec_encoder.item_embedding.weight
            anchor_loss=(
                (user_table[userids].float()-self._student_anchor_user[userids]).square().mean()+
                (item_table[items].float()-self._student_anchor_item[items]).square().mean())
            extra=extra+float(spec["mf_anchor"])*anchor_loss
        loss=result["loss"]+extra
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError("Nonfinite student joint loss")
        result["loss"]=loss
        result["logits"]=pred
        state["seen"]+=1
        if pair: state["pair_steps"]+=1
        if state["seen"]>100 and state["pair_steps"]==0:
            raise RuntimeError("No positive/negative user pair seen after 100 microsteps")
        if state["seen"]==1 or state["seen"]%100==0:
            print(f"[STUDENT LOSS] {trial_name} step={state['seen']} "
                  f"base_c6={float((loss-extra).detach()):.6f} "
                  f"rank={float(rank_loss.detach()):.6f} "
                  f"ens_kd={float(kd_loss.detach()):.6f} "
                  f"mf_anchor={float(anchor_loss.detach()):.6g} "
                  f"pair_steps={state['pair_steps']} "
                  f"total={float(loss.detach()):.6f}",flush=True)
        return result

    MiniGPT4Rec_vx.forward_v2=student_forward
    print(f"[STUDENT LOSSES INSTALLED] {trial_name} rank={spec['rank']} "
          f"ens={spec['ens']} mf_anchor={spec['mf_anchor']} "
          "C6 CTR=.005 gated_MF_KD=.01 unchanged",flush=True)


def _install_valid_metrics(trial_name):
    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx
    from minigpt4.runners.runner_base_rec import RecRunnerBase
    original_generate=MiniGPT4Rec_vx.generate_for_samples_v2
    original_eval=RecRunnerBase.eval_epoch
    captured={"users":[],"labels":[],"scores":[]}

    def generate(self,samples,*args,**kwargs):
        result=original_generate(self,samples,*args,**kwargs)
        if not self.training:
            if not isinstance(result,dict) or "logits" not in result:
                raise RuntimeError("VALID score capture requires logit dictionary")
            for key,source in [("users","UserID"),("labels","label")]:
                captured[key].append(samples[source].detach().cpu().numpy().reshape(-1))
            captured["scores"].append(result["logits"].detach().float().cpu().numpy().reshape(-1))
        return result

    def eval_with_both(runner,split_name,cur_epoch,skip_reload=False):
        captured["users"].clear();captured["labels"].clear();captured["scores"].clear()
        result=original_eval(runner,split_name,cur_epoch,skip_reload)
        if split_name!="valid" or result is None:
            return result
        if not captured["scores"]:
            raise RuntimeError("No VALID scores captured, cannot report UAUC")
        y=np.concatenate(captured["labels"])
        u=np.concatenate(captured["users"])
        s=np.concatenate(captured["scores"])
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y))<2: raise RuntimeError("VALID has only one class")
        measured=float(roc_auc_score(y,s))
        official=float(result["agg_metrics"])
        if abs(measured-official)>.005:
            raise RuntimeError(f"VALID scorer mismatch: official={official}, rawYes={measured}")
        user_values=[]
        for uid in np.unique(u):
            mask=u==uid
            if np.unique(y[mask]).size==2:
                user_values.append(float(roc_auc_score(y[mask],s[mask])))
        if not user_values: raise RuntimeError("No VALID users with both classes")
        uauc=float(np.mean(user_values))
        result["valid_auc"]=measured
        result["valid_uauc"]=uauc
        result["valid_uauc_users"]=len(user_values)
        print(f"[VALID BOTH] {trial_name} epoch={cur_epoch} "
              f"auc={measured:.9f} uauc={uauc:.9f} "
              f"eligible_users={len(user_values)} n={len(y)}",flush=True)
        folder=ROOT/"out"/PROTOCOL/trial_name
        folder.mkdir(parents=True,exist_ok=True)
        with (folder/"valid_every_epoch.csv").open("a",newline="",encoding="utf-8") as f:
            writer=csv.writer(f)
            if f.tell()==0:writer.writerow(["epoch","auc","uauc","uauc_users","n"])
            writer.writerow([cur_epoch,measured,uauc,len(user_values),len(y)])
        # Save every epoch; default runner separately saves best by official AUC.
        runner._save_checkpoint(cur_epoch,is_best=False)
        return result

    MiniGPT4Rec_vx.generate_for_samples_v2=generate
    RecRunnerBase.eval_epoch=eval_with_both


def _install_weight_initialization(checkpoint,need_anchor):
    from minigpt4.runners.runner_base import RunnerBase
    original=RunnerBase.train
    def train_after_loading(self):
        model=self.unwrap_dist_model(self.model)
        _compact_load(model,checkpoint)
        if need_anchor:
            model._student_anchor_user=model.rec_encoder.user_embedding.weight.detach().float().clone()
            model._student_anchor_item=model.rec_encoder.item_embedding.weight.detach().float().clone()
            print("[STUDENT MF ANCHOR] initialized from restored RR1 embedding tables",flush=True)
        if self.resume_ckpt_path:
            raise RuntimeError("Student uses fresh AdamW; do NOT resume optimizer from compact RR1")
        return original(self)
    RunnerBase.train=train_after_loading


def _validate_config(path):
    import yaml
    config=yaml.safe_load(Path(path).read_text())
    run=config["run"]
    if int(run["seed"])!=SEED or run.get("test_splits") or run.get("valid_splits")!=["valid"]:
        raise RuntimeError("Invalid seed42 or non-VALID-only student config")
    if int(run["batch_size_train"])!=BATCH or int(run["accum_grad_iters"])!=ACCUM:
        raise RuntimeError("Student batch / gradient accumulation mismatch")
    if int(run["iters_per_epoch"])!=EPOCH_STEPS:
        raise RuntimeError("Student iterations per epoch mismatch")
    if abs(float(run["init_lr"])-LR)>1e-12:
        raise RuntimeError("Student LR differs from intended continuation LR")
    if config["model"]["freeze_rec"] is not False or config["model"].get("ckpt") or run.get("resume_ckpt_path"):
        raise RuntimeError("Student must train MF from an explicitly loaded RR1 model-only checkpoint")
    return config


def prepare():
    import yaml
    if not RR1.is_file() or not SOURCE.is_file():
        raise FileNotFoundError("RR1 checkpoint/config absent; cannot prepare student continuation")
    conf_root=ROOT/"local_configs"/PROTOCOL
    conf_root.mkdir(parents=True,exist_ok=True)
    (ROOT/"log"/PROTOCOL).mkdir(parents=True,exist_ok=True)
    for name,trial in TRIALS.items():
        config=yaml.safe_load(SOURCE.read_text())
        config["run"].update({
            "init_lr":LR,"warmup_lr":min(float(config["run"].get("warmup_lr",LR/10)),LR/10),
            "min_lr":min(float(config["run"].get("min_lr",1e-6)),1e-6),
            "seed":SEED,"batch_size_train":BATCH,"accum_grad_iters":ACCUM,
            "iters_per_epoch":EPOCH_STEPS,"max_epoch":1000,
            "evaluate":False,"valid_splits":["valid"],"test_splits":[],
            "resume_ckpt_path":None,
            "output_dir":str(ROOT/"output"/PROTOCOL/name)})
        config["model"]["freeze_rec"]=False
        config["model"]["ckpt"]=None
        path=conf_root/(name+".yaml")
        if path.is_file():
            _validate_config(path)
        else:
            path.write_text(yaml.safe_dump(config,sort_keys=False),encoding="utf-8")
            _validate_config(path)
        print(f"[STUDENT PLAN] {name}: rank={trial['rank']} ens={trial['ens']} "
              f"mf_anchor={trial['mf_anchor']} config={path}",flush=True)
    print(f"[STUDENT PREPARED] RR1={RR1} "
          f"offline_teachers={TEACHER_DIR} lr={LR} MF_lr_ratio={MF_LR_RATIO} "
          "seed=42 VALID-only, fresh optimizer, 400 microsteps/epoch",flush=True)


def gpu_worker(name,epochs,patience):
    import cora_residual_routing6 as r6
    trial=TRIALS[name]
    r6.configure()
    r6.gpu0_preflight(f"student-{name}")
    config=ROOT/"local_configs"/PROTOCOL/(name+".yaml")
    _validate_config(config)
    if trial["ens"]:
        for x in ("RR1","RR5"):
            if not (TEACHER_DIR/f"teacher_{x}_train.npz").is_file():
                raise FileNotFoundError(f"Missing offline TRAIN teacher {x}; run capture first")
    r6.late.base.core.patch_runner(screen_epochs=epochs,mf_lr_ratio=MF_LR_RATIO,patience=patience)
    r6.late.base.core.install_compact_atomic_best_checkpoint()
    r6.late.base.combo.install_combo_model(r6.late.base.combo.NAMES["C6_KD_BOTH"])
    r6.late.base._patch_runtime_classes(r6.TRIAL_BY_NAME["RR1_L32_BASE"])
    _install_sampler_and_losses(name,trial)
    _install_valid_metrics(name)
    _install_weight_initialization(RR1,bool(trial["mf_anchor"]))
    print(f"[STUDENT WORKER READY] {name} source_epoch37 "
          f"epochs={epochs} valid_every_epoch=True test_disabled=True "
          f"output={ROOT/'output'/PROTOCOL/name}",flush=True)
    sys.argv=["train_collm_mf_din.py","--cfg-path",str(config)]
    runpy.run_path(str(PROJECT/"train_collm_mf_din.py"),run_name="__main__")


def cpu_test():
    import tempfile
    import pandas as pd
    import torch
    import torch.nn.functional as F
    torch.manual_seed(42)
    frame=pd.DataFrame({
        "UserID":np.repeat(np.arange(1,11),4),
        "TargetItemID":np.arange(40)+1,
        "label":np.tile([0,1,0,1],10)})
    sampler=MixedPairBatchSampler(frame,steps=32)
    batches=list(iter(sampler))
    assert len(batches)==32
    for a,b in batches[::4]:
        assert frame.iloc[a].UserID==frame.iloc[b].UserID
        assert frame.iloc[a].label!=frame.iloc[b].label
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("RR1","RR5"):
            np.savez(Path(tmp)/f"teacher_{name}_train.npz",
                sample_index=np.arange(40),users=frame.UserID.to_numpy(),
                items=frame.TargetItemID.to_numpy(),labels=frame.label.to_numpy(),
                scores=np.arange(40,dtype=np.float32)+(0 if name=="RR1" else 1))
        targets=TeacherTargets(frame,tmp)
        q=targets.lookup(torch.tensor([1,3]),torch.device("cpu"))
        assert q.shape==(2,)
        broken=frame.copy()
        broken.loc[0,"TargetItemID"]+=1
        try:TeacherTargets(broken,tmp)
        except RuntimeError as e:assert "identity mismatch" in str(e)
        else:raise AssertionError("Teacher identity corruption not detected")
    s=torch.tensor([.2,-.3],requires_grad=True)
    loss,paired=_pairwise_logistic(s,torch.tensor([1.,0.]),torch.tensor([3,3]))
    assert paired and abs(float(loss)-float(F.softplus(torch.tensor(-.5))))<1e-7
    (loss + F.binary_cross_entropy_with_logits(s,torch.tensor([1.,0.]))).backward()
    assert torch.isfinite(s.grad).all()
    print("[STUDENT CPU PASS] mix user-pairs, exact teacher sample audit, "
          "KD targets, differentiable same-user ranking, seed42",flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu-test",action="store_true")
    ap.add_argument("--plan",action="store_true")
    ap.add_argument("--run",action="store_true")
    ap.add_argument("--only",default=",".join(TRIALS))
    ap.add_argument("--epochs",type=int,default=6)
    ap.add_argument("--patience",type=int,default=6)
    ap.add_argument("--_worker",default="",help=argparse.SUPPRESS)
    args=ap.parse_args()
    os.chdir(PROJECT)
    if args.cpu_test:cpu_test();return
    if args._worker:
        if args._worker not in TRIALS:raise ValueError("Unknown worker")
        gpu_worker(args._worker,args.epochs,args.patience)
        return
    if args.plan==args.run:ap.error("Specify exactly one of --plan or --run")
    if args.epochs<1 or args.patience<1:ap.error("positive epochs/patience required")
    if args.plan:prepare();return
    names=[x.strip() for x in args.only.split(",") if x.strip()]
    if not names or len(names)!=len(set(names)) or any(n not in TRIALS for n in names):
        ap.error("Unknown/duplicate trial name in --only")
    prepare()
    for name in names:
        log=ROOT/"log"/PROTOCOL/(name+".log")
        out=ROOT/"out"/PROTOCOL/name
        if log.exists() or out.exists():
            raise RuntimeError(f"Refusing overwrite of existing {name}: log={log} out={out}")
    for name in names:
        import cora_residual_routing6 as r6
        r6.gpu0_preflight(f"launch-{name}")
        log=ROOT/"log"/PROTOCOL/(name+".log")
        cmd=[sys.executable,"-u",str(Path(__file__).resolve()),
             "--_worker",name,"--epochs",str(args.epochs),"--patience",str(args.patience)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="0")
        print(f"[STUDENT START] trial={name} foreground serial GPU0 log={log}",flush=True)
        with log.open("x",encoding="utf-8") as f:
            proc=subprocess.Popen(cmd,cwd=PROJECT,env=env,stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,text=True,
                                  encoding="utf-8",errors="replace",bufsize=1)
            for line in proc.stdout:
                print(line,end="",flush=True)
                f.write(line)
            code=proc.wait()
        if code:
            raise SystemExit(f"[STUDENT FAIL] {name} exit={code}; see {log}")
    print("[STUDENT COMPLETE] VALID-only T0/T1/T2/T3, no TEST",flush=True)

if __name__=="__main__":
    main()
