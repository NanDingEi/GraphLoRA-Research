#!/usr/bin/env python3
"""J3: fresh CoRA joint L32 and user-gated L31 training with delayed user-ranking.

Compared with completed J2, change only the user-ranking loss coefficient:
epochs 0-4: 0; epochs 5-9: .004/.008/.012/.016/.020; >=10: .020.
Everything else stays matched to J2: source data, fresh initialization,
D4 collaborative input, C6 auxiliary losses, 1/4 user-pair batches,
fused BCE=.25, final rank coefficient=.02, seed42 and learning rates.
No RR1/RR5 checkpoint load, no teacher model/logits, and no Test evaluation.
One Vicuna-7B on GPU0; full VALID AUC/UAUC every epoch.
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
import torch
import torch.nn.functional as F
import yaml

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent.parent
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))

import cora_residual_routing6 as r6

PROTOCOL="CORA_FRESH_J3_RANK_WARMUP_20261113"
BASE_CONFIG=HERE/"local_configs/CORA_RESIDUAL_ROUTING6_20261103/RR1_L32_BASE.yaml"
CONFIG_DIR=HERE/"local_configs"/PROTOCOL
LOG_DIR=HERE/"log"/PROTOCOL
OUTPUT_DIR=HERE/"output"/PROTOCOL
RESULT_DIR=HERE/"out"/PROTOCOL
SEED=42
BASE_LR=3e-4
MF_LR_RATIO=.1
BATCH_SIZE=2
ACCUM=4
STEPS_PER_EPOCH=400
PAIR_PERIOD=4
GATE_AUX_WEIGHT=.25
RANK_WEIGHT=.02

RANK_WARMUP_EPOCHS=5
RANK_RAMP_EPOCHS=5

TRIALS={
 "J3_FRESH_DELAYED_RANK":dict(
     name="J3_FRESH_DELAYED_RANK",kind="user_gate",
     mode="neighbor_mix",personal_share=.25,
     joint_weight=GATE_AUX_WEIGHT,rank_weight=RANK_WEIGHT),
}


def scheduled_rank_coefficient(epoch, final_weight=RANK_WEIGHT):
    """Return the final rank weight after a fixed, train-time-only warm-up.

    0-based epochs: [0,4] -> 0, [5,9] -> ramp, [10,...] -> 0.02.
    This schedule is predetermined and is NEVER selected on VALID or TEST.
    """
    if epoch < 0:
        raise ValueError("Negative epoch")
    if epoch < RANK_WARMUP_EPOCHS:
        return 0.0
    progress=min(1.0,(epoch - RANK_WARMUP_EPOCHS + 1)/RANK_RAMP_EPOCHS)
    return float(final_weight)*progress


class UserBalancedPairSampler:
    """Same minibatch schedule for every trial; label/identity only from TRAIN."""
    def __init__(self,ann,seed=SEED,steps=STEPS_PER_EPOCH):
        if not all(k in ann.columns for k in ("UserID","TargetItemID","label")):
            raise RuntimeError("Expected original TRAIN Movielens annotation columns")
        user=np.asarray(ann["UserID"],dtype=np.int64)
        label=np.asarray(ann["label"],dtype=np.int64)
        if len(user)<100 or not np.isin(label,[0,1]).all():
            raise RuntimeError("Invalid/too-small TRAIN data for mixed user sampler")
        grouped={}
        for idx,(u,y) in enumerate(zip(user,label)):
            grouped.setdefault(int(u),[[],[]])[int(y)].append(idx)
        self.pairs=[(np.asarray(v[1],dtype=np.int64),np.asarray(v[0],dtype=np.int64))
                    for v in grouped.values() if len(v[0]) and len(v[1])]
        if not self.pairs:
            raise RuntimeError("No train user has both positive and negative interactions")
        self.n=len(user)
        self.seed=int(seed)
        self.steps=int(steps)
        self.epoch=0
    def __len__(self):return self.steps
    def __iter__(self):
        rng=np.random.RandomState(self.seed+self.epoch)
        self.epoch+=1
        for step in range(self.steps):
            if step % PAIR_PERIOD==0:
                positive,negative=self.pairs[int(rng.randint(len(self.pairs)))]
                yield [int(positive[rng.randint(len(positive))]),
                       int(negative[rng.randint(len(negative))])]
            else:
                ids=rng.randint(0,self.n,size=BATCH_SIZE)
                yield [int(ids[0]),int(ids[1])]


def pair_rank_loss(pred,labels,users):
    """Per-user positive/negative score difference, computed within a paired batch."""
    if pred.numel()!=BATCH_SIZE or labels.numel()!=BATCH_SIZE:
        raise RuntimeError("User-ranking batch must contain exactly two samples")
    paired=bool(users[0]==users[1] and labels[0]!=labels[1])
    if not paired:
        return pred.sum()*0.,False
    pos=torch.nonzero(labels>.5,as_tuple=False).flatten()
    neg=torch.nonzero(labels<.5,as_tuple=False).flatten()
    if pos.numel()!=1 or neg.numel()!=1:
        raise RuntimeError("Bad user rank positive/negative pair")
    return F.softplus(-(pred[pos[0]]-pred[neg[0]])),True


def install_equal_sampler(trial):
    """Patch only the TRAIN loader. Validation stays exactly the official dataset."""
    from torch.utils.data import DataLoader
    from minigpt4.runners.runner_base import RunnerBase
    from minigpt4.datasets.datasets.dataloader_utils import PrefetchLoader,IterLoader
    original_get=RunnerBase.train_loader.fget

    def get_sampler_loader(runner):
        if hasattr(runner,"_fresh_joint_train_loader"):
            return runner._fresh_joint_train_loader
        _=original_get(runner)  # Initializes official datasets/VALID loader.
        train=runner.datasets.get("train")
        if isinstance(train,(tuple,list)):
            if len(train)!=1:
                raise RuntimeError("Fresh joint runner expects a single TRAIN dataset")
            train=train[0]
        ann=getattr(train,"annotation",None)
        if ann is None or len(ann)<1000:
            raise RuntimeError("Fresh joint sampler requires full TRAIN annotation DataFrame")
        sampler=UserBalancedPairSampler(ann)
        dataloader=DataLoader(train,batch_sampler=sampler,
                              num_workers=0,pin_memory=True,
                              collate_fn=getattr(train,"collater",None))
        runner._fresh_joint_train_loader=IterLoader(
            PrefetchLoader(dataloader),use_distributed=False)
        print(f"[FRESH TRAIN SAMPLER] trial={trial['name']} "
              f"train_n={len(train)} eligible_pair_users={len(sampler.pairs)} "
              f"pair_every={PAIR_PERIOD} microbatch={BATCH_SIZE} "
              f"steps={STEPS_PER_EPOCH} seed={SEED} "
              "ALL J0/J1/J2 use identical sample schedule",flush=True)
        return runner._fresh_joint_train_loader

    RunnerBase.train_loader=property(get_sampler_loader)


def install_fresh_init_audit(trial):
    """No compact RR1/RR5 state load, no source model.ckpt or resume."""
    from minigpt4.runners.runner_base import RunnerBase
    original_train=RunnerBase.train
    def audited_train(runner):
        model=runner.unwrap_dist_model(runner.model)
        if runner.start_epoch!=0 or runner._optimizer is not None:
            raise RuntimeError("Not a fresh optimization run")
        run=runner.config.run_cfg
        model_config=runner.config.model_cfg
        if run.get("resume_ckpt_path") or model_config.get("ckpt"):
            raise RuntimeError("FRESH experiment must not load any previous CoRA checkpoint")
        trained=sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trained<=0:
            raise RuntimeError("No trainable CoRA/MF parameters")
        expected_gate=(trial["kind"]=="user_gate")
        actual_gate=hasattr(model,"cora_residual_fusion")
        if expected_gate!=actual_gate:
            raise RuntimeError("Gate parameter initialization does not match trial")
        print(f"[FRESH INITIALIZATION PASS] trial={trial['name']} "
              f"trainable_parameters={trained} init_epoch=0 "
              f"optimizer=NEW AdamW CoRA=FRESH "
              f"MF=original_pretrained_init "
              f"external_RR1_RR5_ckpts=NOT_LOADED "
              f"gate={'fresh' if actual_gate else 'none'}",flush=True)
        return original_train(runner)
    RunnerBase.train=audited_train


def install_joint_loss_and_prediction(trial):
    """Retain C6 L32 BCE and MF CTR/KD, add the gated branch without replacing L32."""
    if trial["kind"]=="baseline":
        print("[FRESH MAIN LOSS] J0 original C6 L32+BCE, matched pair sampler",flush=True)
        return

    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx
    original_train=MiniGPT4Rec_vx.forward_v2
    original_eval=MiniGPT4Rec_vx.generate_for_samples_v2
    state={"microsteps":0,"rank_pairs":0,"last_gate":None}

    def capture_and_gate(model,samples,forward_fn,*args,**kwargs):
        result,base,layer,ui,err=r6._capture_live(
            model,samples,forward_fn,True,*args,**kwargs)
        fusion=model.cora_residual_fusion
        gated,strength,weights,residuals,reg=fusion(ui,layer,base)
        if not torch.isfinite(gated).all():
            raise RuntimeError("Nonfinite fused Yes logit")
        return result,base,gated,strength,reg,err

    def forward(model,samples,*args,**kwargs):
        result,base,gated,strength,reg,err=capture_and_gate(
            model,samples,original_train,*args,**kwargs)
        if not isinstance(result,dict) or "loss" not in result:
            raise RuntimeError("C6 forward must yield its original training loss")
        labels=samples["label"].float().to(gated.device).view(-1)
        users=samples["UserID"].long().view(-1)
        if tuple(gated.shape)!=tuple(labels.shape):
            raise RuntimeError("Fused layer logits/label count mismatch")
        main_base=F.binary_cross_entropy_with_logits(base.float(),labels)
        main_gate=F.binary_cross_entropy_with_logits(gated.float(),labels)
        # Original result includes original C6 L32 BCE + CTR + MF KD.
        # Do NOT subtract/replace original BCE as in RR5 single-branch training.
        loss=result["loss"] + trial["joint_weight"]*main_gate + reg
        rank,paired=pair_rank_loss(gated,labels,users)
        if "epoch" not in samples:
            raise RuntimeError("Missing true TRAIN epoch for delayed rank schedule")
        epoch=int(samples["epoch"])
        rank_lambda=scheduled_rank_coefficient(epoch,trial["rank_weight"])
        if rank_lambda:
            loss=loss+rank_lambda*rank
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Nonfinite fresh joint loss")
        result["loss"]=loss
        result["logits"]=gated
        state["microsteps"]+=1
        if paired:
            state["rank_pairs"]+=1
        if state["microsteps"]>100 and state["rank_pairs"]==0:
            raise RuntimeError("Mixed user sampler not yielding same-user positive/negative pairs")
        if state["microsteps"]==1 or state["microsteps"]%100==0:
            print(f"[FRESH JOINT LOSS] {trial['name']} step={state['microsteps']} "
                  f"l32_bce={main_base.item():.6f} "
                  f"gate_bce={main_gate.item():.6f} "
                  f"epoch={epoch} rank_lambda={rank_lambda:.5f} "
                  f"rank_loss={rank.item():.6f} pair={paired} "
                  f"gate_mean={strength.detach().mean().item():.6f} "
                  f"gate_std={strength.detach().std(unbiased=False).item():.6f} "
                  f"gate_reg={reg.detach().item():.6g} "
                  f"pair_count={state['rank_pairs']} "
                  f"total={loss.detach().item():.6f} "
                  f"l32_capture_err={err:.6g}",flush=True)
        return result

    def evaluate(model,samples,*args,**kwargs):
        result,base,gated,strength,reg,err=capture_and_gate(
            model,samples,original_eval,*args,**kwargs)
        labels=samples["label"].float().to(gated.device).view(-1)
        if isinstance(result,tuple) and len(result)==2:
            return result[0],gated
        if not isinstance(result,dict):
            raise RuntimeError("Unexpected validation output from C6")
        result["logits"]=gated
        result["loss"]=F.binary_cross_entropy_with_logits(gated.float(),labels)
        return result

    MiniGPT4Rec_vx.forward_v2=forward
    MiniGPT4Rec_vx.generate_for_samples_v2=evaluate
    print(f"[FRESH JOINT MODULE] {trial['name']} "
          f"one_model_two_paths=True "
          f"loss=L32_BCE+C6_CTR_KD+{trial['joint_weight']}*gate_BCE"
          f"+{trial['rank_weight']}*user_RANK+gate_prior; "
          "VALID=fused Yes logit",flush=True)


def install_epoch_validation_and_checkpoints(trial):
    from minigpt4.runners.runner_base_rec import RecRunnerBase
    original_eval=RecRunnerBase.eval_epoch

    def validate_and_save(runner,split_name,cur_epoch,skip_reload=False):
        result=original_eval(runner,split_name,cur_epoch,skip_reload)
        if split_name!="valid" or result is None:
            return result
        if not isinstance(cur_epoch,int):
            raise RuntimeError("Epoch must be integer in VALID-only experiment")
        if "agg_metrics" not in result or "uauc" not in result:
            raise RuntimeError("Official CoRA validation must return both AUC and UAUC")
        auc=float(result["agg_metrics"])
        uauc=float(result["uauc"])
        if not np.isfinite([auc,uauc]).all():
            raise RuntimeError("Nonfinite VALID AUC or UAUC")
        print(f"[FRESH VALID] {trial['name']} epoch={cur_epoch} "
              f"auc={auc:.9f} uauc={uauc:.9f} "
              f"seed={SEED} test=DISABLED",flush=True)
        results=RESULT_DIR/trial["name"]
        results.mkdir(parents=True,exist_ok=True)
        path=results/"valid_every_epoch.csv"
        with path.open("a",newline="",encoding="utf-8") as handle:
            writer=csv.writer(handle)
            if handle.tell()==0:writer.writerow(["epoch","valid_auc","valid_uauc","seed"])
            writer.writerow([cur_epoch,auc,uauc,SEED])
        # Preserve EVERY epoch's checkpoint so user can compare/select on VALID.
        # Official runner still separately saves checkpoint_best.pth by VALID AUC.
        runner._save_checkpoint(cur_epoch,is_best=False)
        return result

    RecRunnerBase.eval_epoch=validate_and_save


def read_config(path):
    if not Path(path).is_file():raise FileNotFoundError(path)
    cfg=yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    run=cfg["run"]
    check={
        "seed42":int(run["seed"])==SEED,
        "LR_fresh":abs(float(run["init_lr"])-BASE_LR)<1e-12,
        "batch2":int(run["batch_size_train"])==BATCH_SIZE,
        "accum4":int(run["accum_grad_iters"])==ACCUM,
        "400_microsteps":int(run["iters_per_epoch"])==STEPS_PER_EPOCH,
        "no_resume":not run.get("resume_ckpt_path"),
        "no_model_ckpt":not cfg["model"].get("ckpt"),
        "MF_trainable":cfg["model"]["freeze_rec"] is False,
        "valid_only":run.get("valid_splits")==["valid"] and not run.get("test_splits"),
        "original_MF_pretraining":bool(cfg["model"].get("rec_config",{}).get("pretrained_path")),
    }
    bad=[k for k,v in check.items() if not v]
    if bad:raise RuntimeError(f"Fresh joint config invalid ({path}): {bad}")
    return cfg


def prepare():
    if not BASE_CONFIG.is_file():
        raise FileNotFoundError(f"Existing validated R6 source config missing: {BASE_CONFIG}")
    src=read_config(BASE_CONFIG)
    # Reuse exactly R6 architecture/D4/C6, NOT any old checkpoint weights.
    CONFIG_DIR.mkdir(parents=True,exist_ok=True)
    LOG_DIR.mkdir(parents=True,exist_ok=True)
    for name,trial in TRIALS.items():
        path=CONFIG_DIR/(name+".yaml")
        config=yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
        config["run"]["seed"]=SEED
        config["run"]["output_dir"]=str(OUTPUT_DIR/name)
        config["run"]["resume_ckpt_path"]=None
        config["run"]["test_splits"]=[]
        config["run"]["valid_splits"]=["valid"]
        config["run"]["evaluate"]=False
        config["model"]["ckpt"]=None
        if path.exists():
            saved=read_config(path)
            if saved["run"]["output_dir"]!=config["run"]["output_dir"]:
                raise RuntimeError(f"Existing {path} output path differs; refusing overwrite")
        else:
            path.write_text(yaml.safe_dump(config,sort_keys=False),encoding="utf-8")
        read_config(path)
        print(f"[FRESH J3 PLAN] {name} "
              f"L32_BCE=1.0 gate_BCE={trial['joint_weight']} "
              f"same_user_rank=scheduled_0_to_{trial['rank_weight']} "
              f"original_pretrained_MF=True CoRA_fresh=True "
              f"config={path}",flush=True)
    manifest={
        "protocol":PROTOCOL,"seed":SEED,"fresh_from":"original Vicuna-7B + pretrained MF only",
        "disallowed":["RR1 checkpoint initialization","RR5 checkpoint initialization",
                      "two-model teachers","ensemble distillation","optimizer resume"],
        "D4_input":"neighbor_mix TopK40/20/10/5, MF personal_share=.25",
        "C6_auxiliary":{"CTR":.005,"MF_KD":.01,"KD_gate":"agree_conf"},
        "main_loss":"BCE(L32,y) + 0.25*BCE(L32+user_gate*(L31-L32),y) + scheduled pair ranking + gate prior",
        "training":{"lr":BASE_LR,"mf_lr_scale":MF_LR_RATIO,"batch":BATCH_SIZE,
                    "grad_accum":ACCUM,"steps_per_epoch":STEPS_PER_EPOCH,
                    "all_variants_same_user_pair_sampler":True,
                    "pair_microbatch_fraction":1/PAIR_PERIOD},
        "eval":"VALID AUC/UAUC every epoch, checkpoint every epoch, TEST disabled",
        "rank_schedule":{"warmup_epochs":RANK_WARMUP_EPOCHS,
                          "ramp_epochs":RANK_RAMP_EPOCHS,
                          "0_to_4":0,
                          "5_to_9":[0.004,0.008,0.012,0.016,0.020],
                          "10_plus":0.020,
                          "matched_with_J2":"only rank loss timing differs"},
        "trials":TRIALS,
    }
    (CONFIG_DIR/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"[FRESH READY] configs={CONFIG_DIR} log={LOG_DIR} "
          f"out={RESULT_DIR} ckpt={OUTPUT_DIR} "
          "GPU0 ONLY, seed42, NEVER LOAD RR1/RR5",flush=True)


def gpu_worker(name,screen_epochs,patience):
    trial=TRIALS[name]
    r6.configure()
    r6.gpu0_preflight(f"fresh-{name}")
    cfg=read_config(CONFIG_DIR/(name+".yaml"))
    r6.late.base.core.patch_runner(screen_epochs=screen_epochs,
                                   mf_lr_ratio=MF_LR_RATIO,patience=patience)
    r6.late.base.core.install_compact_atomic_best_checkpoint()
    r6.late.base.combo.install_combo_model(r6.late.base.combo.NAMES["C6_KD_BOTH"])
    r6.late.base._patch_runtime_classes(trial)
    if trial["kind"]=="user_gate":
        r6.install_trial_module(trial)
    r6.install_optimizer_audit(trial)
    install_equal_sampler(trial)
    install_joint_loss_and_prediction(trial)
    install_epoch_validation_and_checkpoints(trial)
    install_fresh_init_audit(trial)
    print(f"[FRESH WORKER] {name} source=original_Vicuna+pretrained_MF "
          f"no_RR1_or_RR5_ckpt=True "
          f"optimizer=fresh seed={SEED} "
          f"epoch_cap={screen_epochs or 'patience'} "
          f"test=DISABLED config={CONFIG_DIR/(name+'.yaml')}",flush=True)
    sys.argv=["train_collm_mf_din.py","--cfg-path",str(CONFIG_DIR/(name+".yaml"))]
    runpy.run_path(str(PROJECT/"train_collm_mf_din.py"),run_name="__main__")


def cpu_test():
    import pandas as pd
    torch.manual_seed(SEED)
    ann=pd.DataFrame({
        "UserID":np.repeat(np.arange(1,31),4),
        "TargetItemID":np.arange(120)+1,
        "label":np.tile([0,1,0,1],30)})
    first=UserBalancedPairSampler(ann,steps=24)
    second=UserBalancedPairSampler(ann,steps=24)
    a=list(iter(first))
    b=list(iter(second))
    assert a==b
    for i,(pos,neg) in enumerate(a):
        if i%PAIR_PERIOD==0:
            assert ann.iloc[pos]["UserID"]==ann.iloc[neg]["UserID"]
            assert ann.iloc[pos]["label"]==1 and ann.iloc[neg]["label"]==0
    z=torch.randn(4,2,requires_grad=True)
    ui=torch.randn(2,32)
    gate=r6.ResidualFusion("user_gate",16)
    fused,strength,weights,residuals,reg=gate(ui,z,z[-1])
    labels=torch.tensor([1.,0.])
    users=torch.tensor([3,3])
    rank,pair=pair_rank_loss(fused,labels,users)
    assert pair and rank.ndim==0
    base=F.binary_cross_entropy_with_logits(z[-1],labels)
    aux=F.binary_cross_entropy_with_logits(fused,labels)
    loss=base + .25*aux + .02*rank +reg
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in gate.parameters())
    assert all(not t["name"].startswith("RR") for t in TRIALS.values())
    expected=[0.0]*5 + [.004,.008,.012,.016,.02,.02]
    for epoch,target in enumerate(expected):
        assert abs(scheduled_rank_coefficient(epoch)-target)<1e-10, (epoch,target)
    assert len(TRIALS)==1 and "J3_FRESH_DELAYED_RANK" in TRIALS
    print("[FRESH J3 CPU TEST PASS] deterministic identical sampler, "
          "user positive/negative pairs, L32+gated joint BCE, "
          "rank gradients, parameter registration, no old checkpoints",flush=True)


def run_trials(names,screen_epochs,patience):
    prepare()
    for name in names:
        log=LOG_DIR/(name+".log")
        result=RESULT_DIR/name
        out=OUTPUT_DIR/name
        if log.exists() or result.exists() or (out.exists() and any(out.iterdir())):
            raise RuntimeError(f"Refuse overwriting a prior attempt: {name} log={log} out={out}")
    for name in names:
        r6.gpu0_preflight(f"launcher-{name}")
        log=LOG_DIR/(name+".log")
        cmd=[sys.executable,"-u",str(Path(__file__).resolve()),
             "--_worker",name,"--screen-epochs",str(screen_epochs),
             "--patience",str(patience)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="0")
        print(f"[FRESH SERIAL START] {name} GPU0 seed42 foreground "
              f"log={log} out={OUTPUT_DIR/name}",flush=True)
        with log.open("x",encoding="utf-8") as f:
            proc=subprocess.Popen(cmd,cwd=PROJECT,env=env,stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,text=True,
                                  encoding="utf-8",errors="replace",bufsize=1)
            for line in proc.stdout:
                print(line,end="",flush=True)
                f.write(line)
            code=proc.wait()
        if code!=0:
            raise SystemExit(f"[FRESH FAILED] {name} rc={code}; inspect {log}. "
                             "Later trials not started")
    print("[FRESH COMPLETED] All requested trials, VALID only; no TEST",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-test",action="store_true")
    parser.add_argument("--plan",action="store_true")
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--only",default=",".join(TRIALS))
    parser.add_argument("--screen-epochs",type=int,default=0,
                        help="0=original scheduler with AUC early-stopping; >0=cap valid epochs")
    parser.add_argument("--patience",type=int,default=10)
    parser.add_argument("--_worker",default="",help=argparse.SUPPRESS)
    args=parser.parse_args()
    os.chdir(PROJECT)
    if args.cpu_test:cpu_test();return
    if args._worker:
        if args._worker not in TRIALS:parser.error("Unknown internal trial")
        gpu_worker(args._worker,args.screen_epochs,args.patience)
        return
    if args.plan==args.run:
        parser.error("Specify exactly one of --plan or --run")
    if args.screen_epochs<0 or args.patience<1:
        parser.error("screen-epochs >=0 and patience>=1 required")
    if args.plan:prepare();return
    names=[x.strip() for x in args.only.split(",") if x.strip()]
    if len(names)!=len(set(names)) or any(x not in TRIALS for x in names):
        parser.error("Unknown or duplicate --only trial")
    run_trials(names,args.screen_epochs,args.patience)

if __name__=="__main__":
    main()
