#!/usr/bin/env python3
"""CoRA RR1 warm-start: T0/T1/T2/T3 supervised fine-tuning (seed=42, valid-only).

T0: paired-batch matched extra-training control, original C6.
T1: T0 + same-user positive/negative softplus ranking loss.
T2: T0 + precomputed RR1/RR5 train-only ensemble soft-target KD.
T3: T0 + rank + ensemble KD + MF-table anchor to RR1 initialization.

Other commands: --plan, --capture RR1|RR5, --prepare-teacher, --cpu-test.
Never fetches test labels or uses test data to select training settings.
The existing code, config, PyTorch, CUDA, and checkpoints are unchanged.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import runpy
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
PROJECT = next((p for p in (ROOT, *ROOT.parents)
                if (p/'train_collm_mf_din.py').is_file()), None)
PROTOCOL = 'CORA_TRAIN4_RR1_WARMSTART_20261113'
TEACHER_TAG = 'CORA_TRAIN_TEACHERS_20261113'
SEED=42
LR=3e-5
MF_RATIO=.1
STEPS=400
EPOCHS=6
BATCH=2
ACCUM=4
KDT=2.0
TRIALS={
    'T0_CONTINUE':(0.,0.,0.),
    'T1_USER_RANK':(0.,.02,0.),
    'T2_ENSEMBLE_KD':(.05,0.,0.),
    'T3_JOINT':(.05,.02,5e-4),
}
CKPT_ROOT = ROOT / 'output/CORA_RESIDUAL_ROUTING6_20261103'
RR1_CKPT = CKPT_ROOT / 'RR1_L32_BASE/20261103092/checkpoint_best.pth'
RR5_CKPT = CKPT_ROOT / 'RR5_USER_GATE_R31/20261103153/checkpoint_best.pth'
CONFIG_DIR=ROOT/'local_configs'/PROTOCOL
OUT_DIR=ROOT/'out'/PROTOCOL
LOG_DIR=ROOT/'log'/PROTOCOL
TEACHER_DIR=ROOT/'out'/TEACHER_TAG


def require_root():
    if PROJECT is None:
        raise RuntimeError('CoRA root not found (missing train_collm_mf_din.py)')
    os.chdir(PROJECT)
    if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
    if str(PROJECT) not in sys.path: sys.path.insert(0,str(PROJECT))


def load_checkpoint(model, path, require_exact=True):
    """Warm start trainable-parameter-only checkpoint; NEVER optimizer resume."""
    if not path.is_file(): raise FileNotFoundError(path)
    try: obj=torch.load(str(path),map_location='cpu',weights_only=False)
    except TypeError: obj=torch.load(str(path),map_location='cpu')
    if not isinstance(obj,dict) or not isinstance(obj.get('model'),dict):
        raise RuntimeError('Expected compact model-only checkpoint')
    state=obj['model']
    trainable={name for name,p in model.named_parameters() if p.requires_grad}
    keys=set(state)
    if require_exact and trainable != keys:
        raise RuntimeError('Warm-start parameter mismatch: missing=%s unexpected=%s' %
                           (sorted(trainable-keys)[:12],sorted(keys-trainable)[:12]))
    bad=keys-set(model.state_dict())
    if bad: raise RuntimeError('Unknown checkpoint keys: '+str(sorted(bad)[:12]))
    res=model.load_state_dict(state,strict=False)
    if set(res.missing_keys)&trainable or res.unexpected_keys:
        raise RuntimeError('Unexpected trainable checkpoint load mismatch')
    print('[WARM START] ckpt=%s epoch=%s tensors=%d (fresh AdamW)'%(
        path,obj.get('epoch'),len(state)),flush=True)
    return obj


class PairBatchSampler:
    """Exact user-balanced positive/negative pairs, deterministic seed 42.

    The same sampler is used for all T0-T3, so sampling cannot explain
    a rank-loss effect. One two-item pair per micro-step.
    """
    def __init__(self,users,labels,seed=42):
        users=np.asarray(users).reshape(-1).astype(np.int64)
        labels=np.asarray(labels).reshape(-1).astype(np.int64)
        if len(users)!=len(labels) or not np.isin(labels,[0,1]).all():
            raise ValueError('User IDs and binary labels must align')
        groups=defaultdict(lambda: [[],[]])
        for idx,(u,y) in enumerate(zip(users,labels)):
            groups[int(u)][int(y)].append(idx)
        self.groups={u:v for u,v in groups.items() if v[0] and v[1]}
        self.eligible=np.array(sorted(self.groups),dtype=np.int64)
        if len(self.eligible)==0: raise ValueError('No user with both classes in TRAIN')
        self.length=max(1,len(labels)//2)
        self.epoch=0
        self.seed=int(seed)
    def __len__(self): return self.length
    def __iter__(self):
        rng=np.random.RandomState(self.seed+self.epoch)
        self.epoch+=1
        for _ in range(self.length):
            u=int(self.eligible[rng.randint(len(self.eligible))])
            neg,pos=self.groups[u]
            yield [int(pos[rng.randint(len(pos))]), int(neg[rng.randint(len(neg))])]


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self,base): self.base=base
    def __len__(self): return len(self.base)
    def __getitem__(self,index):
        row=dict(self.base[int(index)])
        row['_ensemble_index']=int(index)
        return row
    def collater(self,rows): return self.base.collater(rows)


def single_train_dataset(runner):
    _=runner.dataloaders # reorganizes dataset keys
    ds=runner.datasets['train']
    if isinstance(ds,(tuple,list)):
        if len(ds)!=1: raise RuntimeError('Need exactly one train dataset, got '+str(len(ds)))
        ds=ds[0]
    if not hasattr(ds,'annotation') or not hasattr(ds.annotation,'columns'):
        raise RuntimeError('Train dataset lacks dataframe annotation')
    for field in ('UserID','TargetItemID','label'):
        if field not in ds.annotation.columns:
            raise RuntimeError('Missing train field '+field)
    if len(ds)!=len(ds.annotation): raise RuntimeError('Train annotation row alignment mismatch')
    return ds


def annotation_arrays(dataset):
    ann=dataset.annotation
    return (ann['UserID'].to_numpy(dtype=np.int64),
            ann['TargetItemID'].to_numpy(dtype=np.int64),
            ann['label'].to_numpy(dtype=np.int64))


def provenance_digest(users,items,labels):
    mat=np.column_stack((users,items,labels)).astype('<i8',copy=False)
    return hashlib.sha256(np.ascontiguousarray(mat).tobytes()).hexdigest()


def validate_teacher_file(path,users,items,labels):
    if not Path(path).is_file(): raise FileNotFoundError(path)
    with np.load(path,allow_pickle=False) as d:
        for key, expected in [('users',users),('items',items),('labels',labels),
                              ('sample_indices',np.arange(len(labels),dtype=np.int64))]:
            if key not in d.files or not np.array_equal(d[key],expected):
                raise RuntimeError('TRAIN teacher mismatch: '+key)
        s=np.asarray(d['teacher_logits'],dtype=np.float32)
        if s.shape!=(len(labels),) or not np.isfinite(s).all():
            raise RuntimeError('Invalid teacher scores')
        if 'provenance' not in d.files or str(d['provenance'])!=provenance_digest(users,items,labels):
            raise RuntimeError('Teacher train-split digest mismatch')
    print('[TEACHER VERIFIED] rows=%d sha256=%s'%(
        len(s),provenance_digest(users,items,labels)),flush=True)
    return torch.from_numpy(s.copy()) # CPU; index into by sample ID


def model_training_patch(trial,teacher_file):
    """Add losses only after the verified RR1+C6 forward; eval unchanged."""
    import cora_residual_routing6 as r6
    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx
    original=MiniGPT4Rec_vx.forward_v2
    kd,rank,anchor=TRIALS[trial]
    def wrapped(self,samples,*args,**kwargs):
        result, z32, layers, _, err=r6._capture_live(
            self,samples,original,False,*args,**kwargs)
        if not isinstance(result,dict) or 'loss' not in result:
            raise RuntimeError('Expected original C6 result loss')
        y=samples['label'].long().to(z32.device).view(-1)
        ids=samples['UserID'].long().to(z32.device).view(-1)
        if len(y)!=2 or int(y[0])!=1 or int(y[1])!=0 or ids[0]!=ids[1]:
            raise RuntimeError('Paired batch violated: require [same-user pos,neg]')
        idx=samples['_ensemble_index'].long().cpu().view(-1)
        total=result['loss']
        rk=F.softplus(-(z32[0].float()-z32[1].float())).mean()
        if rank: total=total+rank*rk
        distill=z32.new_zeros(())
        if kd:
            if not hasattr(self,'_train4_teacher'):
                raise RuntimeError('Teacher not initialized before first forward')
            soft=self._train4_teacher[idx].to(z32.device)
            distill=F.binary_cross_entropy_with_logits(
                z32.float()/KDT,torch.sigmoid(soft/KDT))*(KDT*KDT)
            total=total+kd*distill
        anch=z32.new_zeros(())
        if anchor:
            if not hasattr(self,'_train4_mf_anchor'):
                raise RuntimeError('MF anchor not initialized')
            ur,ir=self._train4_mf_anchor
            iu=samples['UserID'].long().to(z32.device)
            ii=samples['TargetItemID'].long().to(z32.device)
            rec=self.rec_encoder
            anch=(rec.user_embedding.weight[iu].float()-ur[iu].to(z32.device)).square().mean()
            anch=anch+(rec.item_embedding.weight[ii].float()-ir[ii].to(z32.device)).square().mean()
            total=total+anchor*anch
        if not bool(torch.isfinite(total).all()):
            raise FloatingPointError('Nonfinite loss')
        result['loss']=total
        step=getattr(self,'_train4_step',0)+1
        self._train4_step=step
        if step==1 or step%100==0:
            print('[TRAIN4 LOSS] trial=%s step=%d base=%.7g rank=%.7g kd=%.7g anchor=%.7g total=%.7g l32_error=%.6g'%(
                trial,step,float((total-rank*rk-kd*distill-anchor*anch).detach()),
                float(rk.detach()),float(distill.detach()),float(anch.detach()),
                float(total.detach()),err),flush=True)
        return result
    MiniGPT4Rec_vx.forward_v2=wrapped


def install_valid_uauc_and_checkpoints(trial):
    """Retain every epoch checkpoint, and log actual per-user Valid UAUC."""
    import cora_residual_routing6 as r6
    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx
    from minigpt4.runners.runner_base_rec import RecRunnerBase
    original_eval=MiniGPT4Rec_vx.generate_for_samples_v2
    def tracking(self,samples,*args,**kwargs):
        result,z32,layers,err=r6.late._capture_late(self,samples,original_eval,*args,**kwargs)
        if hasattr(self,'_train4_valid_capture'):
            buf=self._train4_valid_capture
            buf.append((samples['UserID'].detach().cpu().numpy().copy(),
                        samples['label'].detach().cpu().numpy().copy(),
                        z32.detach().float().cpu().numpy().copy()))
        return result
    MiniGPT4Rec_vx.generate_for_samples_v2=tracking
    prior=RecRunnerBase.eval_epoch
    def tracked(self,split_name,cur_epoch,skip_reload=False):
        model=self.unwrap_dist_model(self.model)
        model._train4_valid_capture=[]
        try:
            metrics=prior(self,split_name,cur_epoch,skip_reload)
            if split_name=='valid' and isinstance(cur_epoch,int):
                data=model._train4_valid_capture
                if not data: raise RuntimeError('No Valid predictions were captured')
                uu=np.concatenate([x[0] for x in data])
                yy=np.concatenate([x[1] for x in data])
                zz=np.concatenate([x[2] for x in data])
                from sklearn.metrics import roc_auc_score
                global_auc=float(roc_auc_score(yy,zz))
                uas=[]
                for u in np.unique(uu):
                    mask=uu==u
                    if np.unique(yy[mask]).size==2:
                        uas.append(float(roc_auc_score(yy[mask],zz[mask])))
                if not uas: raise RuntimeError('Valid UAUC undefined')
                ua=float(np.mean(uas))
                official=float(metrics['agg_metrics'])
                if abs(global_auc-official)>.005:
                    raise RuntimeError('Official Valid AUC differs from raw Yes logit: %.7f vs %.7f'%(official,global_auc))
                out=Path(self.output_dir)
                out.mkdir(parents=True,exist_ok=True)
                csvpath=out/'valid_auc_uauc.csv'
                exists=csvpath.exists()
                with csvpath.open('a',newline='',encoding='utf8') as f:
                    w=csv.writer(f)
                    if not exists: w.writerow(['trial','epoch','auc','uauc','eligible_users','samples','official_auc'])
                    w.writerow([trial,cur_epoch,global_auc,ua,len(uas),len(yy),official])
                # Existing best AUC checkpoint is separately saved by RunnerBase.train.
                # Save each epoch to audit AUC/UAUC Pareto points without TEST tuning.
                self._save_checkpoint(cur_epoch,is_best=False)
                print('[TRAIN4 VALID] trial=%s epoch=%d auc=%.8f uauc=%.8f eligible_users=%d checkpoint=epoch'%(
                    trial,cur_epoch,global_auc,ua,len(uas)),flush=True)
            return metrics
        finally:
            del model._train4_valid_capture
    RecRunnerBase.eval_epoch=tracked


def gpu_preflight():
    import cora_residual_routing6 as r6
    r6.gpu0_preflight('TRAIN4-pre-load')


def configure_r6():
    import cora_residual_routing6 as r6
    r6.configure()
    return r6


def plan():
    import yaml
    r6=configure_r6()
    base=r6.late.base.core.CONFIG_DIR/'RR1_L32_BASE.yaml'
    if not base.is_file():
        r6.prepare()
    if not base.is_file(): raise FileNotFoundError(base)
    src=yaml.safe_load(base.read_text())
    if int(src['run']['seed'])!=42 or src['model']['freeze_rec'] is not False:
        raise RuntimeError('Expected seeded MF trainable RR1 source config')
    if src['run'].get('test_splits') or src['run'].get('resume_ckpt_path'):
        raise RuntimeError('RR1 source config is not Valid-only fresh-init')
    CONFIG_DIR.mkdir(parents=True,exist_ok=True)
    LOG_DIR.mkdir(parents=True,exist_ok=True)
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    for t in TRIALS:
        cfg=yaml.safe_load(base.read_text())
        run=cfg['run']
        run['output_dir']=str(OUT_DIR/t)
        run['init_lr']=LR
        run['min_lr']=min(float(run.get('min_lr',1e-6)),LR/10)
        run['warmup_lr']=min(float(run.get('warmup_lr',1e-6)),LR/10)
        run['warmup_steps']=0
        run['seed']=42
        run['valid_splits']=['valid']
        run['test_splits']=[]
        run['resume_ckpt_path']=None
        run['evaluate']=False
        run['pref_candidates']=0
        run['batch_size_train']=BATCH
        run['accum_grad_iters']=ACCUM
        run['iters_per_epoch']=STEPS
        cfg['model']['ckpt']=None
        target=CONFIG_DIR/(t+'.yaml')
        if target.is_file():
            old=yaml.safe_load(target.read_text())
            if old!=cfg: raise RuntimeError('Refuse to overwrite changed config: '+str(target))
        else:
            target.write_text(yaml.safe_dump(cfg,sort_keys=False))
        print('[TRAIN4 PLAN] %s LR=%.2g paired-BS2 rank=%g ensKD=%g MFanchor=%g config=%s'%(
            t,LR,TRIALS[t][1],TRIALS[t][0],TRIALS[t][2],target),flush=True)
    info={'protocol':PROTOCOL,'source':'RR1 epoch37 compact checkpoint (model-only)',
          'seed':42,'gpu':'0 serial','epochs':EPOCHS,'microsteps_per_epoch':STEPS,
          'batch':BATCH,'accum':ACCUM,'lr':LR,'mf_ratio':MF_RATIO,
          'loss':'RR1+C6 + train-only ensemble teacher KD + same-user pair softplus + MF anchor',
          'trials':TRIALS,'validation':'every epoch full valid AUC/UAUC + epoch checkpoint; no test',
          'teacher_dir':str(TEACHER_DIR), 'checkpoint':str(RR1_CKPT)}
    (CONFIG_DIR/'manifest.json').write_text(json.dumps(info,indent=2,ensure_ascii=False))
    print('[TRAIN4 PLAN ONLY] no GPU training started',flush=True)


def patch_runner_warmstart(trial,teacher_file):
    from minigpt4.runners.runner_base import RunnerBase
    from minigpt4.datasets.datasets.dataloader_utils import PrefetchLoader,IterLoader
    from torch.utils.data import DataLoader
    orig=RunnerBase.train
    base_train_loader=RunnerBase.train_loader
    def train_loader(self):
        if hasattr(self,'_train4_pair_loader'): return self._train4_pair_loader
        ds=single_train_dataset(self)
        uu,ii,yy=annotation_arrays(ds)
        sampler=PairBatchSampler(uu,yy,SEED)
        wrapped=IndexedDataset(ds)
        dl=DataLoader(wrapped,batch_sampler=sampler,num_workers=int(self.config.run_cfg.num_workers),
                      pin_memory=True,collate_fn=wrapped.collater)
        self._train4_pair_loader=IterLoader(PrefetchLoader(dl),use_distributed=False)
        print('[TRAIN4 PAIR LOADER] trial=%s train_rows=%d eligible_users=%d positive+negative_pairs=1/microstep'%(
            trial,len(ds),len(sampler.eligible)),flush=True)
        if TRIALS[trial][0]:
            t=validate_teacher_file(teacher_file,uu,ii,yy)
            model=self.unwrap_dist_model(self.model)
            model._train4_teacher=t
        return self._train4_pair_loader
    RunnerBase.train_loader=property(train_loader)
    def warm_train(self):
        model=self.unwrap_dist_model(self.model)
        if hasattr(model,'cora_residual_fusion'):
            raise RuntimeError('T0-T3 student must be RR1-only, no extra router')
        load_checkpoint(model,RR1_CKPT)
        if TRIALS[trial][2]:
            rec=model.rec_encoder
            if not hasattr(rec,'user_embedding') or not hasattr(rec,'item_embedding'):
                raise RuntimeError('MF embedding tables missing')
            model._train4_mf_anchor=(rec.user_embedding.weight.detach().float().cpu().clone(),
                                     rec.item_embedding.weight.detach().float().cpu().clone())
            print('[TRAIN4 MF ANCHOR] initialized from RR1 checkpoint',flush=True)
        # _train4_teacher is installed by train_loader property before first forward.
        if self._optimizer is not None:
            raise RuntimeError('Optimizer already initialized before warm-start')
        return orig(self)
    RunnerBase.train=warm_train
    # Some registered runner subclasses may override train(): enforce hook there.
    for cls in RunnerBase.__subclasses__():
        if 'train' in cls.__dict__: cls.train=warm_train


def worker(trial):
    if trial not in TRIALS: raise ValueError(trial)
    require_root()
    gpu_preflight()
    r6=configure_r6()
    cfg=CONFIG_DIR/(trial+'.yaml')
    if not cfg.is_file(): raise FileNotFoundError('Run --plan first: '+str(cfg))
    import yaml
    c=yaml.safe_load(cfg.read_text())
    if c['run'].get('test_splits') or c['run']['seed']!=42:
        raise RuntimeError('Not VALID-only seed42')
    if not RR1_CKPT.is_file(): raise FileNotFoundError(RR1_CKPT)
    teacher=TEACHER_DIR/'ensemble_teacher_train.npz'
    if TRIALS[trial][0] and not teacher.is_file():
        raise FileNotFoundError('Run --capture RR1, --capture RR5, --prepare-teacher: '+str(teacher))
    r6.late.base.core.patch_runner(screen_epochs=EPOCHS,mf_lr_ratio=MF_RATIO,patience=EPOCHS)
    r6.late.base.core.install_compact_atomic_best_checkpoint()
    r6.late.base.combo.install_combo_model(r6.late.base.combo.NAMES['C6_KD_BOTH'])
    r6.late.base._patch_runtime_classes(r6.TRIAL_BY_NAME['RR1_L32_BASE'])
    model_training_patch(trial,teacher)
    install_valid_uauc_and_checkpoints(trial)
    patch_runner_warmstart(trial,teacher)
    print('[TRAIN4 WORKER START] trial=%s seed42, GPU0, valid-only, fresh AdamW from RR1 compact'%trial,flush=True)
    sys.argv=['train_collm_mf_din.py','--cfg-path',str(cfg)]
    runpy.run_path(str(PROJECT/'train_collm_mf_din.py'),run_name='__main__')


def capture_train(teacher):
    """Reconstruct teacher TRAIN-only logits using original model eval path.

    Does not call Runner.train_epoch, optimizer, or test. Each training sample
    is visited once in original dataset order; row ID is checked before save.
    """
    if teacher not in ('RR1','RR5'): raise ValueError(teacher)
    require_root();gpu_preflight()
    r6=configure_r6()
    name='RR1_L32_BASE' if teacher=='RR1' else 'RR5_USER_GATE_R31'
    checkpoint=RR1_CKPT if teacher=='RR1' else RR5_CKPT
    cfg=r6.late.base.core.CONFIG_DIR/(name+'.yaml')
    if not cfg.is_file(): raise FileNotFoundError('Run original cora_residual_routing6.py --plan first')
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    r6.late.base._patch_runtime_classes(r6.TRIAL_BY_NAME[name])
    if teacher=='RR5': r6.install_trial_module(r6.TRIAL_BY_NAME[name])
    from minigpt4.runners.runner_base import RunnerBase
    from minigpt4.datasets.data_utils import prepare_sample
    from torch.utils.data import DataLoader
    def capture_only(self):
        model=self.unwrap_dist_model(self.model)
        load_checkpoint(model,checkpoint)
        self.set_model_mode('v2')
        model.eval()
        ds=single_train_dataset(self)
        uu,ii,yy=annotation_arrays(ds)
        loader=DataLoader(IndexedDataset(ds),batch_size=2,shuffle=False,num_workers=0,
                          pin_memory=True,collate_fn=ds.collater,drop_last=False)
        scores=np.empty(len(ds),dtype=np.float32)
        expected_next=0
        for batch in loader:
            batch=prepare_sample(batch,cuda_enabled=True)
            indices=batch['_ensemble_index'].detach().cpu().numpy().astype(np.int64)
            if not np.array_equal(indices,np.arange(expected_next,expected_next+len(indices))):
                raise RuntimeError('Teacher capture data order mismatch')
            if not (np.array_equal(batch['UserID'].detach().cpu().numpy(),uu[indices]) and
                    np.array_equal(batch['TargetItemID'].detach().cpu().numpy(),ii[indices]) and
                    np.array_equal(batch['label'].detach().cpu().numpy(),yy[indices])):
                raise RuntimeError('Teacher capture sample identity mismatch')
            with torch.no_grad():
                _,z32,layers,ui,err=r6._capture_live(
                    model,batch,type(model).generate_for_samples_v2,teacher=='RR5')
                if teacher=='RR5':
                    score=model.cora_residual_fusion(ui,layers,z32)[0]
                else:
                    score=z32
            scores[indices]=score.detach().float().cpu().numpy()
            expected_next+=len(indices)
            if expected_next%2000<2:
                print('[CAPTURE TRAIN] %s %d/%d'%(teacher,expected_next,len(ds)),flush=True)
        if expected_next!=len(ds) or not np.isfinite(scores).all():
            raise RuntimeError('Teacher capture incomplete or nonfinite')
        TEACHER_DIR.mkdir(parents=True,exist_ok=True)
        target=TEACHER_DIR/(teacher+'_train.npz')
        if target.exists(): raise FileExistsError('Refusing to overwrite teacher: '+str(target))
        np.savez_compressed(target,sample_indices=np.arange(len(ds),dtype=np.int64),
                            users=uu,items=ii,labels=yy,logits=scores,
                            provenance=provenance_digest(uu,ii,yy),
                            checkpoint=str(checkpoint),teacher=teacher)
        print('[CAPTURE TRAIN SAVED] %s rows=%d checkpoint=%s'%(target,len(ds),checkpoint),flush=True)
    RunnerBase.train=capture_only
    for cls in RunnerBase.__subclasses__():
        if 'train' in cls.__dict__: cls.train=capture_only
    print('[CAPTURE START] teacher=%s only TRAIN, no optimization, no TEST'%teacher,flush=True)
    sys.argv=['train_collm_mf_din.py','--cfg-path',str(cfg)]
    runpy.run_path(str(PROJECT/'train_collm_mf_din.py'),run_name='__main__')


def prepare_teacher():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    p1=TEACHER_DIR/'RR1_train.npz';p5=TEACHER_DIR/'RR5_train.npz'
    if not p1.is_file() or not p5.is_file(): raise RuntimeError('Capture both RR1 and RR5 TRAIN first')
    with np.load(p1,allow_pickle=False) as a, np.load(p5,allow_pickle=False) as b:
        for key in ('sample_indices','users','items','labels','provenance'):
            if not np.array_equal(a[key],b[key]): raise RuntimeError('Teacher mismatch: '+key)
        for who,z in [('RR1',a),('RR5',b)]:
            if str(z['teacher'])!=who: raise RuntimeError('Wrong teacher identity for '+who)
        users=a['users'].copy();items=a['items'].copy();labels=a['labels'].copy()
        index=a['sample_indices'].copy()
        mean_a,std_a=float(a['logits'].mean()),float(a['logits'].std())
        mean_b,std_b=float(b['logits'].mean()),float(b['logits'].std())
        if std_a<=1e-10 or std_b<=1e-10: raise RuntimeError('Degenerate teacher')
        blended=.5*(a['logits']-mean_a)/std_a + .5*(b['logits']-mean_b)/std_b
        reg=LogisticRegression(random_state=SEED,max_iter=1000,C=1.0)
        reg.fit(blended.reshape(-1,1),labels)
        logits=(reg.coef_[0,0]*blended+reg.intercept_[0]).astype(np.float32)
        auc=float(roc_auc_score(labels,logits))
        provenance=str(a['provenance'])
    result=TEACHER_DIR/'ensemble_teacher_train.npz'
    if result.is_file(): raise FileExistsError('Do not overwrite frozen teacher: '+str(result))
    np.savez_compressed(result,sample_indices=index,users=users,items=items,labels=labels,
                        teacher_logits=logits,provenance=provenance,
                        teacher_a_checkpoint=str(RR1_CKPT),teacher_b_checkpoint=str(RR5_CKPT),
                        calibration_slope=float(reg.coef_[0,0]),
                        calibration_intercept=float(reg.intercept_[0]),
                        train_auc=auc)
    print('[TEACHER READY] %s rows=%d train_auc=%.8f (TRAIN ONLY, not VALID/TEST)'%(
        result,len(labels),auc),flush=True)


def cpu_test():
    labels=np.tile([1,0,0,1],12)
    users=np.repeat(np.arange(12),4)
    sampler=PairBatchSampler(users,labels)
    a=list(sampler);b=list(PairBatchSampler(users,labels))
    assert a==b
    assert len(a)==len(labels)//2
    for pos,neg in a:
        assert users[pos]==users[neg] and labels[pos]==1 and labels[neg]==0
    model=torch.nn.Linear(4,1)
    x=torch.randn(2,4)
    z=model(x).flatten()
    rk=F.softplus(-(z[0]-z[1]))
    kd=F.binary_cross_entropy_with_logits(z/KDT,torch.sigmoid(torch.tensor([.7,-.2])/KDT))*KDT**2
    (rk*.02+kd*.05).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    us=np.array([1,1,2],dtype=np.int64);it=np.array([3,4,5]);ys=np.array([1,0,1]);
    assert provenance_digest(us,it,ys)==provenance_digest(us.copy(),it.copy(),ys.copy())
    print('[TRAIN4 CPU PASS] user-balanced paired sampler, reproducibility seed42, '
          'finite rank+KD gradients, train row digest',flush=True)


def launch(names):
    for name in names:
        if name not in TRIALS: raise ValueError(name)
        if not (CONFIG_DIR/(name+'.yaml')).is_file(): raise FileNotFoundError('Run --plan first')
        log=LOG_DIR/(name+'.log')
        if log.exists(): raise FileExistsError('Existing experiment log: '+str(log))
    LOG_DIR.mkdir(parents=True,exist_ok=True)
    for name in names:
        gpu_preflight()
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--_worker',name]
        print('[TRAIN4 RUN] '+name+' command='+' '.join(cmd),flush=True)
        log=LOG_DIR/(name+'.log')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
        with log.open('x',encoding='utf8') as f:
            p=subprocess.Popen(cmd,cwd=PROJECT,env=env,stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,encoding='utf8',errors='replace',bufsize=1)
            for line in p.stdout:
                print(line,end='',flush=True);f.write(line)
            rc=p.wait()
        if rc: raise RuntimeError('TRAIN4 stopped trial=%s rc=%d log=%s'%(name,rc,log))
    print('[TRAIN4 COMPLETE] valid-only, checkpoints saved for every epoch, NO TEST',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--plan',action='store_true')
    group.add_argument('--cpu-test',action='store_true')
    group.add_argument('--capture',choices=['RR1','RR5'])
    group.add_argument('--prepare-teacher',action='store_true')
    group.add_argument('--run',action='store_true')
    group.add_argument('--_worker',default='')
    ap.add_argument('--only',default='T0_CONTINUE,T1_USER_RANK,T2_ENSEMBLE_KD,T3_JOINT')
    args=ap.parse_args()
    if args.cpu_test: return cpu_test()
    require_root()
    if args.plan: return plan()
    if args.capture: return capture_train(args.capture)
    if args.prepare_teacher: return prepare_teacher()
    if args._worker: return worker(args._worker)
    if args.run:
        names=[x.strip() for x in args.only.split(',') if x.strip()]
        if not names or len(names)!=len(set(names)): raise ValueError('Empty/duplicate --only')
        return launch(names)

if __name__=='__main__':main()
