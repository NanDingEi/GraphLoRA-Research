#!/usr/bin/env python3
"""CPU-only checks for the approved S2 + four-CoRA, two-epoch experiment plan."""
import copy
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
import yaml

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from night5 import EXPERIMENTS_5, SOURCE, config_for
from s2_cora_adapter import CollaborativeQueryGenerator

runs=EXPERIMENTS_5
assert len(runs)==5
assert len({r['id'] for r in runs})==5
assert all(r['seed']==42 for r in runs)
assert sum(r['mode']=='baseline' for r in runs)==1
assert sum(r['mode']=='cora' for r in runs)==4
assert len({(r['mode'],r['branch'],r['feature'],r['attention'],r['gain']) for r in runs})==5

base=yaml.safe_load(SOURCE.read_text())
configs=[config_for(base,p) for p in runs]
canonical=copy.deepcopy(configs[0])
canonical['run'].pop('output_dir')
for p,cfg in zip(runs,configs):
    assert cfg['run']['seed']==42
    assert cfg['run']['max_epoch']==3
    assert cfg['run']['stop_after_n_epochs']==2
    assert cfg['run']['save_checkpoint_every_n_epochs']==1
    assert cfg['run']['valid_splits']==['valid']
    assert cfg['run']['test_splits']==[]
    assert cfg['run']['accum_grad_iters']==3
    assert cfg['model']['gnn_semantic_weight']==1e-4
    assert cfg['model']['proj_reg_weight']==5e-4
    assert cfg['model']['ckpt'] is None
    assert cfg['run']['resume_ckpt_path'] is None
    assert cfg['model']['rec_config']['pretrained_path']==base['model']['rec_config']['pretrained_path']
    clone=copy.deepcopy(cfg)
    clone['run'].pop('output_dir')
    assert clone==canonical, f"Unfair recipe difference in {p['id']}"

torch.manual_seed(42)
user=torch.randn(3,256)
item=torch.randn(3,256)
for p in runs:
    if p['mode']=='baseline':
        continue
    model=CollaborativeQueryGenerator(hidden_size=16,graph_dim=256,rank=8,
                                       feature_mode=p['feature'],
                                       attention_mode=p['attention'])
    strengths=model(user,item)
    assert strengths.shape==(3,2,8), p['id']
    h=torch.randn(3,7,8)
    output_q=F.linear(h*strengths[:,0].unsqueeze(1),model.q_up.weight)
    output_v=F.linear(h*strengths[:,1].unsqueeze(1),model.v_up.weight)
    assert torch.count_nonzero(output_q)==0 and torch.count_nonzero(output_v)==0
    (output_q.sum()+output_v.sum()).backward()
    assert model.q_up.weight.grad.abs().sum()>0
    assert model.v_up.weight.grad.abs().sum()>0
print('[FIVE-PLAN-CPU-OK] S2+4 architectures, seed42, two complete epochs, identical S2 recipe, zero-init CoRA')
