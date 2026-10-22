#!/usr/bin/env python3
"""CPU-only preflight: enforce fixed seed, 9 unique mechanisms and 2 epochs."""
import copy
from pathlib import Path
import sys
import yaml

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import night9

runs=night9.EXPERIMENTS_9
assert len(runs)==9, len(runs)
assert len({p["id"] for p in runs})==9
assert all(p["seed"]==42 for p in runs)
assert sum(p["mode"]=="baseline" for p in runs)==1
assert sum(p["mode"]=="cora" for p in runs)==8
signatures=set()
for p in runs:
    sig=(p["mode"],p["branch"],p["gain"],p["feature"],p["attention"])
    assert sig not in signatures, f"Duplicate architecture {sig}"
    signatures.add(sig)
    assert p["branch"] in ("both","q","v")
    assert p["feature"] in ("both","user_only","item_only","interaction")
    assert p["attention"] in ("full","no_self","no_cross")
    assert 0 < p["gain"] <= 5

original=yaml.safe_load(night9.SOURCE.read_text(encoding="utf-8"))
canonical=None
for p in runs:
    cfg=night9.config_for(original,p)
    assert cfg["run"]["max_epoch"]==3
    assert cfg["run"]["stop_after_n_epochs"]==2
    assert cfg["run"]["seed"]==42
    assert cfg["run"]["save_checkpoint_every_n_epochs"]==1
    assert not cfg["model"].get("ckpt")
    assert cfg["run"]["resume_ckpt_path"] is None
    assert cfg["model"]["rec_config"]["pretrained_path"]==original["model"]["rec_config"]["pretrained_path"]
    assert cfg["model"]["proj_reg_weight"]==0.0005
    assert cfg["model"]["gnn_semantic_weight"]==0.0001
    assert cfg["run"]["accum_grad_iters"]==3
    assert cfg["run"]["lr_scale_gnn"]==0.25
    assert cfg["run"]["test_splits"]==[]
    cfg["run"].pop("output_dir")
    if canonical is None:
        canonical=cfg
    else:
        assert cfg==canonical, f"{p['id']} unexpectedly changes the S2 recipe"

from s2_cora_adapter import CollaborativeQueryGenerator
import torch
from torch.nn import functional as F

torch.manual_seed(42)
inputs=(torch.randn(3,256),torch.randn(3,256))
for p in runs:
    if p["mode"]=="baseline":
        continue
    model=CollaborativeQueryGenerator(hidden_size=16,rank=8,
                                      feature_mode=p["feature"],
                                      attention_mode=p["attention"])
    strength=model(*inputs)
    assert strength.shape==(3,2,8)
    low=torch.randn(3,7,8)
    q=F.linear(low*strength[:,0].unsqueeze(1),model.q_up.weight)
    v=F.linear(low*strength[:,1].unsqueeze(1),model.v_up.weight)
    assert torch.count_nonzero(q)==0 and torch.count_nonzero(v)==0
    (q.sum()+v.sum()).backward()
    assert model.q_up.weight.grad.abs().sum()>0
    assert model.v_up.weight.grad.abs().sum()>0

print("[NINE-PLAN-CPU-PASS] 9 unique architectures, ALL seed 42, "
      "identical pretrained S2 config, 2 full epochs, zero-init trainable CoRA")
