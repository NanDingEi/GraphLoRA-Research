#!/usr/bin/env python3
"""GPU-independent tests of the NO31 CoRA conditioning and zero-init behavior."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))

import torch
from torch.nn import functional as F
from runtime import NO31ConditionalAdapter,exact_pair_features

torch.manual_seed(42)
rank=8
mod=NO31ConditionalAdapter(hidden_dim=32,rank=rank)
uids=torch.tensor([2,3])
targets=torch.tensor([5,4])
num_users=839
# Each batch has a user, history item(s), and the actual candidate item.
ids=torch.tensor([2,839+7,839+5, 3,839+9,839+4])
rows=torch.tensor([0,0,0,1,1,1])
emb=torch.randn(len(ids),256,requires_grad=True)
fb={"batch_indices":rows,
    "token_node_ids":ids,
    "local_node_ids":torch.arange(len(ids)),
    "token_gnn":emb}
samples={"UserID":uids,"TargetItemID":targets}
user,item=exact_pair_features(samples,fb,num_users,detach=True)
assert user.shape==item.shape==(2,256)
assert not user.requires_grad and not item.requires_grad
assert torch.equal(item[0],emb[2].detach()) and torch.equal(item[1],emb[5].detach())
coefs=mod(user,item)
assert coefs.shape==(2,2,8)
assert float(coefs.min())>=.5 and float(coefs.max())<=1.5
low=torch.randn(2,5,8)
q=F.linear(low*coefs[:,0].unsqueeze(1),mod.q_up.weight)
v=F.linear(low*coefs[:,1].unsqueeze(1),mod.v_up.weight)
assert torch.count_nonzero(q)==0 and torch.count_nonzero(v)==0
(q.sum()+v.sum()).backward()
assert mod.q_up.weight.grad is not None and mod.q_up.weight.grad.abs().sum()>0
assert mod.v_up.weight.grad is not None and mod.v_up.weight.grad.abs().sum()>0
assert emb.grad is None, "Default CoRA must not add gradients to NGCF"
user2,item2=exact_pair_features(samples,fb,num_users,detach=False)
assert user2.requires_grad and item2.requires_grad
print("[NO31-CORA-CPU-PASS] exact target matching; zero initial residual; Q/V gradients; default detach",flush=True)
