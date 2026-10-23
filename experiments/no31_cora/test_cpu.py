#!/usr/bin/env python3
"""NO31-CoRA pretraining-time invariants; CPU, no data, no Vicuna required."""
from pathlib import Path
from types import SimpleNamespace
import sys

import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from adapter import PairConditionGenerator

base=(ROOT/"minigpt4/models/minigpt4rec_v2.py").read_text()
runner=(ROOT/"minigpt4/runners/runner_base.py").read_text()
for marker in (
    "def _s2_final_gnn_scores(self, fb):",
    'self.s2_small_kd = float(os.environ.get("S2_SMALL_KD","0"))',
    '[NIGHT8-LOSS]',
    'assert float(gnn_scale) == 0.0',
    "self._no31_cora_hook(result, lora_a_output)",
):
    assert marker in base, f"Missing historical NO31 or CoRA anchor: {marker}"
assert "[NIGHT8-STOP]" in runner

torch.manual_seed(42)
model=PairConditionGenerator(hidden_size=16,rank=8)
u=torch.randn(3,256)
i=torch.randn(3,256)
r=model(u,i)
assert r.shape==(3,2,8)
low=torch.randn(3,7,8)
outq=F.linear(low*r[:,0].unsqueeze(1),model.q_up.weight)
outv=F.linear(low*r[:,1].unsqueeze(1),model.v_up.weight)
assert torch.count_nonzero(outq)==0 and torch.count_nonzero(outv)==0
(outq.square().sum()+outv.square().sum()).backward()
assert model.q_up.weight.grad is not None
assert model.v_up.weight.grad is not None
# Zero upstream derivative at zero-output is expected with squared losses.
# Use first-order signal to verify the up matrices do receive gradients.
model.zero_grad()
strength=model(u,i)
(F.linear(low*strength[:,0].unsqueeze(1),model.q_up.weight).sum()+
 F.linear(low*strength[:,1].unsqueeze(1),model.v_up.weight).sum()).backward()
assert model.q_up.weight.grad.abs().sum()>0
assert model.v_up.weight.grad.abs().sum()>0
# Verify that the actual runtime residual preserves *all* NO31 outputs
# at initialization and changes ID positions ONLY after updating CoRA weights.
from adapter import No31CoRARuntime
rt=No31CoRARuntime(gain=0.1,detach_gnn=True,active="both")
mock_q=SimpleNamespace(
    module_tag="q",gnn_scale=0.0,gcn_output=torch.ones(2,8),
    _no31_cora_payload=torch.ones(2,8),
    _no31_cora_rows=torch.tensor([0,0,1],dtype=torch.long),
    _no31_cora_positions=torch.tensor([1,3,2],dtype=torch.long))
rt.model=SimpleNamespace(my_linears=[mock_q],r3_adapter=model)
h=torch.arange(2*5*16,dtype=torch.float32).reshape(2,5,16)/100.0
z=torch.ones((2,5,8))
init_result=rt.apply("q",h,z)
assert torch.equal(init_result,h),"CoRA zero-init perturbed historical NO31 predictions"
with torch.no_grad():
    model.q_up.weight[0,0]=1.0
changed=rt.apply("q",h,z)
mask=torch.zeros(2,5,dtype=torch.bool)
mask[mock_q._no31_cora_rows,mock_q._no31_cora_positions]=True
assert torch.equal(changed[~mask],h[~mask]),"CoRA modified non-ID tokens"
assert torch.any(changed[mask]!=h[mask]),"CoRA failed to modify ID tokens"
assert torch.equal(changed[...,1:], h[...,1:]), "CoRA modified unexpected output dimensions in CPU test"
print("[NO31-CORA-CPU-PASS] zero-init exact NO31 forward + ID-only delta + gradients")
