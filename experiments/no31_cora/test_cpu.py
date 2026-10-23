#!/usr/bin/env python3
"""NO31-CoRA pretraining-time invariants; CPU, no data, no Vicuna required."""
from pathlib import Path
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
print("[NO31-CORA-CPU-PASS] archival CTR/KD source, zero-init rank8 Q/V, trainable gradients")
