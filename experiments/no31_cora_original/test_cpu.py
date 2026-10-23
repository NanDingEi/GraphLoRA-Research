#!/usr/bin/env python3
"""CPU smoke test of the ORIGINAL CoRA generated-matrix + PLoRA formula."""
import sys
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.no31_cora_original.weight_generator import PerceptualWeightsGenerator

torch.manual_seed(42)
# Scale only the dimensions for cheap CPU checking. Original training uses
# e=256, hidden=4096, rank=16, QFormer=8 layers / 8 heads.
generator = PerceptualWeightsGenerator(
    num_queries=4, user_embedding_size=16, hidden_size=32, r=4
)
u_i = torch.randn(3,32)
w = generator(u_i)
assert w.shape == (3,32,4), w.shape
assert torch.allclose(w.flatten(1).norm(dim=-1), torch.ones(3), atol=1e-5)
assert len(generator.qformer.encoder.layer)==8
x = torch.randn(3,5,32)
p = torch.nn.Linear(4,32,bias=False)
torch.nn.init.zeros_(p.weight)
base = F.linear(x,torch.eye(32))
delta = 2 * p(torch.bmm(x,w))
assert torch.count_nonzero(delta)==0
assert torch.equal((base+delta),base), "Zero P must preserve frozen W exactly"
(delta.sum()).backward()
assert p.weight.grad is not None and p.weight.grad.abs().sum()>0
p.weight.grad.zero_()
p.weight.data[0,0] = 0.001
generator.zero_grad()
loss=p(torch.bmm(x,generator(u_i))).sum()
loss.backward()
assert generator.final.weight.grad is not None
assert generator.final.weight.grad.abs().sum()>0
print("[CORA-ORIGINAL-CPU-PASS] QFormer8 full dynamic LoRA-A shape norm, zero-P parity, P/gen grads")
