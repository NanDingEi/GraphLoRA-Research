#!/usr/bin/env python3
"""CPU-only, fail-fast algebra check for CoRA's initially-zero delta."""
import torch
from torch.nn import functional as F
from s2_cora_adapter import CollaborativeQueryGenerator

torch.manual_seed(42)
m = CollaborativeQueryGenerator(hidden_size=4096, rank=8)
u = torch.randn(2, 256)
i = torch.randn(2, 256)
r = m(u, i)
assert r.shape == (2, 2, 8), r.shape
low = torch.randn(2, 9, 8)
q = F.linear(low * r[:, 0].unsqueeze(1), m.q_up.weight)
v = F.linear(low * r[:, 1].unsqueeze(1), m.v_up.weight)
assert torch.count_nonzero(q) == 0 and torch.count_nonzero(v) == 0
(q.sum() + v.sum()).backward()
assert m.q_up.weight.grad is not None
assert m.v_up.weight.grad is not None
assert m.q_up.weight.grad.abs().sum() > 0
assert m.v_up.weight.grad.abs().sum() > 0
print("[S2-CORA-CPU-PASS] zero initial delta + nonzero trainable Q/V gradients")
