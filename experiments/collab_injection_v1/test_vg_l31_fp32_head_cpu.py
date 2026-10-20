#!/usr/bin/env python3
"""CPU test for the exact FP32 final Yes-head helper used in VG1 diagnostic."""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

path = Path(__file__).resolve().parent / "eval_vg_l31_fp32_head.py"
tree = ast.parse(path.read_text(encoding="utf-8"))
selected = [x for x in tree.body
            if isinstance(x, ast.FunctionDef) and
            x.name == "fp32_yes_from_decoder"]
assert len(selected) == 1
scope = {"torch": torch}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
fn = scope["fp32_yes_from_decoder"]

torch.manual_seed(42)
h = torch.randn(4, 64, dtype=torch.float16)
norm = SimpleNamespace(weight=nn.Parameter(torch.ones(64)), variance_epsilon=1e-6)
head = nn.Linear(64, 13, bias=False)
pos_id = 7
real = fn(h, norm, head, pos_id)
x = h.float()
x = x / torch.sqrt(x.square().mean(-1, keepdim=True) + 1e-6)
expected = torch.matmul(x * norm.weight.float(), head.weight[pos_id].float())
assert torch.allclose(real.flatten(), expected, atol=1e-5, rtol=1e-6)
assert real.shape == (4, 1)
assert real.dtype == torch.float32
assert not real.requires_grad
try:
    fn(h.unsqueeze(0), norm, head, pos_id)
except RuntimeError:
    pass
else:
    raise AssertionError("Expected shape guard")
print("[VG-FP32-CPU] PASS: correct Yes row, RMSNorm and FP32 recomputation")
