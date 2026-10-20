#!/usr/bin/env python3
"""CPU-only check of the actual target-fusion candidate/user/history ablations."""
import ast
from pathlib import Path

import torch

p = Path(__file__).resolve().parent / "eval_target_fusion_causal.py"
tree = ast.parse(p.read_text(encoding="utf-8"))
funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "transform"]
assert len(funcs) == 1
env = {"torch": torch}
exec(compile(ast.Module(body=funcs, type_ignores=[]), str(p), "exec"), env)
transform = env["transform"]

width, batch = 256, 4
torch.manual_seed(42)
u = torch.randn(batch, width)
t = torch.randn(batch, width)
h = torch.randn(batch, width)
x = torch.cat((u, t, h, u*t, h*t, (u-t).abs()), dim=-1)

assert torch.equal(transform(x, "original"), x)
assert torch.equal(transform(x, "baseline"), x)
shuffle = transform(x, "shuffle_target")
su, st, sh, sut, sht, sd = shuffle.split(width, dim=-1)
assert torch.equal(st, t.roll(1, 0))
assert torch.equal(su, u)
assert torch.equal(sh, h)
assert torch.allclose(sut, su*st)
assert torch.allclose(sht, sh*st)
assert torch.allclose(sd, (su-st).abs())

us = transform(x, "shuffle_user")
uu, ut, uh, uut, uht, ud = us.split(width, dim=-1)
assert torch.equal(uu, u.roll(1, 0))
assert torch.equal(ut, t)
assert torch.allclose(uut, uu * ut)
assert torch.allclose(ud, (uu - ut).abs())

no_target = transform(x, "no_target")
assert torch.count_nonzero(no_target[:, 1*width:2*width]) == 0
assert torch.count_nonzero(no_target[:, 3*width:]) == 0
no_hist = transform(x, "no_history")
assert torch.count_nonzero(no_hist[:, 2*width:3*width]) == 0
assert torch.count_nonzero(no_hist[:, 4*width:5*width]) == 0

try:
    transform(x[:1], "shuffle_target")
except RuntimeError:
    pass
else:
    raise AssertionError("shuffle_target must fail with singleton batch")

print("[CAUSAL-CPU] PASS: candidate/user swaps are consistent; feature masking matches definitions")
