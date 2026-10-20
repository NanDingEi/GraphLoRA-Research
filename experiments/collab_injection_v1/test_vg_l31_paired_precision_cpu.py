#!/usr/bin/env python3
"""CPU-only sanity checks for VG1 paired L31 activation diagnostic helpers."""
import ast
import copy
import random
from pathlib import Path

import numpy as np
import torch

path = Path(__file__).resolve().parent / "eval_vg_l31_paired_precision.py"
tree = ast.parse(path.read_text(encoding="utf-8"))
wanted = {"rng_snapshot", "rng_restore", "diff_stat", "Aggregate"}
nodes = [n for n in tree.body
         if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in wanted]
assert set(n.name for n in nodes) == wanted
scope = {"torch": torch, "np": np, "random": random, "copy": copy}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
snap = scope["rng_snapshot"]()
a = (random.random(), np.random.rand(), torch.randn(4))
scope["rng_restore"](snap)
b = (random.random(), np.random.rand(), torch.randn(4))
assert a[0] == b[0] and a[1] == b[1] and torch.equal(a[2], b[2])

base = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
same = scope["diff_stat"](base, base)
assert same["nonzero_fraction"] == same["mean_abs"] == same["max_abs"] == 0.0
shifted = base + 1.0
r = scope["diff_stat"](shifted, base)
assert r["mean_abs"] == 1.0 and r["nonzero_fraction"] == 1.0
assert r["max_abs"] == 1.0 and r["vectors"] == 2
agg = scope["Aggregate"]()
agg.add("noisy/off", r)
agg.add("noisy/off", r)
o = agg.output()["noisy/off"]
assert o["num_elements"] == 8
assert o["mean_abs"] == o["nonzero_fraction"] == o["max_abs"] == 1.0
print("[VG-PAIRED-CPU] PASS: paired RNG, identical-output noise, aggregation")
