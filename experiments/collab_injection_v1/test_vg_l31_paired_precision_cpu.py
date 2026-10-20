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
wanted = {"rng_snapshot", "rng_restore", "rng_equal", "prompt_state_snapshot", "prompt_state_restore", "rng_differences", "diff_stat", "Aggregate"}
nodes = [n for n in tree.body
         if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in wanted]
assert set(n.name for n in nodes) == wanted
scope = {"torch": torch, "np": np, "random": random, "copy": copy}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
snap = scope["rng_snapshot"]()
assert scope["rng_equal"](snap, scope["rng_snapshot"]())
a = (random.random(), np.random.rand(), torch.randn(4))
scope["rng_restore"](snap)
b = (random.random(), np.random.rand(), torch.randn(4))
assert a[0] == b[0] and a[1] == b[1] and torch.equal(a[2], b[2])
assert not scope["rng_equal"](snap, scope["rng_snapshot"]())

# Regress the bug seen in the real GPU run: prompt printing consumes
# one extra Python random.choice on the first forward only.
class FakeModel:
    has_print_prompt = False
    has_pri_decode = False
    prompt_list_p = None


fake = FakeModel()
random.seed(42)
start_rng = scope["rng_snapshot"]()
start_prompt = scope["prompt_state_snapshot"](fake)
ends = []
for factor in (0, 0, 1, 50):
    scope["rng_restore"](start_rng)
    scope["prompt_state_restore"](fake, start_prompt)
    random.choice([0, 1, 2, 3])  # prompt template
    if not fake.has_print_prompt:
        random.choice([0, 1, 2, 3])  # first prompt-example print
        fake.has_print_prompt = True
    if not fake.has_pri_decode:
        fake.has_pri_decode = True
    if fake.prompt_list_p is None:
        fake.prompt_list_p = [0, 1, 2, 3]
    ends.append(scope["rng_snapshot"]())
assert all(scope["rng_equal"](ends[0], state) for state in ends[1:])
assert all(not scope["rng_differences"](ends[0], state) for state in ends[1:])
assert fake.has_print_prompt and fake.has_pri_decode
scope["prompt_state_restore"](fake, start_prompt)
assert fake.has_print_prompt is False and fake.has_pri_decode is False
print("[VG-PAIRED-CPU] first-prompt RNG/state replay PASS")

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
