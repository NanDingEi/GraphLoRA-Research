#!/usr/bin/env python3
"""CPU-only VG0/VG1 invariance, gradients, and true context indexing checks."""
import ast
import types
from pathlib import Path

import torch
import torch.nn as nn

path = Path(__file__).resolve().parents[2] / "minigpt4/models/minigpt4rec_v2.py"
tree = ast.parse(path.read_text(encoding="utf-8"))
gate_node = [v for v in tree.body if isinstance(v, ast.ClassDef)
             and v.name == "TargetConditionedVGate"]
model_node = [v for v in tree.body if isinstance(v, ast.ClassDef)
              and v.name == "MiniGPT4Rec_v2"]
assert len(gate_node) == len(model_node) == 1
ctx_node = [v for v in model_node[0].body if isinstance(v, ast.FunctionDef)
            and v.name == "_make_target_v_context"]
assert len(ctx_node) == 1
env = {"torch": torch, "nn": nn}
exec(compile(ast.Module(body=gate_node, type_ignores=[]), str(path), "exec"), env)
Gate = env["TargetConditionedVGate"]
exec(compile(ast.Module(body=ctx_node, type_ignores=[]), str(path), "exec"), env)
context_fn = env["_make_target_v_context"]

torch.manual_seed(42)
cpu_rng = torch.get_rng_state().clone()
gate0 = Gate(graph_size=4, rank=8, mid=12)
assert torch.equal(cpu_rng, torch.get_rng_state()), "Gate created RNG drift"
gate1 = Gate(graph_size=4, rank=8, mid=12)
assert torch.equal(cpu_rng, torch.get_rng_state()), "VG1 gate created RNG drift"
assert gate0.state_dict().keys() == gate1.state_dict().keys()

class Fake:
    num_users = 10
    collab_mode = "target_v_gate_candidate"

fake = Fake()
nodes = torch.tensor([0, 1, 10, 11, 12], dtype=torch.long)
g = torch.arange(len(nodes)*256, dtype=torch.float32).reshape(len(nodes), 256)/1000
fb = {
    "sub_node_ids": nodes,
    "gnn_penult": g,
    "batch_indices": torch.tensor([0, 0, 1, 1], dtype=torch.long),
    "local_node_ids": torch.tensor([0, 3, 1, 4], dtype=torch.long),
}
samples = {"UserID": torch.tensor([0, 1]),
           "TargetItemID": torch.tensor([1, 2])}
vg1 = context_fn(fake, fb, samples)
assert vg1.shape == (4, 1024)
u1, t1, n1, p1 = vg1.split(256, dim=-1)
assert torch.equal(u1, torch.stack([g[0],g[0],g[1],g[1]]))
assert torch.equal(t1, torch.stack([g[3],g[3],g[4],g[4]]))
assert torch.equal(n1, g[fb["local_node_ids"]])
assert torch.equal(p1, t1*n1)

fake.collab_mode = "target_v_gate_no_candidate"
vg0 = context_fn(fake, fb, samples)
u0, t0, n0, p0 = vg0.split(256, dim=-1)
assert torch.equal(u0, u1) and torch.equal(n0, n1)
assert torch.count_nonzero(t0) == 0 and torch.count_nonzero(p0) == 0
print("[VG-CPU] context indexing/target masking PASS", flush=True)

for mode, context in [("VG0", vg0), ("VG1", vg1)]:
    test_gate = Gate(graph_size=256, rank=8)
    delta = test_gate(context)
    assert torch.count_nonzero(delta) == 0
    torch.manual_seed(6)
    z = torch.randn(4, 8)
    a = torch.randn(4, 8)
    scale = 0.1
    base = a + scale*z
    injected = base + (z.float() * scale * delta).to(dtype=a.dtype)
    assert torch.equal(injected, base), "Initial VG must exactly match S2"
    loss = (injected.square()).sum()
    loss.backward()
    grad = test_gate.network[-1].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, \
        "No gradient at zero-init V gate output"
    assert torch.count_nonzero(test_gate.network[-1].weight) == 0, \
        "Backward should not itself update parameters"
    with torch.no_grad():
        test_gate.network[-1].weight.normal_(0, 0.05)
        active = test_gate(context)
    assert torch.isfinite(active).all()
    assert (active.abs() <= 0.5).all(), "Bounded gate correction violated"
    assert torch.count_nonzero(active) > 0, "VG branch never changed after update"
    print(f"[VG-CPU] {mode} zero-init S2 identity, gradient, bounded gate PASS", flush=True)

try:
    bad = dict(fb)
    bad["local_node_ids"] = torch.tensor([0,-1,1,4])
    context_fn(fake, bad, samples)
except RuntimeError:
    pass
else:
    raise AssertionError("Invalid local-node index was not rejected")
print("[VG-CPU] invalid-index preflight PASS")
