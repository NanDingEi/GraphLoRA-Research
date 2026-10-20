#!/usr/bin/env python3
"""Fast CPU-only invariance/gradient checks for TargetAwareDecisionFusion.

Extract the actual class via AST so Vicuna-7B and GraphLoRA dependencies do
not need to be instantiated merely to test initialization and backprop.
"""
import ast
from pathlib import Path
import torch
import torch.nn as nn

source = Path(__file__).resolve().parents[2] / "minigpt4/models/minigpt4rec_v2.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
nodes = [n for n in tree.body if isinstance(n, ast.ClassDef)
         and n.name == "TargetAwareDecisionFusion"]
assert len(nodes) == 1, "Missing candidate-aware fusion class"
env = {"torch": torch, "nn": nn}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), env)
Fusion = env["TargetAwareDecisionFusion"]

torch.manual_seed(42)
state = torch.get_rng_state().clone()
f = Fusion(hidden_size=24, graph_size=256, rank=16)
assert torch.equal(state, torch.get_rng_state()), "Fusion init changed global RNG"

h = torch.randn(3, 24, requires_grad=True)
x = torch.randn(3, 6 * 256, requires_grad=True)
residual = f(h, x)
assert residual.shape == h.shape
assert torch.count_nonzero(residual) == 0, "Zero-init must preserve original output"

loss = ((h + residual) ** 2).sum()
loss.backward()
grad = f.to_hidden.weight.grad
assert grad is not None and torch.isfinite(grad).all()
assert grad.abs().sum() > 0, "No gradient reaches fusion output projector"
assert x.grad is not None and torch.isfinite(x.grad).all(), "GNN feature grad absent"
print("[TARGET-CPU] PASS: zero residual, RNG-neutral init, nonzero projector grad")
