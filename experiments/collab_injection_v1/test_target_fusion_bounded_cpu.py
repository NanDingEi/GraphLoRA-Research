#!/usr/bin/env python3
"""CPU gradient, bound and zero-init checks for smooth target fusion.

Uses AST to import only TargetAwareDecisionFusion, no large LLaMA checkpoint.
"""
import ast
from pathlib import Path
import torch
import torch.nn as nn

source = Path(__file__).resolve().parents[2] / "minigpt4/models/minigpt4rec_v2.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
class_nodes = [n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "TargetAwareDecisionFusion"]
if len(class_nodes) != 1:
    raise RuntimeError("Exactly one TargetAwareDecisionFusion class is required")
env = {"torch": torch, "nn": nn}
exec(compile(ast.Module(body=class_nodes, type_ignores=[]), str(source), "exec"), env)
Fusion = env["TargetAwareDecisionFusion"]

torch.manual_seed(42)
state = torch.get_rng_state().clone()
model = Fusion(hidden_size=24, graph_size=256, rank=16,
               mode="relative_cap", relative_cap=0.02)
if not torch.equal(state, torch.get_rng_state()):
    raise AssertionError("Target fusion initialization consumes global RNG")
h = torch.randn(5, 24, requires_grad=True)
features = torch.randn(5, 6 * 256, requires_grad=True)
residual = model(h, features)
assert residual.shape == h.shape
assert torch.count_nonzero(residual) == 0, "Initial residual must be exactly zero"
loss = (h + residual).square().sum()
loss.backward()
grad = model.to_hidden.weight.grad
assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, \
    "Zero-init projector must receive nonzero finite gradients"
print("[CAP-CPU] zero residual, no RNG leak, nonzero gradient: PASS", flush=True)

with torch.no_grad():
    model.to_hidden.weight.normal_(std=1.0)
    model.gate_out.weight.fill_(0.05)
residual = model(h.detach(), features.detach())
ratio = residual.float().norm(dim=-1) / h.detach().float().norm(dim=-1)
if not bool(torch.isfinite(ratio).all()) or float(ratio.max()) > 0.020001:
    raise AssertionError(f"Relative norm bound violated: {ratio}")
print(f"[CAP-CPU] 2% relative residual bound: PASS (max={float(ratio.max()):.9f})", flush=True)

model.to_hidden.weight.data.zero_()
h0 = torch.zeros(2, 24, requires_grad=True)
x0 = torch.zeros(2, 6 * 256, requires_grad=True)
out = model(h0, x0)
assert torch.isfinite(out).all() and torch.count_nonzero(out) == 0
out.sum().backward()
assert torch.isfinite(model.to_hidden.weight.grad).all(), "Gradient NaN near zero"
print("[CAP-CPU] zero-hidden/zero-residual numerical stability: PASS", flush=True)

# Historical linear mode must remain compatible.
legacy = Fusion(hidden_size=24, graph_size=256, rank=16,
                mode="linear", scale=0.05)
legacy_residual = legacy(h.detach(), features.detach())
assert legacy_residual.shape == h.shape
assert torch.count_nonzero(legacy_residual) == 0
print("[CAP-CPU] historical linear mode: PASS", flush=True)
