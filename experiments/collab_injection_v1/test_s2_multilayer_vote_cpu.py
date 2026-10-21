#!/usr/bin/env python3
"""CPU regression test of model-free late-layer logit-lens and soft voting.

AST-load only the pure helpers: no 7B LLM, dataset, CUDA, transformers or
train_collm entry loaded, and no network access.
"""
import ast
from pathlib import Path
import numpy as np
import torch

path = Path(__file__).with_name("eval_s2_multilayer_vote.py")
tree = ast.parse(path.read_text(encoding="utf-8"))
names = {"parse_layers", "zscore_columns", "predefined_votes",
         "fp32_layer_yes"}
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
         and n.name in names]
assert {n.name for n in nodes} == names
ns = {"np": np, "torch": torch}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
parse = ns["parse_layers"]
assert parse("24,26,28,30,31") == [24,26,28,30,31]
for spec in ("", "24,24,31", "31,24", "0,30", "24,32,31"):
    try:
        parse(spec)
    except ValueError:
        pass
    else:
        raise AssertionError(f"Invalid layer spec passed: {spec}")
print("[VOTE-CPU] layer selection validation PASS", flush=True)

# Deliberate score-scale disparity: z-score votes must be invariant under
# a positive affine change of any intermediate logit column.
np.random.seed(42)
x = np.random.randn(60,5)
native = np.random.randn(60)
votes, info = ns["predefined_votes"](x, native, [24,26,28,30,31])
rescaled = x.copy()
rescaled[:, 2] *= 1000
rescaled[:, 2] += 12345
votes2, _ = ns["predefined_votes"](
    rescaled, native, [24,26,28,30,31])
assert votes.keys() == votes2.keys()
for k in votes:
    assert np.allclose(votes[k], votes2[k], atol=1e-10), k
assert len(votes) == 4
assert info["no_label_fitting"] is True
print("[VOTE-CPU] four fixed soft votes and unsupervised scaling PASS", flush=True)

# Anchored 75/25 vote MUST use native S2 score, not a second copy of
# the layer-31 logit-lens. Replacing only native changes anchored votes
# but not uniform/depth-only votes.
native2 = native[::-1].copy()
votes3, _ = ns["predefined_votes"](x, native2, [24,26,28,30,31])
assert np.array_equal(votes3["vote_equal_all"], votes["vote_equal_all"])
assert np.array_equal(votes3["vote_linear_depth"], votes["vote_linear_depth"])
assert not np.array_equal(votes3["vote_s2_75_aux_25"],
                          votes["vote_s2_75_aux_25"])
print("[VOTE-CPU] native S2 anchor is a distinct expert PASS", flush=True)

# Small synthetic final-RMSNorm Yes row sanity.
class Norm:
    eps = 1e-6
    weight = torch.nn.Parameter(torch.tensor([2.,3.,4.]))
norm = Norm()
h = torch.tensor([[1., -2., 3.], [2., 1., 1.]])
w = torch.tensor([1., 0., -1.])
score = ns["fp32_layer_yes"](h, norm, w)
expected = ((h*h).mean(-1, keepdim=True)+1e-6).rsqrt()*h
expected = ((expected*norm.weight)*w).sum(-1)
assert torch.allclose(score, expected, atol=1e-6)
print("[VOTE-CPU] final RMSNorm and Yes row FP32 reconstruction PASS", flush=True)
