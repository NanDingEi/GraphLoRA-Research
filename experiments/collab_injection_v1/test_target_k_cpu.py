#!/usr/bin/env python3
"""CPU-only checks of exact roles, zero-init L31 Key injection and gradients."""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

src = Path(__file__).resolve().parents[2] / "minigpt4/models/minigpt4rec_v2.py"
tree = ast.parse(src.read_text(encoding="utf-8"))
names = ("build_id_token_roles", "TargetConditionedKeyAdapter",
         "mismatch_candidate_features")
nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
         and node.name in names]
assert len(nodes) == 3
model = [node for node in tree.body
         if isinstance(node, ast.ClassDef) and node.name == "MiniGPT4Rec_v2"]
assert len(model) == 1
hook = [node for node in model[0].body if isinstance(node, ast.FunctionDef)
        and node.name == "_target_key_projection_hook"]
context = [node for node in model[0].body if isinstance(node, ast.FunctionDef)
           and node.name == "_make_target_v_context"]
assert len(hook) == len(context) == 1
env = {"nn": nn, "torch": torch}
exec(compile(ast.Module(body=nodes + hook + context, type_ignores=[]), str(src), "exec"), env)

roles = env["build_id_token_roles"](
    torch.tensor([[0, 0], [0, 1], [0, 3], [0, 4]]),
    [1, 0, 2], 3)
assert roles.tolist() == [1, 1, 0, 2], roles
other = env["build_id_token_roles"](
    torch.tensor([[0, 0], [0, 1], [0, 2]]), [2, 0, 1], 3)
assert other.tolist() == [2, 0, 1], other
print("[KEY-CPU] exact segment roles under prompt reordering PASS", flush=True)

Gate = env["TargetConditionedKeyAdapter"]
torch.manual_seed(42)
global_rng = torch.get_rng_state().clone()
key0 = Gate(output_dim=16, graph_size=256, middle=8)
assert torch.equal(global_rng, torch.get_rng_state())
key1 = Gate(output_dim=16, graph_size=256, middle=8)
assert torch.equal(global_rng, torch.get_rng_state())
assert set(key0.state_dict()) == set(key1.state_dict())
assert len(key0.state_dict()) == 4

class FakeModel:
    target_key_scope = "history"
    target_key_adapter = key0
    _target_key_payload = None
    collab_mode = "target_k_candidate"
    num_users = 10
    _target_key_scope_empty_batches = 0
    _target_key_scope_used_batches = 0
    training = True

fake = FakeModel()
b = torch.tensor([0, 0, 0, 1, 1, 1])
pos = torch.tensor([1, 2, 3, 1, 2, 3])
role_ids = torch.tensor([0, 1, 2, 0, 1, 2])
ctx = torch.randn(6, 1024)
fake._target_key_payload = (ctx, b, pos, role_ids)
original = torch.randn(2, 6, 16)
hook_fn = env["_target_key_projection_hook"]
modified = hook_fn(fake, None, None, original)
assert torch.equal(modified, original), "K adapter must preserve S2 at zero-init"
loss = modified.square().sum()
loss.backward()
grad = key0.network[-1].weight.grad
assert grad is not None and grad.abs().sum() > 0 and torch.isfinite(grad).all()
assert key0.network[0].weight.grad is not None, "Autograd graph dropped"
print("[KEY-CPU] zero-init, RNG isolation, nonzero Key gradient PASS", flush=True)

# After a controlled output projector change, only history K slots move.
with torch.no_grad():
    key0.network[-1].weight.normal_(std=0.01)
new_output = hook_fn(fake, None, None, original)
changed = (new_output - original).abs().sum(-1) > 0
expected = torch.zeros(2, 6, dtype=torch.bool)
expected[0, 2] = True
expected[1, 2] = True
assert torch.equal(changed, expected), (changed, expected)
fake._target_key_payload = None
assert torch.equal(hook_fn(fake, None, None, original), original)
print("[KEY-CPU] K correction affects only requested history token positions PASS", flush=True)

# Full context must mask candidate in K0, preserve in K1.
nodes = torch.tensor([0, 1, 10, 11, 12], dtype=torch.long)
g = torch.arange(5*256, dtype=torch.float32).reshape(5, 256) * 1e-3
fb = {"sub_node_ids": nodes, "gnn_penult": g,
      "batch_indices": torch.tensor([0, 0, 1, 1]),
      "local_node_ids": torch.tensor([0, 3, 1, 4])}
samples = {"UserID": torch.tensor([0, 1]),
           "TargetItemID": torch.tensor([1, 2])}
ctx_func = env["_make_target_v_context"]
fake.collab_mode = "target_k_candidate"
full = ctx_func(fake, fb, samples)
fake.collab_mode = "target_k_no_candidate"
masked = ctx_func(fake, fb, samples)
fu, ft, fn, fp = full.split(256, -1)
mu, mt, mn, mp = masked.split(256, -1)
assert torch.equal(fu, mu) and torch.equal(fn, mn)
assert ft.abs().sum() > 0 and fp.abs().sum() > 0
assert mt.count_nonzero() == mp.count_nonzero() == 0
print("[KEY-CPU] K0/K1 candidate masking is isolated PASS", flush=True)

# KS: preserve candidates in prompt/GNN; replace only Key context and require
# wrong target IDs. In-batch all-same/size-one fallback also must mismatch.
negative = env["mismatch_candidate_features"]
idlist = torch.tensor([11, 12, 11])
features = torch.tensor([[1., 2.], [3., 4.], [1., 2.]])
nodes = torch.tensor([0, 21, 22, 23])
penult = torch.tensor([[0., 0.], [1., 2.], [3., 4.], [5., 6.]])
mismatch, fallback = negative(idlist, features, nodes, penult, 10)
assert fallback == 0
assert torch.equal(mismatch, torch.tensor([[3., 4.], [1., 2.], [3., 4.]]))
assert torch.equal(idlist, torch.tensor([11, 12, 11]))
single, fb_count = negative(torch.tensor([11]), features[:1], nodes, penult, 10)
assert fb_count == 1 and not torch.equal(single[0], features[0])
print("[KEY-CPU] KS candidate mismatch and single-target fallback PASS", flush=True)

# History+target scope: user Key unchanged; the other two groups change.
fake.target_key_scope = "history_target"
fake._target_key_payload = (ctx, b, pos, role_ids)
joined = hook_fn(fake, None, None, original)
joined_changed = (joined - original).abs().sum(-1) > 0
expected_joined = torch.zeros(2, 6, dtype=torch.bool)
expected_joined[:, 2:4] = True
assert torch.equal(joined_changed, expected_joined), (joined_changed, expected_joined)
print("[KEY-CPU] history_target positions only PASS", flush=True)

# Reproduce H0 first 24 batches succeeding but a later batch with no
# history tokens failing backward because Key was the ONLY trainable module.
# Returning original K tensor directly would leave loss.requires_grad=False.
fake.target_key_scope = "history"
fake._target_key_payload = (ctx, b, pos, torch.tensor([0, 2, 2, 0, 2, 2]))
key0.zero_grad(set_to_none=True)
no_history_result = hook_fn(fake, None, None, original)
assert torch.equal(no_history_result, original), "Empty scope changed S2 K values"
assert no_history_result.requires_grad, "Empty scope detached the only trainable path"
no_history_result.square().sum().backward()
out_weight_grad = key0.network[-1].weight.grad
assert out_weight_grad is not None
assert torch.count_nonzero(out_weight_grad) == 0, "Empty scope must not learn from non-matching samples"
assert fake._target_key_scope_empty_batches == 1
print("[KEY-CPU] empty history batch: S2 identity, backward succeeds, zero Key gradient PASS", flush=True)

