#!/usr/bin/env python3
"""CPU regression for P0 L30 auxiliary BCE, with actual methods AST-loaded.

No transformers, data files, checkpoints or GPU. The result is structural
and gradient-path unit coverage, NOT a claim of successful 7B training.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "minigpt4/models/minigpt4rec_v2.py"
module = ast.parse(path.read_text(encoding="utf-8"))
cls = next(n for n in module.body if isinstance(n, ast.ClassDef)
           and n.name == "MiniGPT4Rec_v2")
methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)]
required = {"_l30_aux_capture", "_l30_auxiliary_bce"}
selected = [n for n in methods if n.name in required]
assert {n.name for n in selected} == required
scoped = ast.Module(body=selected, type_ignores=[])
ns = {"torch":torch,"F":F}
exec(compile(scoped,str(path),"exec"),ns)

class Mock:
    _l30_aux_capture = ns["_l30_aux_capture"]
    _l30_auxiliary_bce = ns["_l30_auxiliary_bce"]

    def __init__(self):
        norm = SimpleNamespace(
            weight=torch.nn.Parameter(torch.tensor([2., 3., 4.]),
                                      requires_grad=False),
            variance_epsilon=1e-6)
        head = SimpleNamespace(weight=torch.nn.Parameter(
            torch.tensor([[1., 0., -1.], [0., 1., 1.]]),
            requires_grad=False))
        self.llama_model = SimpleNamespace(
            model=SimpleNamespace(norm=norm),lm_head=head)
        self._l30_aux_active = False
        self._l30_aux_position = None
        self._l30_aux_hidden = None
        self._l30_aux_forward_calls = 0

m = Mock()
w = torch.nn.Parameter(torch.tensor([1.,-2.,3.]))
x = torch.stack([w,w*.5],dim=0).unsqueeze(1).repeat(1,4,1)
# P0 control/eval should NEVER capture, even if called through hook.
m._l30_aux_capture(None,(),(x,None))
assert m._l30_aux_hidden is None
# Actual decision position must be final PROMPT token, not last ANSWER token.
m._l30_aux_active = True
m._l30_aux_position = 2
m._l30_aux_capture(None,(),(x,None))
assert m._l30_aux_forward_calls == 1
assert torch.equal(m._l30_aux_hidden,x[:,2,:])
loss = m._l30_auxiliary_bce(
    m._l30_aux_hidden, 0, torch.tensor([1.,0.]))
loss.backward()
assert w.grad is not None and float(w.grad.abs().sum()) > 0
assert not m.llama_model.model.norm.weight.requires_grad
assert not m.llama_model.lm_head.weight.requires_grad
print("[L30-CPU] decision-token hook captures autograd, Yes BCE backprops PASS",
      flush=True)

m._l30_aux_hidden = None
m._l30_aux_capture(None,(),x)
assert m._l30_aux_hidden is not None
try:
    m._l30_aux_capture(None,(),x)
except RuntimeError as e:
    assert "more than once" in str(e)
else:
    raise AssertionError("Duplicate L30 capture should fail")
print("[L30-CPU] guard for double decoder calls PASS",flush=True)

# Most important regression: loss addition must PRESERVE auxiliary BCE even
# when S2 semantic alignment is enabled, rather than overwrite it.
forward = next(n for n in methods if n.name == "forward_v2")
source = ast.get_source_segment(path.read_text(encoding="utf-8"),forward)
assert "loss = loss + self.sem_corr_weight * loss_corr" in source
assert "loss = loss + self.gnn_semantic_weight * loss_sem" in source
assert "loss = loss + self.l30_aux_weight * aux_loss" in source
assert "self._l30_aux_active = active_l30" in source
assert "self._l30_aux_active = False" in source
# Confirm float-vs-int comparison regression: int(0.1)==0 must not
# incorrectly reject the original S2 gnn_scale=0.1.
assert 'abs(float(gnn_scale) - 0.1) > 1e-10' in path.read_text(encoding="utf-8")
assert 'int(gnn_scale) != 0.1' not in path.read_text(encoding="utf-8")
assert not (abs(float(0.1)-0.1)>1e-10)
print("[L30-CPU] S2 gnn_scale=0.1 accepted by float guard PASS",flush=True)

# from_config must pass aux kwargs, but old checkpoints omit aux config and
# use the default zero weight in __init__.
from_cfg = next(n for n in methods if n.name == "from_config")
cfg_source = ast.get_source_segment(path.read_text(encoding="utf-8"),from_cfg)
assert 'cfg.get("l30_aux_weight", 0.0)' in cfg_source
assert "l30_aux_weight=l30_aux_weight" in cfg_source
print("[L30-CPU] P0 off is backward-compatible; semantic and aux sum PASS",
      flush=True)
print("[L30-CPU] ALL PASS",flush=True)
