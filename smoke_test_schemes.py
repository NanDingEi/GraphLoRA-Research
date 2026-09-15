"""Smoke test for B9/B10/B11 schemes on top of the mutual-learning model.

  B11: disagreement/threshold gating of the mutual losses + GNN confidence head
  B9 : independent collaborative LoRA expert (zero-init) + per-token router,
       injected via decoder-layer forward hooks
  B10: multi-layer candidate adapters + layer router alpha (softmax)

Run:  ~/anaconda3/envs/minigpt4/bin/python smoke_test_schemes.py \
        --cfg-path tran_configs/collm_pretrain_mf_ood_b3_expert_ml.yaml
"""
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.datasets.data_utils import prepare_sample

# ---- check 0: REL losses standalone (CPU, no model needed) ----
import minigpt4.models.minigpt4rec_v2 as _mmod
_h = torch.zeros(6, 3)
_h[0, 0] = 1; _h[1, 0] = 1; _h[1, 1] = 0.1
_h[2, 1] = 1; _h[3, 1] = 1; _h[4, 2] = 1; _h[5, 2] = 1
_rows = torch.arange(6)
_z = _h.clone()
_s2g, _g2l = _mmod.MiniGPT4Rec_v2._rel_losses(None, _h, _z, _rows)
assert _s2g.item() < 1e-6 and _g2l.item() < 1e-6, "identical views must give zero relation MSE"
_h2 = torch.randn(6, 3, requires_grad=True)
_z2 = torch.randn(6, 3, requires_grad=True)
_s2g2, _g2l2 = _mmod.MiniGPT4Rec_v2._rel_losses(None, _h2, _z2, _rows)
assert _s2g2.item() > 1e-6, "different views must give a positive relation MSE"
_s2g2.backward(retain_graph=True)
assert _h2.grad is None and _z2.grad is not None and _z2.grad.abs().sum() > 0, \
    "L_S2G must only touch the GNN side"
_h2.grad = None; _z2.grad = None
_g2l2.backward()
assert _z2.grad is None and _h2.grad is not None and _h2.grad.abs().sum() > 0, \
    "L_G2L must only touch the LLM side"
_w0 = torch.zeros(6)
_s2g_w0, _ = _mmod.MiniGPT4Rec_v2._rel_losses(None, _h, _z, _rows, w_l_item=_w0)
assert _s2g_w0.item() == 0.0, "zero confidence weights must zero the weighted loss"
_s2g_d, _ = _mmod.MiniGPT4Rec_v2._rel_losses(None, _h, _z, _rows[:1])
print(f"[0/6] REL losses standalone OK (identical->0, grad directions, weighting, degenerate K=1={_s2g_d.item():.2e})")

parser = argparse.ArgumentParser()
parser.add_argument("--cfg-path",
                    default="tran_configs/collm_pretrain_mf_ood_b3_expert_ml.yaml")
parser.add_argument("--options", nargs="+", default=None)
args = parser.parse_args()

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
cudnn.benchmark = False
cudnn.deterministic = True

cfg = Config(args)
cfg.model_cfg.ckpt = None  # skip warm-start load

task = tasks.setup_task(cfg)
datasets = task.build_datasets(cfg)

try:
    data_dir = cfg.datasets_cfg.movie_ood.path
except Exception:
    data_dir = cfg.datasets_cfg.amazon_ood.path
train_ = pd.read_pickle(data_dir + "train_ood2.pkl")
valid_ = pd.read_pickle(data_dir + "valid_ood2.pkl")
test_ = pd.read_pickle(data_dir + "test_ood2.pkl")
cfg.model_cfg.rec_config.user_num = int(max(train_.uid.max(), valid_.uid.max(), test_.uid.max()) + 1)
cfg.model_cfg.rec_config.item_num = int(max(train_.iid.max(), valid_.iid.max(), test_.iid.max()) + 1)

print("building model ...", flush=True)
model = task.build_model(cfg)
model.set_mode("v2")
model.to("cuda")

layers = model.injection_layers
print(f"[cfg] injection_mode={model.injection_mode}, layers={layers}, "
      f"gate={model.mutual_gate}, entropy_w={model.router_entropy_weight}")

# ---- grab a real batch + capture fb (views) from a train forward ----
inner = list(datasets.values())[0]
loader = DataLoader(inner["train"], batch_size=2, shuffle=False,
                    collate_fn=getattr(inner["train"], "collater", None))
samples = prepare_sample(next(iter(loader)), cuda_enabled=True)

hook_calls = []
orig_hooks = model._make_injection_hook

def counting_hook_factory(slot):
    base = orig_hooks(slot)
    def hook(module, inputs, output):
        if model._injection_on:  # count only real injections, not guarded no-ops
            hook_calls.append(slot)
        return base(module, inputs, output)
    return hook

# re-register counting hooks
for h in model._inj_handles:
    h.remove()
model._inj_handles = [
    model.llama_model_lora.base_model.model.model.layers[l].register_forward_hook(
        counting_hook_factory(slot))
    for slot, l in enumerate(layers)
]

model.train()
fb_stash = {}
orig_fb = model.compute_feedback_gcn

def spy_fb(*a, **kw):
    out = orig_fb(*a, **kw)
    fb_stash.update({k: v for k, v in out.items() if v is not None})
    return out

model.compute_feedback_gcn = spy_fb
random.seed(0); torch.manual_seed(0)
out = model(samples)
model.compute_feedback_gcn = orig_fb

# ---- check 1: hooks fire once per injection layer, only in the main forward ----
if model.injection_mode == "expert" and model._mutual_active():
    # pass-1 runs layers 0..mutual_layer-1; an unguarded layer hook would fire there
    from collections import Counter
    c = Counter(hook_calls)
    expected = {s: 1 for s in range(len(layers))}
    assert dict(c) == expected, f"hook calls wrong: {dict(c)} (pass-1 leak?)"
    print(f"[1/6] hooks fire exactly once per layer {layers} (pass-1 guarded) OK")
else:
    print(f"[1/6] skipped (mode={model.injection_mode}, mutual={model._mutual_active()})")

# ---- check 2: expert_scale=0 equivalence (fp16-safe injection design) ----
if model.injection_mode == "expert":
    scale_bak = model.expert_scale.data.clone()
    model.expert_scale.data.fill_(0.0)   # add == 0 -> injection is a no-op
    for h in model._inj_handles:
        h.remove()
    model.zero_grad(set_to_none=True)
    random.seed(0); torch.manual_seed(0)
    out_off = model(samples)
    diff = (out["logits"] - out_off["logits"]).abs().max().item()
    assert diff < 1e-6, f"scale=0 expert must be a no-op: loss diff {diff}"
    # re-register counting hooks for later checks
    model._inj_handles = [
        model.llama_model_lora.base_model.model.model.layers[l].register_forward_hook(
            counting_hook_factory(slot))
        for slot, l in enumerate(layers)
    ]
    print(f"[2/6] expert_scale=0 equivalence OK (loss diff={diff:.2e})")
else:
    print("[2/6] skipped (legacy mode)")

# ---- check 3: forced-open expert changes logits; router/adapter grads flow ----
if model.injection_mode == "expert":
    model.expert_scale.data.fill_(0.5)
    model.zero_grad(set_to_none=True)
    random.seed(0); torch.manual_seed(0)
    out_open = model(samples)
    # NOTE: assert on LOGITS, not loss — an untrained model's Yes-logit can be
    # saturated (+-30), making BCE insensitive to moderate hidden-state changes
    l_diff = (out_open["logits"] - out_off["logits"]).abs().max().item()
    assert l_diff > 1e-3, f"non-zero expert must change the logits (l_diff={l_diff:.2e})"
    out_open["loss"].backward()
    # grads: collab adapters receive loss grads; router too now that scale != 0
    assert model.collab_A[0].weight.grad is not None and torch.isfinite(model.collab_A[0].weight.grad).all()
    assert model.collab_B[0].weight.grad is not None
    assert model.inj_router[0].weight.grad is not None, "router must receive grads once scale != 0"
    assert model.layer_router.weight.grad is not None, "layer router must receive grads"
    # alpha is a softmax over candidate layers
    a = model._last_alpha
    assert a.shape[-1] == len(layers) and torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-5)
    print(f"[3/6] expert/router/alpha grads OK (forced-open loss diff={l_diff:.2e}, alpha[0]={a[0].tolist()})")
else:
    print("[3/6] skipped (legacy mode)")

# ---- check 4: B11 gating selects directions correctly ----
if model._mutual_active() and "h_mutual" in fb_stash and model.mutual_mode == "b3":
    # recapture a fresh fb (previous graph may be freed)
    model.train()
    model.compute_feedback_gcn = spy_fb
    random.seed(0); torch.manual_seed(0)
    model(samples)
    model.compute_feedback_gcn = orig_fb
    labels = samples["label"].float().to("cuda")

    # gate OFF: masked mean == plain mean of per-user losses
    model.zero_grad(set_to_none=True)
    gate_bak = model.mutual_gate
    model.mutual_gate = "off"
    l2g_off, g2l_off, conf_off = model.compute_mutual_losses(fb_stash, None, None)
    l2g_u, g2l_u = model._b3_ranking_losses(fb_stash["h_mutual"], fb_stash["token_gnn"],
                                            fb_stash["token_node_ids"], fb_stash["batch_indices"])
    assert abs(l2g_off.item() - l2g_u.mean().item()) < 1e-6, "gate off must equal plain mean"
    # gate ON, LLM hyper-confident: G->L must shut off completely
    model.mutual_gate = "threshold"
    l2g_a, g2l_a, conf_a = model.compute_mutual_losses(fb_stash,
                                                       torch.full((labels.numel(),), 10.0, device="cuda"),
                                                       labels)
    assert g2l_a.item() == 0.0, f"confident LLM must disable G->L: {g2l_a.item()}"
    # gate ON, LLM no-clue: L->G must shut off completely
    l2g_b, g2l_b, _ = model.compute_mutual_losses(fb_stash,
                                                  torch.full((labels.numel(),), -10.0, device="cuda"),
                                                  labels)
    assert l2g_b.item() == 0.0, f"clueless LLM must disable L->G: {l2g_b.item()}"
    # kappa confidence head receives grads from its own BCE
    if not hasattr(model, "gnn_conf_kappa"):
        model.gnn_conf_kappa = torch.nn.Parameter(torch.tensor(5.0, device="cuda"))
        s_g_re = model.gnn_conf_kappa * F.cosine_similarity(
            fb_stash["token_gnn"][:2], fb_stash["token_gnn"][:2], dim=-1)
        conf_a = F.binary_cross_entropy_with_logits(
            s_g_re, labels[:2].to(s_g_re.dtype))
    assert conf_a is not None
    (conf_a * 1.0).backward()
    assert model.gnn_conf_kappa.grad is not None and torch.isfinite(model.gnn_conf_kappa.grad).item()
    model.mutual_gate = gate_bak
    model.zero_grad(set_to_none=True)
    print("[4/6] B11 gating OK (confident LLM -> only L->G; clueless LLM -> only G->L; kappa grads)")
else:
    print("[4/6] skipped (mutual off or non-b3)")

# ---- check 5: eval path with expert injection runs and matches hooks-off ----
if model.injection_mode == "expert":
    model.expert_scale.data.copy_(scale_bak)   # restore init scale
    model.eval()
    with torch.no_grad():
        random.seed(2)
        out_eval_on = model.generate_for_samples(samples)
        hook_calls.clear()
        for h in model._inj_handles:
            h.remove()
        random.seed(2)
        out_eval_off = model.generate_for_samples(samples)
        d = (out_eval_on["logits"] - out_eval_off["logits"]).abs().max().item()
        assert d < 1e-2, f"eval logits must match with zero-init experts: {d}"
        model._inj_handles = [
            model.llama_model_lora.base_model.model.model.layers[l].register_forward_hook(
                counting_hook_factory(slot))
            for slot, l in enumerate(layers)
        ]
    print(f"[5/6] eval expert injection OK (zero-init logit diff={d:.2e})")
else:
    print("[5/6] skipped (legacy mode)")

# ---- check 6: full loss backward reaches everything ----
model.train()
model.zero_grad(set_to_none=True)
random.seed(3); torch.manual_seed(3)
out = model(samples)
out["loss"].backward()
g_lora = [p.grad for n, p in model.llama_model_lora.named_parameters()
          if "lora_A" in n and "layers.3" in n and p.grad is not None]
assert g_lora, "L_rec+mutual must reach LoRA"
if model.injection_mode == "expert":
    assert model.collab_B[0].weight.grad is not None
assert torch.isfinite(out["loss"]).item()
print("[6/6] full backward OK")

print("\nALL SMOKE TESTS PASSED")
