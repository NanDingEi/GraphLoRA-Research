"""Smoke test for B2/B3 bidirectional mutual learning (LLM <-> GNN).

Builds MiniGPT4Rec_v2 from a mutual yaml (ckpt disabled for speed), then checks:
  1. config wiring: mutual_mode loaded, mutual_layer < gcn_layer, lambdas set
  2. B3 gradient direction: L_{L->G} backprop touches GNN/MF tables only;
     L_{G->L} backprop touches the LLM side (LoRA via pass-1) only
  3. lambda=0 equivalence: mutual_mode on with zero lambdas gives bit-identical
     train loss to mutual off (pass-1 skipped, RNG untouched)
  4. B3 KL sanity: identical relation distributions -> both KLs = 0
  5. B2 projectors: both directions give grads to their own projector only
  6. eval path unchanged: generate_for_samples logits identical with mutual on/off
Run on the server:  ~/anaconda3/envs/minigpt4/bin/python smoke_test_mutual.py \
    --cfg-path tran_configs/collm_pretrain_mf_ood_b3.yaml
"""
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.datasets.data_utils import prepare_sample

parser = argparse.ArgumentParser()
parser.add_argument("--cfg-path",
                    default="tran_configs/collm_pretrain_mf_ood_b3.yaml")
parser.add_argument("--options", nargs="+", default=None,
                    help="override some settings in the used config")
args = parser.parse_args()

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
cudnn.benchmark = False
cudnn.deterministic = True

cfg = Config(args)
cfg.model_cfg.ckpt = None  # skip the multi-GB warm-start load in this smoke test

task = tasks.setup_task(cfg)
datasets = task.build_datasets(cfg)

# same user_num/item_num injection as train_collm_mf_din.py
try:
    data_dir = cfg.datasets_cfg.movie_ood.path
except Exception:
    data_dir = cfg.datasets_cfg.amazon_ood.path
train_ = pd.read_pickle(data_dir + "train_ood2.pkl")
valid_ = pd.read_pickle(data_dir + "valid_ood2.pkl")
test_ = pd.read_pickle(data_dir + "test_ood2.pkl")
cfg.model_cfg.rec_config.user_num = int(max(train_.uid.max(), valid_.uid.max(), test_.uid.max()) + 1)
cfg.model_cfg.rec_config.item_num = int(max(train_.iid.max(), valid_.iid.max(), test_.iid.max()) + 1)

print("building model (vicuna fp16 load ~1-2 min) ...", flush=True)
model = task.build_model(cfg)
model.set_mode("v2")
model.to("cuda")

# ---- check 1: config wiring ----
assert model.mutual_mode in ("b2", "b3"), f"mutual_mode not loaded: {model.mutual_mode}"
assert model.mutual_layer < 31, "mutual_layer must be < gcn_layer"
assert model.lambda_llm_to_gnn > 0 and model.lambda_gnn_to_llm > 0
assert model.semantic_feedback is False
print(f"[1/6] config wiring OK (mode={model.mutual_mode}, "
      f"l2g={model.lambda_llm_to_gnn}, g2l={model.lambda_gnn_to_llm})")

# ---- grab a real batch (multi-sample to exercise per-sample grouping) ----
inner_datasets = list(datasets.values())[0]  # {split_name: dataset}
loader = DataLoader(inner_datasets["train"], batch_size=4, shuffle=False,
                    collate_fn=getattr(inner_datasets["train"], "collater", None))
samples = prepare_sample(next(iter(loader)), cuda_enabled=True)

# ---- capture fb (mutual views) from a real train forward ----
model.train()
fb_stash = {}
orig_fb = model.compute_feedback_gcn

def spy_fb(*a, **kw):
    out = orig_fb(*a, **kw)
    fb_stash.update({k: v for k, v in out.items() if k != "e_tilde" or v is not None})
    return out

model.compute_feedback_gcn = spy_fb
random.seed(0); torch.manual_seed(0)
out = model(samples)
model.compute_feedback_gcn = orig_fb
assert "h_mutual" in fb_stash and "token_gnn" in fb_stash, \
    f"mutual views missing from fb: {list(fb_stash.keys())}"
assert fb_stash["h_mutual"].shape[-1] == 4096 and fb_stash["token_gnn"].shape[-1] == 256
print(f"[fb] h_mutual {tuple(fb_stash['h_mutual'].shape)}, "
      f"token_gnn {tuple(fb_stash['token_gnn'].shape)}, "
      f"train loss={out['loss'].item():.4f}")

# helper: find one LoRA param on the pass-1 path (layers 0..7) and one gcn param
lora_param = None
for name, p in model.llama_model_lora.named_parameters():
    if "lora_A" in name and "layers.3" in name:
        lora_param = (name, p)
        break
gcn_param = ("gcn.W1.weight", model.gcn.layers[0].W1.weight)
mf_param = ("rec_encoder.item_embedding.weight", model.rec_encoder.item_embedding.weight)

# ---- check 2: B3 gradient direction ----
if model.mutual_mode == "b3":
    model.zero_grad(set_to_none=True)
    l2g, g2l, _conf = model.compute_mutual_losses(fb_stash)
    l2g.backward(retain_graph=True)
    assert gcn_param[1].grad is not None and torch.isfinite(gcn_param[1].grad).all(), \
        "L_L->G must reach the GNN"
    assert mf_param[1].grad is not None, "L_L->G must reach the MF tables (GNN input features)"
    assert lora_param[1].grad is None, "L_L->G must NOT touch the LLM side (target detached)"
    assert torch.isfinite(out["loss"]).item()

    model.zero_grad(set_to_none=True)
    g2l.backward()
    assert lora_param[1].grad is not None and torch.isfinite(lora_param[1].grad).all(), \
        "L_G->L must reach the LLM side (pass-1 LoRA)"
    assert gcn_param[1].grad is None, "L_G->L must NOT touch the GNN side (target detached)"
    print("[2/6] B3 gradient direction OK (L->G -> GNN/MF only, G->L -> LoRA only)")
else:
    print("[2/6] skipped (B2 config)")

# ---- check 3: lambda=0 equivalence with mutual off ----
model.zero_grad(set_to_none=True)
model.lambda_llm_to_gnn = 0.0
model.lambda_gnn_to_llm = 0.0
random.seed(1); torch.manual_seed(1)
loss_zero = model(samples)["loss"].detach()
model.lambda_llm_to_gnn = float(cfg.model_cfg.lambda_llm_to_gnn)
model.lambda_gnn_to_llm = float(cfg.model_cfg.lambda_gnn_to_llm)

model.mutual_mode = "off"
random.seed(1); torch.manual_seed(1)
loss_off = model(samples)["loss"].detach()
model.mutual_mode = cfg.model_cfg.mutual_mode

diff = (loss_zero - loss_off).abs().item()
assert diff < 1e-6, f"lambda=0 loss differs from mutual off: {diff}"
print(f"[3/6] lambda=0 equivalence OK (diff={diff:.2e})")

# ---- check 4: B3 sanity on synthetic data ----
if model.mutual_mode == "b3":
    NU = model.num_users  # item ids must be >= num_users to be seen as items
    bi = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device="cuda")
    nid = torch.tensor([0, NU, NU + 1, NU + 2, 1, NU + 3, NU + 4, NU + 5], device="cuda")

    if model.mutual_rank_loss:
        # ranking hinge: craft an orthonormal-ish toy space
        # user0 (token 0) = e0; its items: token1 ~ e0 (cos 1), tokens 2,3 = e1 (cos 0)
        # user1 (token 4) = e2-ish; its items: token5 aligned (cos 1), tokens 6,7 orthogonal
        h = torch.zeros(8, 4, device="cuda")
        h[0, 0] = 1.0
        h[1, 0] = 1.0; h[1, 1] = 0.1
        h[2, 1] = 1.0; h[3, 1] = 1.0
        h[4, 2] = 1.0
        h[5, 2] = 1.0; h[5, 3] = 0.1
        h[6, 3] = 1.0; h[7, 3] = 1.0
        # (a) teacher decisive, student WRONG order -> positive loss, grad to student only
        z = h.clone()
        z[0] = h[2]; z[2] = h[0]  # GNN view of user0 now prefers token2 over token1
        l2g_u, g2l_u = model._b3_ranking_losses(h, z, nid, bi)
        assert l2g_u.mean().item() > 1e-4, f"wrong student order must be penalized: {l2g_u.mean().item()}"
        model.zero_grad(set_to_none=True)
        h2 = h.clone().requires_grad_(True); z2 = z.clone().requires_grad_(True)
        l2g2, _ = model._b3_ranking_losses(h2, z2, nid, bi)
        l2g2.mean().backward(retain_graph=True)
        assert h2.grad is None and z2.grad is not None and z2.grad.abs().sum() > 0, \
            "ranking L->G must only touch the student (z) side"
        h2.grad = None; z2.grad = None
        _, g2l2 = model._b3_ranking_losses(h2, z2, nid, bi)
        g2l2.mean().backward()
        assert z2.grad is None and h2.grad is not None, \
            "ranking G->L must only touch the student (h) side"
        # (b) both views agree with margin (or teacher ties) -> loss EXACTLY 0, no grad
        z3 = h.clone().requires_grad_(True)
        l2g3, g2l3 = model._b3_ranking_losses(h, z3, nid, bi)
        assert l2g3.mean().item() == 0.0 and g2l3.mean().item() == 0.0, \
            f"agreed views must give zero hinge: {l2g3.mean().item()}, {g2l3.mean().item()}"
        l2g3.mean().backward()
        assert z3.grad is None or z3.grad.abs().sum().item() == 0.0, \
            "agreed views must exert zero gradient (self-terminating)"
        print("[4/6] B3 ranking hinge OK (wrong order penalized, agreed views self-terminate)")
    else:
        # KL mode regression: identical views -> KL = 0
        h = torch.randn(8, 64, device="cuda", requires_grad=True)
        z = h.clone()
        l2g, g2l = model._b3_losses(h, z, nid, bi)
        assert l2g.mean().item() < 1e-5 and g2l.mean().item() < 1e-5
        l2g.mean().backward(retain_graph=True)
        assert h.grad is None and z.grad is not None
        print("[4/6] B3 KL sanity OK")
else:
    print("[4/6] skipped (B2 config)")

# ---- check 5: B2 projector gradient direction (projectors created on the fly
# for testing regardless of which yaml was loaded) ----
model.mutual_mode = "b2"
if not hasattr(model, "mutual_proj_L"):
    model.mutual_proj_L = nn.Linear(4096, 256).cuda()
    model.mutual_proj_G = nn.Linear(256, 256).cuda()
    model._mutual_stats = {"n": 0, "l2g": 0.0, "g2l": 0.0}
# capture a fresh fb: the graph captured in check 2 was freed by backward
model.train()
model.compute_feedback_gcn = spy_fb
random.seed(0); torch.manual_seed(0)
model(samples)
model.compute_feedback_gcn = orig_fb
model.zero_grad(set_to_none=True)
fb2 = dict(fb_stash)
assert "h_mutual" in fb2 and "token_gnn" in fb2
l2g, g2l, _conf = model.compute_mutual_losses(fb2)
l2g.backward(retain_graph=True)
assert model.mutual_proj_G.weight.grad is not None, "L_L->G must train P_G"
assert model.mutual_proj_L.weight.grad is None, "L_L->G must not train P_L (target detached)"
model.zero_grad(set_to_none=True)
g2l.backward()
assert model.mutual_proj_L.weight.grad is not None, "L_G->L must train P_L"
assert model.mutual_proj_G.weight.grad is None, "L_G->L must not train P_G (target detached)"
model.mutual_mode = cfg.model_cfg.mutual_mode
print("[5/6] B2 projector gradient direction OK")

# ---- check 6: eval path unchanged ----
model.eval()
with torch.no_grad():
    model.compute_feedback_gcn = spy_fb
    fb_stash.clear()
    random.seed(2)
    out_on = model.generate_for_samples(samples)
    assert "h_mutual" not in fb_stash, \
        "eval forward must not run the mutual pass-1 (training-only)"
    model.compute_feedback_gcn = orig_fb

    model.mutual_mode = "off"
    random.seed(2)
    out_off = model.generate_for_samples(samples)
    model.mutual_mode = cfg.model_cfg.mutual_mode

logit_diff = (out_on["logits"] - out_off["logits"]).abs().max().item()
assert logit_diff < 1e-2, f"eval logits changed with mutual on: {logit_diff}"
print(f"[6/6] eval path unchanged OK (logit diff={logit_diff:.2e})")

print("\nALL SMOKE TESTS PASSED")
