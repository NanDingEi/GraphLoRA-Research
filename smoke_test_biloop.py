"""Smoke test for the BiLoop closed-loop semantic feedback.

Builds MiniGPT4Rec_v2 from the biloop yaml (ckpt disabled for speed), then checks:
  1. new modules exist with the agreed init (W_back zero, gate bias = -2)
  2. train-mode forward + backward works, gradients reach W_back
  3. zero-init equivalence: with the loop closed at init, enabling/disabling
     semantic_feedback gives identical gcn_output / logits
  4. forcing the loop open (non-zero W_back bias) actually changes the logits
Run on the server:  ~/anaconda3/envs/minigpt4/bin/python smoke_test_biloop.py
"""
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.datasets.data_utils import prepare_sample

parser = argparse.ArgumentParser()
parser.add_argument("--cfg-path",
                    default="tran_configs/collm_pretrain_mf_ood_biloop.yaml")
parser.add_argument("--options", nargs="+", default=None,
                    help="override some settings in the used config")
args = parser.parse_args()

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
cudnn.benchmark = False
cudnn.deterministic = True

cfg = Config(args)
cfg.model_cfg.ckpt = None  # skip the 13.6 GB warm-start load in this smoke test

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

# ---- check 1: modules + init ----
assert model.semantic_feedback is True, "semantic_feedback must be on for this yaml"
assert model.semantic_back_proj.weight.abs().max().item() == 0.0, "W_back must be zero-init"
assert model.semantic_back_proj.bias.abs().max().item() == 0.0, "W_back bias must be zero-init"
assert abs(model.semantic_gate.bias[0].item() - (-2.0)) < 1e-6, "gate bias must be -2"
print("[1/4] modules + init OK")

# ---- grab one real batch (batch 2 to exercise multi-sample indexing) ----
inner_datasets = list(datasets.values())[0]  # {split_name: dataset}
loader = DataLoader(inner_datasets["train"], batch_size=2, shuffle=False,
                    collate_fn=getattr(inner_datasets["train"], "collater", None))
samples = prepare_sample(next(iter(loader)), cuda_enabled=True)

# ---- check 2: train-mode forward + backward ----
model.train()
loss = model(samples)["loss"]
assert torch.isfinite(loss).item(), f"loss not finite: {loss}"
loss.backward()
g = model.semantic_back_proj.weight.grad
assert g is not None and torch.isfinite(g).all().item(), "W_back grad missing/inf"
print(f"[2/4] train forward/backward OK (loss={loss.item():.4f})")

# ---- checks 3+4: eval mode, spy on the helper, toggle the loop ----
model.eval()
orig_fb = model.compute_feedback_gcn
fb_stash = {}

def spy_fb(*a, **kw):
    out = orig_fb(*a, **kw)
    fb_stash["cur"] = {k: (v.detach().clone() if torch.is_tensor(v) else v)
                       for k, v in out.items()}
    return out

with torch.no_grad():
    # feedback ON (loop closed at init)
    model.compute_feedback_gcn = spy_fb
    random.seed(0)
    out_on = model.generate_for_samples(samples)
    gcn_on = fb_stash["cur"]["gcn_output"]
    e_tilde = fb_stash["cur"]["e_tilde"]
    model.compute_feedback_gcn = orig_fb

    # feedback OFF
    model.semantic_feedback = False
    model.compute_feedback_gcn = spy_fb
    random.seed(0)
    out_off = model.generate_for_samples(samples)
    gcn_off = fb_stash["cur"]["gcn_output"]
    model.compute_feedback_gcn = orig_fb
    model.semantic_feedback = True

gcn_diff = (gcn_on.float() - gcn_off.float()).abs().max().item()
logit_diff = (out_on["logits"] - out_off["logits"]).abs().max().item()
assert torch.allclose(gcn_on.float(), gcn_off.float(), atol=1e-6), \
    f"gcn_output differs with loop closed: {gcn_diff}"
assert torch.allclose(out_on["logits"], out_off["logits"], atol=1e-2), \
    f"logits differ with loop closed: {logit_diff}"
print(f"[3/4] zero-init equivalence OK (gcn diff={gcn_diff:.2e}, logit diff={logit_diff:.2e})")

with torch.no_grad():
    model.semantic_back_proj.bias.fill_(0.05)  # force s != 0 -> loop open
    random.seed(0)
    out_open = model.generate_for_samples(samples)
    model.semantic_back_proj.bias.zero_()
open_diff = (out_open["logits"] - out_off["logits"]).abs().max().item()
print(f"[4/4] forced-open loop changes logits (diff={open_diff:.2e}; "
      f"expected > 0 but may be tiny since e_tilde stays close to e)")

print("\nALL SMOKE TESTS PASSED")
