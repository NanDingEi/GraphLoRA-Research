"""
Probe v2: localize the exact gradient cut between layer 7 and 8.

A) functional test: perturb lora_B at L0 / L7 / L8 and diff the logits —
   a forward-side disconnection shows up as IDENTICAL logits.
B) autograd.grad(loss -> captured layer-7 / layer-8 output tensors):
   if grad(L7 out) == 0 while grad(L8 out) != 0, the backward graph is cut
   exactly between them.
C) llama_proj grad (input-side, trainable): how far down does the cut reach?
"""
import argparse
import random

import torch
from torch.utils.data import DataLoader

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.models import *
from minigpt4.tasks import *


def build(cfg):
    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    import pandas as pd
    data_dir = cfg.datasets_cfg.amazon_ood.path
    tr = pd.read_pickle(data_dir + "train_ood2.pkl")
    va = pd.read_pickle(data_dir + "valid_ood2.pkl")
    te = pd.read_pickle(data_dir + "test_ood2.pkl")
    cfg.model_cfg.rec_config.user_num = int(max(tr.uid.max(), va.uid.max(), te.uid.max()) + 1)
    cfg.model_cfg.rec_config.item_num = int(max(tr.iid.max(), va.iid.max(), te.iid.max()) + 1)
    model = task.build_model(cfg).to(cfg.run_cfg.device)
    model.set_mode("v2")
    model.train()
    train_ds = [s["train"] for s in datasets.values() if "train" in s][0]
    loader = DataLoader(train_ds, batch_size=3, shuffle=False,
                        collate_fn=getattr(train_ds, "collater", None), num_workers=0)
    samples = next(iter(loader))
    samples = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in samples.items()}
    samples.update({"epoch": 0, "num_iters_per_epoch": 11297, "iters": 0})
    return model, samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg-path", default="tran_configs/collm_pretrain_mf_ood_rlm_eval_base.yaml")
    args = p.parse_args()
    random.seed(42)
    torch.manual_seed(42)
    cfg = Config(argparse.Namespace(cfg_path=args.cfg_path, options=None))
    model, samples = build(cfg)

    llama = model.llama_model_lora
    mods = dict(llama.named_modules())

    # ---- capture decoder-layer output tensors with a forward hook ----
    captured = {}
    hooks = []
    for li in (0, 6, 7, 8, 9):
        layer_mod = mods[f"base_model.model.model.layers.{li}"]
        def mk(li):
            def hook(m, inp, out):
                captured[li] = out[0] if isinstance(out, tuple) else out
            return hook
        hooks.append(layer_mod.register_forward_hook(mk(li)))

    def run_loss():
        model.zero_grad(set_to_none=True)
        ret = model.forward(samples)
        loss = ret["loss"]
        loss.backward(retain_graph=True)
        return loss.item(), ret

    loss0, ret0 = run_loss()
    print(f"loss = {loss0:.6f}")

    # B) autograd.grad into captured layer outputs
    for li in (0, 6, 7, 8, 9):
        t = captured[li]
        g = torch.autograd.grad(ret0["loss"], t, retain_graph=True, allow_unused=True)[0]
        print(f"grad(loss -> L{li} output): "
              f"{'UNUSED (graph cut above)' if g is None else f'{g.norm().item():.6e}'}")

    # C) input-side trainable grads
    gp = model.llama_proj
    gnorm = sum(p_.grad.norm().item() for p_ in gp.parameters() if p_.grad is not None)
    print(f"llama_proj total |grad| = {gnorm:.6e}")

    # A) functional perturbation: B += 0.05 at L0 / L7 / L8, diff logits
    for li in (0, 7, 8):
        mod = mods[f"base_model.model.model.layers.{li}.self_attn.q_proj"]
        B = mod.lora_B["default"].weight
        with torch.no_grad():
            B.add_(0.05)
        model.zero_grad(set_to_none=True)
        ret1 = model.forward(samples)
        model.zero_grad(set_to_none=True)
        d = (ret1["loss"] - ret0["loss"]).abs().item()
        dl = (ret1["logits"] - ret0["logits"]).abs().max().item() if "logits" in ret1 else float("nan")
        print(f"perturb B(L{li}) += 0.05 -> dloss={d:.3e}  dlogits_max={dl:.3e}")
        with torch.no_grad():
            B.sub_(0.05)


if __name__ == "__main__":
    main()
