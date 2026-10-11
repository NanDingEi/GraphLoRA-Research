"""
Gradient probe: why do layers 0-7's LoRA adapters receive exactly-zero
gradients in the champion while layers >= 8 do not?

Loads the champion (eval-style config with ckpt), runs ONE training-mode
forward/backward on a real batch, then prints grad norms of lora_A/lora_B
for every decoder layer (q/v). Pure diagnosis, no training.
"""
import argparse
import random

import torch
from torch.utils.data import DataLoader

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.models import *
from minigpt4.tasks import *


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg-path", default="tran_configs/collm_pretrain_mf_ood_rlm_eval_base.yaml")
    args = p.parse_args()
    ns = argparse.Namespace(cfg_path=args.cfg_path, options=None)
    cfg = Config(ns)

    random.seed(42)
    torch.manual_seed(42)

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    import pandas as pd
    data_dir = cfg.datasets_cfg.amazon_ood.path
    tr = pd.read_pickle(data_dir + "train_ood2.pkl")
    va = pd.read_pickle(data_dir + "valid_ood2.pkl")
    te = pd.read_pickle(data_dir + "test_ood2.pkl")
    cfg.model_cfg.rec_config.user_num = int(max(tr.uid.max(), va.uid.max(), te.uid.max()) + 1)
    cfg.model_cfg.rec_config.item_num = int(max(tr.iid.max(), va.iid.max(), te.iid.max()) + 1)

    model = task.build_model(cfg)
    model = model.to(cfg.run_cfg.device)
    model.set_mode("v2")
    # training mode so LoRA dropout is active exactly as in real training;
    # gradient FACTS (zero vs nonzero) are what we are probing
    model.train()

    train_ds = None
    for name, split in datasets.items():
        if "train" in split:
            train_ds = split["train"]
    loader = DataLoader(train_ds, batch_size=3, shuffle=False,
                        collate_fn=getattr(train_ds, "collater", None), num_workers=0)
    samples = next(iter(loader))
    samples = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in samples.items()}
    samples.update({"epoch": 0, "num_iters_per_epoch": 11297, "iters": 0})

    ret = model.forward(samples)
    ret["loss"].backward()

    llama = model.llama_model_lora
    print(f"loss = {ret['loss'].item():.6f}")
    print(f"{'layer':>5s} {'q |dA|':>12s} {'q |dB|':>12s} {'v |dA|':>12s} {'v |dB|':>12s}")
    for layer in range(32):
        row = []
        for proj in ("q_proj", "v_proj"):
            mod = dict(llama.named_modules()).get(
                f"base_model.model.model.layers.{layer}.self_attn.{proj}")
            if mod is None or not hasattr(mod, "lora_A"):
                row += [float("nan"), float("nan")]
                continue
            for branch in ("A", "B"):
                w = getattr(mod, f"lora_{branch}")["default"].weight
                g = w.grad
                row.append(0.0 if g is None else g.norm().item())
        print(f"{layer:5d} " + " ".join(f"{x:12.3e}" for x in row))

    # also check requires_grad flags of a shallow vs deep adapter
    for layer in (0, 8, 31):
        mod = dict(llama.named_modules()).get(
            f"base_model.model.model.layers.{layer}.self_attn.q_proj")
        print(f"L{layer}: type={type(mod).__name__} "
              f"A.requires_grad={mod.lora_A['default'].weight.requires_grad} "
              f"B.requires_grad={mod.lora_B['default'].weight.requires_grad}")


if __name__ == "__main__":
    main()
