#!/usr/bin/env python3
"""One-minibatch gradient-path probe for the GraphLoRA semantic-teacher autocast bug.

Run each arm in its OWN process with exactly the same seed and first minibatch:
  CUDA_VISIBLE_DEVICES=0 python -u probe_graphlora_grad_paths.py \
    --cfg-path tran_configs/LAYER_P3_L30.yaml --fix-autocast-cache false
  CUDA_VISIBLE_DEVICES=0 python -u probe_graphlora_grad_paths.py \
    --cfg-path tran_configs/LAYER_P3_L30.yaml --fix-autocast-cache true

No optimizer steps, checkpoint updates, epoch training, or Test evaluation.
"""
import argparse
import json
import os
import random
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import init_distributed_mode, get_rank
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from minigpt4.datasets.data_utils import prepare_sample


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", required=True)
    parser.add_argument("--fix-autocast-cache", required=True, choices=("true", "false"))
    parser.add_argument("--batch-size", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    flag = args.fix_autocast_cache == "true"
    # Config consumes --cfg-path and --options; these are normal model overrides.
    cfg = Config(argparse.Namespace(
        cfg_path=args.cfg_path,
        options=["model.fix_autocast_cache=" + args.fix_autocast_cache],
    ))
    init_distributed_mode(cfg.run_cfg)
    seed = int(cfg.run_cfg.seed) + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    data_name = list(datasets.keys())[0]
    data_dir = cfg.datasets_cfg[data_name].path
    dframes = [pd.read_pickle(os.path.join(data_dir, s))
               for s in ("train_ood2.pkl", "valid_ood2.pkl", "test_ood2.pkl")]
    cfg.model_cfg.rec_config.user_num = int(max(x.uid.max() for x in dframes) + 1)
    cfg.model_cfg.rec_config.item_num = int(max(x.iid.max() for x in dframes) + 1)
    del dframes

    dset = datasets[data_name]["train"]
    loader = DataLoader(
        dset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=getattr(dset, "collater", None), drop_last=True,
    )
    model = task.build_model(cfg).to(torch.device("cuda"))
    # RunnerBase.train() calls self.set_model_mode(run.mode) before train_epoch().
    # This direct one-batch probe must mirror the same initialization.
    run_mode = str(cfg.run_cfg.mode)
    if run_mode != "v2":
        raise RuntimeError(f"GraphLoRA gradient probe expects run.mode=v2, got {run_mode!r}")
    model.set_mode(run_mode)
    assert model.run_mode_ == "v2", "Model run_mode_ was not initialized"
    model.train()
    batch = prepare_sample(next(iter(loader)), cuda_enabled=True)
    batch.update({"epoch": 0, "num_iters_per_epoch": len(loader), "iters": 0})
    print("[GRAD-PROBE] cfg={} fix_autocast_cache={} batch_size={} seed={}".format(
        args.cfg_path, flag, args.batch_size, seed), flush=True)
    print("[GRAD-PROBE] semantic_align={} teacher_layer={} preserve_bias={}".format(
        model.gnn_semantic_align, model._gnn_semantic_layer,
        model.preserve_source_qv_bias), flush=True)

    with torch.cuda.amp.autocast(enabled=True):
        loss = task.train_step(model, batch)
    scaler = torch.cuda.amp.GradScaler()
    scaler.scale(loss).backward()

    seen = {}
    # Probe LoRA B: unlike LoRA A, B can receive gradients at init (B=0).
    # Keep distinct layer and q/v names. None means autograd graph was severed.
    pattern = re.compile(r"\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)\.lora_B\.default\.weight$")
    for name, param in model.named_parameters():
        m = pattern.search(name)
        if m:
            key = (int(m.group(1)), m.group(2))
            g = param.grad
            seen[key] = {
                "name": name,
                "grad_is_none": g is None,
                "grad_norm_scaled": None if g is None else float(g.float().norm().item()),
            }

    layers = [0, 1, 7, 8, 9, 16, 24, 29, 30, 31]
    for layer in layers:
        for tag in ("q_proj", "v_proj"):
            if (layer, tag) not in seen:
                raise RuntimeError("Missing LoRA B weight for layer {} {}".format(layer, tag))
            r = seen[(layer, tag)]
            print("[GRAD-PROBE] layer={} module={} grad_none={} scaled_norm={}".format(
                layer, tag, r["grad_is_none"], r["grad_norm_scaled"]), flush=True)

    missing_early = [key for key in [(0, "q_proj"), (0, "v_proj"),
                                    (7, "q_proj"), (7, "v_proj")]
                     if seen[key]["grad_is_none"]]
    missing_late = [key for key in [(8, "q_proj"), (8, "v_proj"),
                                   (31, "q_proj"), (31, "v_proj")]
                    if seen[key]["grad_is_none"]]
    print("[GRAD-PROBE-SUMMARY] {}".format(json.dumps({
        "fix_autocast_cache": flag,
        "batch_size": args.batch_size,
        "loss": float(loss.detach().item()),
        "missing_early_B": [str(x) for x in missing_early],
        "missing_late_B": [str(x) for x in missing_late],
    })), flush=True)
    if missing_late:
        raise RuntimeError("Late-layer gradients unexpectedly missing: {}".format(missing_late))
    if flag and missing_early:
        raise RuntimeError("Cache FIX failed; early LoRA B grad still missing: {}".format(missing_early))


if __name__ == "__main__":
    main()
