#!/usr/bin/env python3
"""Audit D1's *actual* Yes logits on one identical Valid batch, NO training.

Loads a trained N2 or N3 checkpoint. Holds all data, model, random states,
and network parameters fixed while switching ONLY decision-Q injection:
  1) OFF  (ID-V only)
  2) ON   (trained decision-Q projection, as used in original test)
  3) 20x  (sensitivity control, NOT an experiment/Test metric).

Run: CUDA_VISIBLE_DEVICES=0 python audit_D1_logits.py --ckpt <N2 complete checkpoint>
"""
import argparse
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank, init_distributed_mode
from minigpt4.common.logger import setup_logger
from minigpt4.common.utils import now
from minigpt4.datasets.data_utils import prepare_sample
from minigpt4.common.registry import registry
from minigpt4.datasets.builders import *  # noqa: F401,F403
from minigpt4.models import *  # noqa: F401,F403
from minigpt4.processors import *  # noqa: F401,F403
from minigpt4.runners import *  # noqa: F401,F403
from minigpt4.tasks import *  # noqa: F401,F403


def rng_states():
    return (
        random.getstate(), np.random.get_state(),
        torch.get_rng_state(), torch.cuda.get_rng_state_all(),
    )


def restore_rng(states):
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    torch.cuda.set_rng_state_all(states[3])


def delta_report(label, on, off):
    d = on - off
    print("[Q-PROBE]", label,
          "max_abs={:.9g} mean_abs={:.9g} exact_changed={}/{}".format(
              d.abs().max().item(), d.abs().mean().item(),
              int(torch.count_nonzero(d).item()), d.numel()
          ), flush=True)
    print("[Q-PROBE]", label, "per-sample-difference=", d.tolist(), flush=True)
    return float(d.abs().max().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg-path",
                    default="tran_configs/NIGHT4_N2_DECISION_Q_ID_V.yaml")
    ap.add_argument("--scale-check", type=float, default=20.0)
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()
    if args.scale_check <= 1:
        raise ValueError("Scale check must be greater than one")

    opts = [
        "model.ckpt=" + args.ckpt,
        "run.evaluate=true",
        "run.batch_size_eval=" + str(args.batch_size),
        "run.output_dir=output/D1_Q_probe",
    ]
    cfg = Config(SimpleNamespace(cfg_path=args.cfg_path, options=opts))
    init_distributed_mode(cfg.run_cfg)
    seed = cfg.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    setup_logger()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    root = cfg.datasets_cfg.amazon_ood.path
    df = [pd.read_pickle(root + x) for x in
          ("train_ood2.pkl", "valid_ood2.pkl", "test_ood2.pkl")]
    cfg.model_cfg.rec_config.user_num = int(max(d.uid.max() for d in df)) + 1
    cfg.model_cfg.rec_config.item_num = int(max(d.iid.max() for d in df)) + 1
    model = task.build_model(cfg)
    runner_class = registry.get_runner_class(
        cfg.run_cfg.get("runner", "rec_runner_base"))
    runner = runner_class(
        cfg=cfg, job_id=now(), task=task, model=model, datasets=datasets)
    runner.set_model_mode(cfg.run_cfg.mode)
    model = runner.model
    model.eval()
    if model.d1_mode not in ("decision_q_v", "decision_q_v_gated"):
        raise ValueError("Use an N2 or N3 config and matching checkpoint")
    qmods = [m for m in model.my_linears if m.module_tag == "q"]
    if len(qmods) != 1:
        raise RuntimeError("Expected one D1 decision q module")
    if not hasattr(model, "d1_pair_proj") or model.d1_pair_proj is None:
        raise RuntimeError("D1 projector missing")
    state = model.d1_pair_proj.weight.detach().clone()
    print("[Q-PROBE] checkpoint:", args.ckpt, flush=True)
    print("[Q-PROBE] mode:", model.d1_mode,
          "P_norm:", state.float().norm().item(), flush=True)
    if torch.count_nonzero(state).item() == 0:
        raise RuntimeError("D1 projector is zero, no trained Q signal")

    loader = runner.dataloaders["valid"]
    batch = prepare_sample(next(iter(loader)), cuda_enabled=True)
    first_rng = rng_states()
    original_mode = qmods[0].d1_mode
    try:
        with torch.no_grad():
            qmods[0].d1_mode = "v_only"
            restore_rng(first_rng)
            off = model.generate_for_samples_v2(batch)["logits"].detach().float().cpu()
            qmods[0].d1_mode = original_mode
            restore_rng(first_rng)
            on = model.generate_for_samples_v2(batch)["logits"].detach().float().cpu()
            with torch.no_grad():
                model.d1_pair_proj.weight.mul_(args.scale_check)
            restore_rng(first_rng)
            boosted = model.generate_for_samples_v2(batch)["logits"].detach().float().cpu()
    finally:
        qmods[0].d1_mode = original_mode
        with torch.no_grad():
            model.d1_pair_proj.weight.copy_(state)

    orig_diff = delta_report("original_on_minus_off", on, off)
    scale_diff = delta_report(
        "{}x_projection_minus_off".format(args.scale_check), boosted, off)
    if orig_diff == 0 and scale_diff > 0:
        print("[Q-PROBE] Diagnosis: original decision-Q path is "
              "numerically INAUDIBLE at Yes logit; scaled path is live.")
    elif scale_diff == 0:
        print("[Q-PROBE] Diagnosis: even scaled decision-Q has NO observable "
              "logit effect; inspect token position/path before any more training.")
    else:
        print("[Q-PROBE] Diagnosis: decision-Q affects logits. "
              "Match per-sample scores, not just 10-digit aggregate AUC.")
    print("[Q-PROBE] All model parameters restored, no optimizer steps.")


if __name__ == "__main__":
    main()
