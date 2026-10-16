#!/usr/bin/env python3
"""Audit D1's *actual* Yes logits on identical Valid batches, NO training.

Loads a trained N2 or N3 checkpoint. Holds all data, model, random states,
and network parameters fixed while switching ONLY decision-Q injection:
  1) OFF  (ID-V only)
  2) ON   (trained decision-Q projection, as used in original test)
  3) Scales 1x, 10x, 20x, 50x (sensitivity checks, NOT Test metrics).

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
    ap.add_argument("--scale-check", type=float, default=20.0,
                    help="Legacy single amplified scale (default 20x)")
    ap.add_argument("--scales", default=None,
                    help="Comma-separated evaluation multipliers, e.g. 1,10,20,50")
    ap.add_argument("--num-batches", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()
    if args.num_batches < 1:
        raise ValueError("num-batches must be positive")
    scales = ([1.0, args.scale_check] if args.scales is None else
              [float(x.strip()) for x in args.scales.split(",")])
    if not scales or any(not np.isfinite(x) or x <= 0 for x in scales):
        raise ValueError("Every multiplier must be finite and positive")
    scales = list(dict.fromkeys(scales))

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

    # Runner's split loader is MultiIterLoader, which defines __next__
    # but NOT __iter__. Its .loaders entries are PrefetchLoader instances,
    # exactly the same objects traversed by RecBaseTask.evaluation().
    multi_loader = runner.dataloaders["valid"]
    if not hasattr(multi_loader, "loaders") or len(multi_loader.loaders) != 1:
        raise RuntimeError("Expected one underlying Valid PrefetchLoader")
    valid_iter = iter(multi_loader.loaders[0])
    original_mode = qmods[0].d1_mode
    stats = {scale: {"n": 0, "changed": 0, "sum_abs": 0.0,
                     "max_abs": 0.0, "signed_sum": 0.0}
             for scale in scales}
    batches_done = 0
    try:
        with torch.no_grad():
            for batch_num in range(args.num_batches):
                try:
                    raw_batch = next(valid_iter)
                except StopIteration:
                    break
                batch = prepare_sample(raw_batch, cuda_enabled=True)
                # All branch comparisons use the *same* batch and identical
                # RNG state. The model and optimizer are never updated.
                first_rng = rng_states()
                qmods[0].d1_mode = "v_only"
                restore_rng(first_rng)
                off = model.generate_for_samples_v2(batch)["logits"].detach().float().cpu()
                qmods[0].d1_mode = original_mode
                for scale in scales:
                    model.d1_pair_proj.weight.copy_(state * scale)
                    restore_rng(first_rng)
                    on = model.generate_for_samples_v2(batch)["logits"].detach().float().cpu()
                    diff = on - off
                    absdiff = diff.abs()
                    st = stats[scale]
                    st["n"] += int(diff.numel())
                    st["changed"] += int(torch.count_nonzero(diff).item())
                    st["sum_abs"] += float(absdiff.sum().item())
                    st["signed_sum"] += float(diff.sum().item())
                    st["max_abs"] = max(st["max_abs"], float(absdiff.max().item()))
                    if args.num_batches == 1:
                        delta_report(
                            ("original_on_minus_off" if scale == 1
                             else "{}x_projection_minus_off".format(scale)),
                            on, off)
                batches_done += 1
    finally:
        qmods[0].d1_mode = original_mode
        with torch.no_grad():
            model.d1_pair_proj.weight.copy_(state)

    print("[Q-PROBE] batches={} samples={} scales={}".format(
        batches_done, sum(s["n"] for s in stats.values()) // len(scales),
        scales), flush=True)
    for scale in scales:
        st = stats[scale]
        n = st["n"]
        if n == 0:
            raise RuntimeError("Valid loader returned no data")
        print("[Q-PROBE][aggregate] scale={:g} changed={}/{} rate={:.2%} "
              "mean_abs={:.9g} max_abs={:.9g} mean_signed={:.9g}".format(
                  scale, st["changed"], n, st["changed"] / n,
                  st["sum_abs"] / n, st["max_abs"], st["signed_sum"] / n),
              flush=True)
    if 1.0 in stats:
        base = stats[1.0]
        amplified = [(scale, stats[scale]) for scale in scales if scale > 1.0]
        if base["changed"] == 0 and any(x["changed"] > 0 for _, x in amplified):
            print("[Q-PROBE] Diagnosis: trained Q is numerically silent on "
                  "these samples, but amplification makes it observable.")
        elif base["changed"] == 0 and all(x["changed"] == 0 for _, x in amplified):
            print("[Q-PROBE] Diagnosis: no observable Q effect even at "
                  "tested amplification; inspect injection path.")
        else:
            print("[Q-PROBE] Diagnosis: trained Q affects some Yes logits. "
                  "Check rate and magnitude before designing new training.")
    print("[Q-PROBE] All parameters restored; no optimizer steps.")


if __name__ == "__main__":
    main()
