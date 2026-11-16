#!/usr/bin/env python3
"""Controlled 3x2 CoRA rank4 soft-routing factorial (VALID only).

E1/E2: user-only shared router; E3/E4: user-item pair shared router;
E5/E6: user-item pair with four layer-group route heads.
For each route: lambda=0.2 vs lambda=1.0 in mask=1+lambda*(4*p-1).

Does NOT duplicate the rank16 dynamic W_ui generator or layer-wise P matrices.
All four contiguous rank4 experts stay active. Uses unchanged D4_TOPK_RES025
collaborative input and C6 CTR=.005, KD=.010 training configuration.
Fixed seed=42; GPU0 only, serial; train/Valid only, never Test.

Run from the CoRA_official repository root:
  python experiments/no31_cora_original/cora_route_factorial6.py --cpu-test
  python experiments/no31_cora_original/cora_route_factorial6.py --plan
  CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/cora_route_factorial6.py --run
  python experiments/no31_cora_original/cora_route_factorial6.py --summary

Separate protocol names avoid overwriting historical M1-M6 runs.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cora_collab_moe6 as moe
import cora_collab_moe6_summary as recover

DEFAULT_PROTOCOL = "CORA_ROUTE_FACTORIAL6_20261110"
SEED = 42
TRIALS = [
    dict(name="E1_USER_L02", expert_mode="user_soft4", route="user", route_lambda=0.2,
         mode="neighbor_mix", personal_share=0.25),
    dict(name="E2_USER_L10", expert_mode="user_soft4", route="user", route_lambda=1.0,
         mode="neighbor_mix", personal_share=0.25),
    dict(name="E3_PAIR_L02", expert_mode="pair", route="pair", route_lambda=0.2,
         mode="neighbor_mix", personal_share=0.25),
    dict(name="E4_PAIR_L10", expert_mode="pair", route="pair", route_lambda=1.0,
         mode="neighbor_mix", personal_share=0.25),
    dict(name="E5_LAYER_L02", expert_mode="layer_pair", route="layer_pair", route_lambda=0.2,
         mode="neighbor_mix", personal_share=0.25),
    dict(name="E6_LAYER_L10", expert_mode="layer_pair", route="layer_pair", route_lambda=1.0,
         mode="neighbor_mix", personal_share=0.25),
]
ORDER = [x["name"] for x in TRIALS]
TRIAL_BY_NAME = {x["name"]: x for x in TRIALS}
if len(TRIAL_BY_NAME) != 6:
    raise RuntimeError("Expected exactly six independent trials")


def _install_grad_norm_hooks():
    """Collect router gradient norms; original MoE's finite-grad hooks remain."""
    cls = moe.CollaborativeMoE
    if getattr(cls, "_factorial_grad_hook_installed", False):
        return
    previous = cls._grad_hook

    def tracked(self, name):
        original_observer = previous(self, name)

        def observe(grad):
            grad = original_observer(grad)
            if grad is not None:
                history = getattr(self, "_factorial_grad_norm", None)
                if history is None:
                    history = {}
                    self._factorial_grad_norm = history
                value = float(grad.detach().float().norm().item())
                total, count = history.get(name, (0.0, 0))
                history[name] = (total + value, count + 1)
            return grad

        return observe

    cls._grad_hook = tracked
    cls._factorial_grad_hook_installed = True


def install_model(trial):
    # Each trial gets its own fresh Python worker, so this global coefficient
    # cannot leak from a different variant's training process.
    lam = float(trial["route_lambda"])
    if lam not in (0.2, 1.0):
        raise RuntimeError(f"Unapproved lambda: {lam}")
    moe.RESIDUAL_LAMBDA = lam
    _install_grad_norm_hooks()
    moe.install_moe_model(trial)
    print(f"[FACTORIAL6 MODEL] trial={trial['name']} "
          f"route={trial['route']} lambda={lam:.1f} rank=4x4 "
          "D4/C6 unchanged; original P and generator shared", flush=True)


def install_audit(trial):
    moe.install_audit(trial)
    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx
    earlier = MiniGPT4Rec_vx.forward_v2

    def log_gradient_norm(self, samples, *args, **kwargs):
        # The prior completed backward hooks are available at the next forward;
        # no extra forward/backward or GPU run is introduced.
        result = earlier(self, samples, *args, **kwargs)
        step = getattr(self, "_r4e_train_steps", 0)
        if step == 1 or step % 100 == 0:
            gen = self.llama_model_lora.base_model.weight_generator
            router = gen.rank4_experts
            records = getattr(router, "_factorial_grad_norm", {})
            means = {
                key: (value / count if count else 0.0)
                for key, (value, count) in sorted(records.items())
            }
            print(f"[FACTORIAL6 GRAD] trial={trial['name']} "
                  f"step={step} router_grad_norm_mean={means}", flush=True)
            router._factorial_grad_norm = {}
        return result

    MiniGPT4Rec_vx.forward_v2 = log_gradient_norm


def configure(protocol):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", protocol):
        raise ValueError("Protocol must consist of letters, digits, '_' or '-'")
    runner = moe.base
    runner.TRIALS = TRIALS
    runner.TRIAL_BY_NAME = TRIAL_BY_NAME
    runner.ORDER = ORDER
    runner.SCRIPT = SCRIPT
    runner.install_model = install_model
    runner.install_loss_audit = install_audit
    runner.configure(protocol)


def plan(protocol):
    configure(protocol)
    moe.base.plan(protocol)
    root = HERE / "logs" / protocol
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": protocol,
        "source": "CORA_ROUTE_FACTORIAL6",
        "hypothesis": "Separate effects of context type, layer granularity, and routing strength",
        "design": TRIALS,
        "mask": "1 + lambda * (4 * softmax(logits) - 1)",
        "routing": {
            "user": "norm(user MF) -> shared 4-expert router",
            "pair": "concat(norm(user MF), norm(item MF), elementwise product) -> shared router",
            "layer_pair": "same pair input -> four 4-expert layer-group outputs",
        },
        "expert": "existing rank16 = 4 contiguous rank4, all four active",
        "layers": ["1-8", "9-16", "17-24", "25-32"],
        "controls": {
            "seed": 42, "gpu": "physical GPU0, serial", "train_valid_only": True,
            "D4": "TOPK_RES025; personal_share=.25",
            "C6": "main Yes-logit BCE + CTR=.005 + reliability-gated KD=.010",
            "balance_loss": 0.001,
            "cora_lr": 0.0003, "mf_lr": 0.00003,
            "batch_size": 2, "accum_grad_iters": 4, "iters_per_epoch": 400,
            "patience": 10, "original_llm": "Vicuna-7B", "rank": 16,
        },
        "baseline": {
            "S4_valid_auc_historical": 0.733476,
            "S4_valid_uauc_historical": 0.697128,
            "note": "Different experiment batch, not paired with this run",
        },
        "notes": [
            "E2 matches S4's user-only Soft4 mask equation and router shape; full historical-reproduction equivalence requires checking every training setting.",
            "E5 corresponds to M6's lambda=0.2 architecture under the shared protocol.",
            "No freezing or reinitialization of independent LoRA experts: P matrices remain those of the original CoRA.",
            "Do not run GPU1 or Test; do not overwrite old logs/checkpoints.",
        ],
    }
    path = root / "route_factorial6_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[FACTORIAL6 PLAN] manifest={path}")
    print("[FACTORIAL6 PLAN PASS] Six independent configs, Valid only, no GPU work")


def cpu_test():
    torch.manual_seed(SEED)
    dim, batch = 16, 8
    rec = SimpleNamespace(user_embedding=nn.Embedding(50, dim),
                          item_embedding=nn.Embedding(60, dim))
    user_ids = torch.arange(1, batch + 1)
    item_ids = torch.arange(2, batch + 2)
    for trial in TRIALS:
        lam = trial["route_lambda"]
        mode = trial["expert_mode"]
        moe.RESIDUAL_LAMBDA = lam
        model = moe.CollaborativeMoE(mode, dim)
        # Final router layer must begin at zero, making the full-rank
        # CoRA dynamic weights identical before optimizer updates.
        masks, aux, _, probs = model(rec, user_ids, item_ids, {})
        assert tuple(probs.shape) == (batch, 4, 4)
        assert len(masks) == 4
        for mask in masks:
            assert tuple(mask.shape) == (batch, 16)
            torch.testing.assert_close(mask, torch.ones_like(mask), atol=1e-6, rtol=0)
        # Deliberately perturb the logits to prove the formula applies
        # the variant-specific lambda, not the original fixed lambda=0.2.
        with torch.no_grad():
            model.router[-1].bias.copy_(torch.tensor([1.5, -1.0, 0.5, -0.7] *
                                                     (4 if mode == "layer_pair" else 1)))
        masks, aux, _, probs = model(rec, user_ids, item_ids, {})
        expected = 1 + lam * (4 * probs - 1)
        for group, mask in enumerate(masks):
            torch.testing.assert_close(
                mask,
                expected[:, group, :].repeat_interleave(4, dim=-1),
                atol=1e-6, rtol=1e-6,
            )
        if mode != "layer_pair":
            for group in range(1, 4):
                torch.testing.assert_close(probs[:, 0], probs[:, group])
        # Gradients reach the router under the real rank16 weighted computation.
        signal = torch.randn(batch, 4, 4)
        loss = sum(
            (signal[:, g] * (mask.reshape(batch, 4, 4).mean(dim=-1))).sum()
            for g, mask in enumerate(masks)
        ) + aux
        loss.backward()
        required = [p for p in model.parameters() if p.requires_grad]
        assert required and all(p.grad is not None for p in required)
        assert all(bool(torch.isfinite(p.grad).all()) for p in required)
        print(f"[FACTORIAL6 CPU] {trial['name']} mode={mode} "
              f"lambda={lam} init_identity=True gradient=True", flush=True)
    print("[FACTORIAL6 CPU PASS] 6 modes, masks, scaling, identity and gradients")


def summary(protocol, debug=False):
    # Reuse the validated history-safe log/metrics/checkpoint recovery code.
    recover.NAMES = ORDER
    rows = recover.collect(protocol)
    baseline = rows[0]
    ref_auc = baseline["valid_auc"]
    ref_uauc = baseline["valid_uauc_at_best_auc"]
    for row in rows:
        row.pop("delta_auc_vs_S1", None)
        row.pop("delta_uauc_vs_S1", None)
        row["lambda"] = next(t["route_lambda"] for t in TRIALS if t["name"] == row["trial"])
        row["route"] = next(t["route"] for t in TRIALS if t["name"] == row["trial"])
        row["delta_auc_vs_E1"] = (
            row["valid_auc"] - ref_auc
            if row["valid_auc"] is not None and ref_auc is not None else None
        )
        row["delta_uauc_vs_E1"] = (
            row["valid_uauc_at_best_auc"] - ref_uauc
            if row["valid_uauc_at_best_auc"] is not None and ref_uauc is not None else None
        )
        row["delta_auc_vs_historical_S4"] = (
            row["valid_auc"] - 0.733476 if row["valid_auc"] is not None else None
        )
        row["delta_uauc_vs_historical_S4"] = (
            row["valid_uauc_at_best_auc"] - 0.697128
            if row["valid_uauc_at_best_auc"] is not None else None
        )
    header = ("trial", "route", "lambda", "valid_auc", "valid_uauc_at_best_auc",
              "best_epoch", "delta_auc_vs_E1", "delta_uauc_vs_E1",
              "delta_auc_vs_historical_S4", "delta_uauc_vs_historical_S4",
              "status", "checkpoint", "checkpoint_paths", "source",
              "stdout_valid_points", "valid_blocks", "valid_log_files",
              "checkpoint_files", "search_dirs")
    target_dir = HERE / "out" / protocol
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "route_factorial6_valid_summary.csv"
    with target.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print("\nROUTE FACTORIAL6 | VALID only | best AUC paired with same-epoch UAUC")
    print(f"{'trial':18} {'AUC':>10} {'UAUC':>10} {'dAUC-E1':>10} {'dUAUC-E1':>10} {'ep':>4} {'status':>18}")
    for row in rows:
        f = lambda x: "—" if x is None else f"{x:.6f}"
        print(f"{row['trial']:18} {f(row['valid_auc']):>10} "
              f"{f(row['valid_uauc_at_best_auc']):>10} "
              f"{f(row['delta_auc_vs_E1']):>10} "
              f"{f(row['delta_uauc_vs_E1']):>10} "
              f"{str(row['best_epoch'] or '—'):>4} {row['status']:>18}")
        if debug:
            print("  source:", row["source"])
            print("  ckpt:", row["checkpoint_paths"])
    print(f"[FACTORIAL6 SUMMARY] CSV={target}")
    print("[FACTORIAL6 SUMMARY] Checkpoint FOUND_UNVERIFIED is not a model-load test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-test", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--only", default="")
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--screen-epochs", type=int, default=0)
    parser.add_argument("--_worker", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(moe.base.PROJECT_ROOT)
    if args.cpu_test:
        cpu_test()
        return
    if args.patience < 1 or args.screen_epochs < 0:
        parser.error("patience >=1 and screen-epochs >=0 required")
    configure(args.protocol)
    if args._worker:
        moe.base.worker(args._worker, args.protocol, args.patience, args.screen_epochs)
        return
    if args.plan:
        plan(args.protocol)
        return
    if args.summary:
        summary(args.protocol, debug=args.debug)
        return
    if args.run:
        chosen = [x.strip() for x in args.only.split(",") if x.strip()] if args.only else ORDER
        if len(chosen) != len(set(chosen)) or any(x not in TRIAL_BY_NAME for x in chosen):
            parser.error("--only contains an unknown or duplicated trial name")
        if not moe.base.late.base.core.CONFIG_DIR.is_dir():
            plan(args.protocol)
        moe.base.run_selected(chosen, args.protocol, args.patience, args.screen_epochs)
        print("[FACTORIAL6 COMPLETE] All selected runs finished. Valid only.", flush=True)
        return
    parser.error("Specify --cpu-test, --plan, --run or --summary")


if __name__ == "__main__":
    main()
