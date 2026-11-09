#!/usr/bin/env python3
"""Six controlled routing-structure x routing-strength CoRA rank-split experiments.

E1/E2: user-only shared router, lambda 0.2 / 1.0.
E3/E4: user-item pair shared router, lambda 0.2 / 1.0.
E5/E6: user-item pair with four independent layer-group output heads,
       lambda 0.2 / 1.0.

All runs reuse the existing CoRA C6+D4 trainer. Rank16 is partitioned into
four contiguous rank4 channels inside the SAME dynamic W_ui and P tensors.
No copied LoRA matrices, no hard Top-K, all experts remain in the forward path.
Valid only; one physical GPU0; serial subprocesses; seed 42.

Commands:
  python experiments/no31_cora_original/cora_routing_factorial6.py --cpu-test
  python experiments/no31_cora_original/cora_routing_factorial6.py --plan
  python -u experiments/no31_cora_original/cora_routing_factorial6.py --run
  python experiments/no31_cora_original/cora_routing_factorial6.py --summary
"""
import argparse
import csv
import json
import os
import re
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import cora_collab_moe6 as engine


HERE = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()
DEFAULT_PROTOCOL = "CORA_ROUTING_FACTORIAL6_20261109"
SEED = 42
EXPERTS = 4
RANK_PER_EXPERT = 4
RANK = EXPERTS * RANK_PER_EXPERT
DIAG_INTERVAL = 100

TRIALS = [
    dict(name="E1_USER_SOFT4_L02", expert_mode="user", route_lambda=0.2,
         mode="neighbor_mix", personal_share=0.25),
    dict(name="E2_USER_SOFT4_L10", expert_mode="user", route_lambda=1.0,
         mode="neighbor_mix", personal_share=0.25),
    dict(name="E3_PAIR_SHARED_L02", expert_mode="pair", route_lambda=0.2,
         mode="neighbor_mix", personal_share=0.25),
    dict(name="E4_PAIR_SHARED_L10", expert_mode="pair", route_lambda=1.0,
         mode="neighbor_mix", personal_share=0.25),
    dict(name="E5_LAYER_PAIR_L02", expert_mode="layer_pair", route_lambda=0.2,
         mode="neighbor_mix", personal_share=0.25),
    dict(name="E6_LAYER_PAIR_L10", expert_mode="layer_pair", route_lambda=1.0,
         mode="neighbor_mix", personal_share=0.25),
]
TRIAL_BY_NAME = {x["name"]: x for x in TRIALS}
ORDER = [x["name"] for x in TRIALS]

_BaseCollaborativeMoE = engine.CollaborativeMoE
_ORIGINAL_INSTALL_MOE_MODEL = engine.install_moe_model
_ORIGINAL_INSTALL_AUDIT = engine.install_audit


class UserOnlyRouter(nn.Module):
    """Accept the parent pair-router tensor but use only its user prefix."""
    def __init__(self, dim):
        super().__init__()
        self.dim = int(dim)
        self.net = engine._build_mlp(self.dim, EXPERTS)

    def forward(self, pair_features):
        if pair_features.ndim != 2 or pair_features.shape[-1] != 3 * self.dim:
            raise RuntimeError(
                f"User router expected [B,{3*self.dim}] parent input, "
                f"got {tuple(pair_features.shape)}"
            )
        return self.net(pair_features[:, :self.dim])


class FactorialCollaborativeMoE(_BaseCollaborativeMoE):
    """Existing CoRA MoE router with an added user-only routing mode."""
    def __init__(self, mode, dim, labels=None, centers=None):
        if mode == "user":
            # The parent 'pair' forward path constructs [user,item,user*item].
            # UserOnlyRouter deliberately consumes only the first user block.
            nn.Module.__init__(self)
            self.mode = "pair"
            self.dim = int(dim)
            self._grad_seen = set()
            self._grad_nonfinite = set()
            self.router = UserOnlyRouter(dim)
            for name, param in self.named_parameters():
                if param.requires_grad:
                    param.register_hook(self._grad_hook(name))
        else:
            super().__init__(mode, dim, labels=labels, centers=centers)
        self.route_type = mode
        self._last_probs = None

    def forward(self, rec, user_ids, item_ids, samples):
        result = super().forward(rec, user_ids, item_ids, samples)
        self._last_probs = result[3].detach()
        return result


def _new_window():
    return {
        "prob_sum": torch.zeros(EXPERTS, EXPERTS),
        "prob_sumsq": torch.zeros(EXPERTS, EXPERTS),
        "entropy_sum": torch.zeros(EXPERTS),
        "samples": 0,
    }


def _record_rank_gradient(grad, generator, group):
    if grad.shape[-1] != RANK:
        raise RuntimeError(
            f"Expected rank-{RANK} generated CoRA weight gradient, got {tuple(grad.shape)}"
        )
    chunks = grad.detach().float().reshape(-1, EXPERTS, RANK_PER_EXPERT)
    norms = torch.linalg.vector_norm(chunks, dim=(0, 2)).cpu()
    window = getattr(generator, "_rfx6_grad_window", None)
    if window is None:
        window = {
            "norm_sum": torch.zeros(EXPERTS, EXPERTS),
            "calls": torch.zeros(EXPERTS),
            "nonfinite": 0,
        }
        generator._rfx6_grad_window = window
    window["norm_sum"][group] += norms
    window["calls"][group] += 1
    if not bool(torch.isfinite(grad).all()):
        window["nonfinite"] += 1
    return grad


def install_factorial_model(trial):
    """Install the existing model patch with the lambda selected for this run."""
    engine.RESIDUAL_LAMBDA = float(trial["route_lambda"])
    engine.CollaborativeMoE = FactorialCollaborativeMoE
    _ORIGINAL_INSTALL_MOE_MODEL(trial)

    from minigpt4.models.ppeft.weight_generator import PerceptualWeightsGenerator

    previous_forward = PerceptualWeightsGenerator.forward
    if getattr(previous_forward, "_rfx6_diagnostics", False):
        raise RuntimeError("RFX6 generator diagnostics were already installed")

    def forward_with_diagnostics(self, ui_embedding):
        result = previous_forward(self, ui_embedding)
        if (
            self.training and torch.is_grad_enabled()
            and isinstance(result, (list, tuple))
            and hasattr(self, "rank4_experts")
        ):
            probs = self.rank4_experts._last_probs
            if probs is None or tuple(probs.shape)[1:] != (EXPERTS, EXPERTS):
                raise RuntimeError("RFX6 expected router probabilities [B,4,4]")
            route_window = getattr(self, "_rfx6_route_window", None)
            if route_window is None:
                route_window = _new_window()
                self._rfx6_route_window = route_window
            p = probs.detach().float().cpu()
            route_window["prob_sum"] += p.sum(dim=0)
            route_window["prob_sumsq"] += p.square().sum(dim=0)
            entropy = -(p.clamp_min(1e-8) * p.clamp_min(1e-8).log()).sum(dim=-1)
            route_window["entropy_sum"] += entropy.sum(dim=0)
            route_window["samples"] += int(p.shape[0])

            seen = set()
            for layer_idx, weight in enumerate(result):
                if (
                    not torch.is_tensor(weight) or not weight.requires_grad
                    or id(weight) in seen
                ):
                    continue
                seen.add(id(weight))
                group = layer_idx // 8
                if group < 0 or group >= EXPERTS:
                    raise RuntimeError(f"Unexpected CoRA layer index: {layer_idx}")
                weight.register_hook(
                    lambda grad, gen=self, grp=group:
                    _record_rank_gradient(grad, gen, grp)
                )
        return result

    forward_with_diagnostics._rfx6_diagnostics = True
    PerceptualWeightsGenerator.forward = forward_with_diagnostics
    print(
        f"[RFX6 INIT] trial={trial['name']} route={trial['expert_mode']} "
        f"lambda={trial['route_lambda']} rank16=4xrank4 "
        "shared original CoRA W_ui/P; no expert parameter copies",
        flush=True,
    )


def _print_diagnostics(generator, trial_name, step, route_lambda):
    route = getattr(generator, "_rfx6_route_window", None)
    grad = getattr(generator, "_rfx6_grad_window", None)

    if route is not None and route["samples"] > 0:
        n = route["samples"]
        mean = route["prob_sum"] / n
        var = (route["prob_sumsq"] / n - mean.square()).clamp_min(0)
        entropy = route["entropy_sum"] / n
        print(
            f"[RFX6 ROUTE] {trial_name} step={step} lambda={route_lambda:.1f} "
            f"group_prob_mean={mean.tolist()} "
            f"group_prob_std={var.sqrt().tolist()} "
            f"group_entropy={entropy.tolist()}",
            flush=True,
        )
        delattr(generator, "_rfx6_route_window")

    if grad is not None:
        denom = grad["calls"].clamp_min(1).unsqueeze(1)
        grad_mean = grad["norm_sum"] / denom
        print(
            f"[RFX6 GRAD] {trial_name} step={step} "
            f"rank4_grad_norm_by_group={grad_mean.tolist()} "
            f"hook_calls_by_group={grad['calls'].tolist()} "
            f"nonfinite_hooks={grad['nonfinite']}",
            flush=True,
        )
        delattr(generator, "_rfx6_grad_window")


def install_factorial_audit(trial):
    """Keep the tested loss/optimizer audit and add per-group route/grad logs."""
    _ORIGINAL_INSTALL_AUDIT(trial)
    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx

    previous_forward = MiniGPT4Rec_vx.forward_v2

    def forward_with_factorial_diagnostics(self, samples, *args, **kwargs):
        generator = self.llama_model_lora.base_model.weight_generator
        # The preceding training forward has completed backward by the time
        # the next forward starts, so its rank-channel gradients are available.
        completed_steps = getattr(self, "_r4e_train_steps", 0)
        if completed_steps > 0 and completed_steps % DIAG_INTERVAL == 0:
            _print_diagnostics(
                generator, trial["name"], completed_steps,
                float(trial["route_lambda"]),
            )
        return previous_forward(self, samples, *args, **kwargs)

    MiniGPT4Rec_vx.forward_v2 = forward_with_factorial_diagnostics


def configure(protocol):
    engine.TRIALS = TRIALS
    engine.TRIAL_BY_NAME = TRIAL_BY_NAME
    engine.ORDER = ORDER
    engine.SCRIPT = SCRIPT
    engine.CollaborativeMoE = FactorialCollaborativeMoE
    engine.install_moe_model = install_factorial_model
    engine.install_audit = install_factorial_audit
    engine.configure(protocol)


def plan(protocol):
    configure(protocol)
    engine.plan(protocol)
    manifest_path = engine.base.late.base.core.LOG_DIR / "collab_moe6_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "experiment": "3 routing structures x 2 residual routing strengths",
        "trials": TRIALS,
        "lambda_by_trial": {
            trial["name"]: trial["route_lambda"] for trial in TRIALS
        },
        "router_inputs": {
            "user": "normalized detached MF user embedding only",
            "pair": "normalized detached [user, item, user*item], one shared route for all 4 layer groups",
            "layer_pair": "same user-item pair input, 4 independent output groups of 4 expert logits",
        },
        "mask": "per-trial lambda; see lambda_by_trial",
        "mask_formula": "mask = 1 + lambda * (4 * softmax(logits) - 1)",
        "lambda": "per_trial",
        "lambda_values": [0.2, 1.0],
        "diagnostics": [
            "per-layer-group expert probability mean/std",
            "per-layer-group routing entropy",
            "per-layer-group per-rank4 generated-W_ui gradient norms",
            "nonfinite gradient-hook count",
        ],
        "controls": {
            "seed": SEED, "experts": EXPERTS, "rank_per_expert": RANK_PER_EXPERT,
            "total_rank": RANK, "same_C6_D4_training_pipeline": True,
            "valid_only": True, "test_disabled": True,
            "same_MF_checkpoint_and_fresh_CoRA_init_per_trial": True,
            "same_balance_loss": 1e-3,
        },
    })
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[RFX6 PLAN PASS] six trials; manifest={manifest_path}; no GPU model loaded")


def summary(protocol, debug=False):
    import cora_collab_moe6_summary as summary_module
    summary_module.NAMES = ORDER
    rows = summary_module.collect(protocol)
    for row in rows:
        row["delta_auc_vs_E1"] = row.pop("delta_auc_vs_S1")
        row["delta_uauc_vs_E1"] = row.pop("delta_uauc_vs_S1")

    print("\nROUTING FACTORIAL 6 — BEST VALID (NOT TEST)")
    print(f'{"trial":24} {"AUC":>10} {"UAUC@best":>12} '
          f'{"dAUC vs E1":>12} {"dUAUC vs E1":>13} {"Epoch":>7} '
          f'{"Status":>16} {"Checkpoint":>18}')
    print("-" * 126)
    for row in rows:
        auc = row["valid_auc"]
        uauc = row["valid_uauc_at_best_auc"]
        da = row["delta_auc_vs_E1"]
        du = row["delta_uauc_vs_E1"]
        epoch = str(row["best_epoch"]) if row["best_epoch"] is not None else "—"
        fmt = lambda value: "—" if value is None else f"{value:.6f}"
        signed = lambda value: "—" if value is None else f"{value:+.6f}"
        print(
            f'{row["trial"]:24} {fmt(auc):>10} {fmt(uauc):>12} '
            f'{signed(da):>12} {signed(du):>13} {epoch:>7} '
            f'{row["status"]:>16} {row["checkpoint"]:>18}'
        )
        if debug:
            print("  source=", row["source"])
            print("  checkpoints=", row["checkpoint_paths"])
            print("  search_dirs=", row["search_dirs"])

    out = HERE / "out" / protocol
    out.mkdir(parents=True, exist_ok=True)
    target = out / "routing_factorial6_valid_summary.csv"
    with target.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[RFX6 SUMMARY] {target}")
    print(
        "UAUC is paired with the best-AUC validation record. "
        "FOUND_UNVERIFIED checks file/ZIP structure only, not model loadability. "
        "No GPU or TEST evaluation was performed."
    )


def cpu_test():
    from types import SimpleNamespace

    torch.manual_seed(SEED)
    dim, batch = 16, 8
    rec = SimpleNamespace(
        user_embedding=nn.Embedding(40, dim),
        item_embedding=nn.Embedding(50, dim),
    )
    users = torch.arange(1, batch + 1)
    items = torch.arange(2, batch + 2)
    samples = {}
    assert len(TRIALS) == 6

    for trial in TRIALS:
        route_lambda = float(trial["route_lambda"])
        engine.RESIDUAL_LAMBDA = route_lambda
        module = FactorialCollaborativeMoE(trial["expert_mode"], dim)
        masks, aux, stats, probs = module(rec, users, items, samples)

        assert tuple(probs.shape) == (batch, EXPERTS, EXPERTS)
        assert len(masks) == EXPERTS
        assert all(tuple(mask.shape) == (batch, RANK) for mask in masks)
        assert all(torch.isfinite(mask).all() for mask in masks)
        assert all(bool((mask >= -1e-7).all()) for mask in masks)

        # Zero-initialized router outputs give the exact original CoRA identity.
        for mask in masks:
            torch.testing.assert_close(
                mask, torch.ones_like(mask), atol=1e-6, rtol=0
            )
            chunks = mask.reshape(batch, EXPERTS, RANK_PER_EXPERT)
            torch.testing.assert_close(
                chunks, chunks[:, :, :1].expand_as(chunks), atol=1e-6, rtol=0
            )

        # Validate the mathematically expected mask bounds for this lambda.
        random_p = torch.softmax(torch.randn(batch, EXPERTS), dim=-1)
        random_mask = 1.0 + route_lambda * (EXPERTS * random_p - 1.0)
        assert float(random_mask.min()) >= 1.0 - route_lambda - 1e-6
        assert float(random_mask.max()) <= 1.0 + 3.0 * route_lambda + 1e-6

        if trial["expert_mode"] == "pair":
            for group in range(1, EXPERTS):
                torch.testing.assert_close(probs[:, 0], probs[:, group])
        if trial["expert_mode"] == "user":
            changed_items = (items + 7) % 49 + 1
            _, _, _, user_probs_again = module(rec, users, changed_items, samples)
            torch.testing.assert_close(probs, user_probs_again)

        coeffs = [torch.randn_like(mask) for mask in masks]
        loss = sum((mask * coeff).sum() for mask, coeff in zip(masks, coeffs)) + aux
        loss.backward()
        params = [p for p in module.parameters() if p.requires_grad]
        assert params and all(p.grad is not None for p in params)
        assert all(bool(torch.isfinite(p.grad).all()) for p in params)

        if trial["expert_mode"] == "layer_pair":
            with torch.no_grad():
                module.router[-1].weight.normal_(mean=0.0, std=0.05)
            _, _, _, varied = module(rec, users, items, samples)
            assert any(
                not torch.allclose(varied[:, 0], varied[:, group])
                for group in range(1, EXPERTS)
            ), "Layer-pair heads must be able to produce different group routes"

    print(
        "[RFX6 CPU PASS] 6 trials; user-only/pair-shared/layer-pair routing; "
        "lambda 0.2/1.0; identity initialization; mask bounds; "
        "router gradients; layer-group route independence",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-test", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--only", default="")
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--patience", type=int, default=engine.base.PATIENCE)
    parser.add_argument("--screen-epochs", type=int, default=0)
    parser.add_argument("--_worker", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    os.chdir(engine.base.PROJECT_ROOT)
    if args.cpu_test:
        cpu_test()
        return
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.protocol):
        parser.error("Protocol accepts letters, digits, '_' and '-' only")
    if args.patience < 1 or args.screen_epochs < 0:
        parser.error("patience >=1 and screen-epochs >=0 required")

    configure(args.protocol)
    if args._worker:
        engine.base.worker(
            args._worker, args.protocol, args.patience, args.screen_epochs
        )
        return
    if args.plan:
        plan(args.protocol)
        return
    if args.summary:
        summary(args.protocol, debug=args.debug)
        return
    if args.run:
        selected = (
            [x.strip() for x in args.only.split(",") if x.strip()]
            if args.only else list(ORDER)
        )
        if len(selected) != len(set(selected)) or any(
            x not in TRIAL_BY_NAME for x in selected
        ):
            parser.error("--only contains unknown or duplicate trial names")
        if not engine.base.late.base.core.CONFIG_DIR.is_dir():
            plan(args.protocol)
        engine.base.run_selected(
            selected, args.protocol, args.patience, args.screen_epochs
        )
        print("[RFX6 COMPLETE] selected Valid-only trials finished", flush=True)
        return
    parser.error("Specify --cpu-test, --plan, --run, or --summary")


if __name__ == "__main__":
    main()
