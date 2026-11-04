#!/usr/bin/env python3
"""Six controlled rank-split CoRA expert experiments (VALID only).

Prerequisite: cora_rank4_experts5.py (fixed hasattr version) and the existing
Joint8 / Combo8 / RankGran7 / Struct10 / LateVote4 experiment modules.

Keeps 32-layer Q/K/V/O CoRA, per-layer rank16=4xrank4 without duplicating
adapter P matrices or the dynamic weight generator. Builds on the tested
training/optimizer/AMP/Valid-only/checkpoint pipeline in cora_rank4_experts5.

Experiments:
 S1_SPLIT_ALL       : split rank16 into 4xrank4, activate all (identity).
 S2_FIXED_TOP2      : deterministic user-ID top2, active rank8.
 S3_SIM_TOP2        : train-only MF user K4 centroids, nearest two experts.
 S4_SOFT4           : learned per-user soft weights over all four, init 1 each.
 S5_WARMUP_TOP2     : first two epochs soft4 -> straight-through top2.
 S6_SHARED_C2F      : low all4, mid shared1+personal2, high shared1+personal1,
                      train-time warmup over first two epochs.

All runs: fresh CoRA + MF init, seed 42, one GPU0, serial foreground,
D4_TOPK_RES025 input, C6 CTR=.005 KD=.01, batch2/accum4/400 steps,
Valid-only early stop patience10. Each trial writes separate logs/checkpoints.

From CoRA_official root:
 python experiments/no31_cora_original/cora_split_experts6.py --cpu-test
 python experiments/no31_cora_original/cora_split_experts6.py --plan
 python -u experiments/no31_cora_original/cora_split_experts6.py --run
 python experiments/no31_cora_original/cora_split_experts6.py --summary
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

EXP_DIR = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import cora_rank4_experts5 as base  # noqa: E402


DEFAULT_PROTOCOL = "CORA_SPLIT_EXPERTS6_20261104"
EXPERTS = 4
RANK_PER = 4
RANK = EXPERTS * RANK_PER
HIDDEN = 64
WARMUP_MICROSTEPS = 2 * 400
BALANCE_WEIGHT = 1e-3

TRIALS = [
    dict(name="S1_SPLIT_ALL", expert_mode="split_all",
         mode="neighbor_mix", personal_share=0.25),
    dict(name="S2_FIXED_TOP2", expert_mode="fixed_top2",
         mode="neighbor_mix", personal_share=0.25),
    dict(name="S3_SIM_TOP2", expert_mode="prototype",
         mode="neighbor_mix", personal_share=0.25),
    dict(name="S4_SOFT4", expert_mode="soft4",
         mode="neighbor_mix", personal_share=0.25),
    dict(name="S5_WARMUP_TOP2", expert_mode="warmup_top2",
         mode="neighbor_mix", personal_share=0.25),
    dict(name="S6_SHARED_C2F", expert_mode="shared_c2f",
         mode="neighbor_mix", personal_share=0.25),
]
TRIAL_BY_NAME = {t["name"]: t for t in TRIALS}
ORDER = [t["name"] for t in TRIALS]
LEARNED = {"soft4", "warmup_top2", "shared_c2f"}
PROTOTYPE = {"prototype"}
WARMUP = {"warmup_top2", "shared_c2f"}


def _hard_mask(indices, k):
    if indices.ndim != 2 or indices.shape[1] != k:
        raise RuntimeError(f"Expert choices must be [B,{k}], got {tuple(indices.shape)}")
    active = torch.zeros(
        (indices.shape[0], EXPERTS), device=indices.device, dtype=torch.float32
    ).scatter_(1, indices.long(), 1.0)
    return active * math.sqrt(EXPERTS / k)


def _to_rank(channels):
    if channels.ndim != 2 or channels.shape[1] != EXPERTS:
        raise RuntimeError(f"Expert mask must be [B,{EXPERTS}]")
    return channels.repeat_interleave(RANK_PER, dim=-1)


def _soft4(probs):
    # Initial logits=0 -> each p=.25 -> every expert weight exactly 1.
    return EXPERTS * probs


def _straight_topk(probs, k):
    indices = probs.topk(k, dim=-1).indices
    hard = _hard_mask(indices, k)
    # Hard forward / soft backward, with matching total selection mass.
    differentiable = k * probs
    hard_ste = hard + differentiable - differentiable.detach()
    return hard_ste, indices


def _balance(probs):
    return BALANCE_WEIGHT * (probs.mean(dim=0) - 1.0 / EXPERTS).square().sum()


class SplitExperts6(nn.Module):
    """User-only routing masks for four contiguous rank4 CoRA experts.

    The existing CoRA dynamic generator and adapter P remain full rank16.
    No independent expert replicas are created.
    """

    def __init__(self, mode, user_dim, prototype_labels=None, prototypes=None):
        super().__init__()
        self.mode = mode
        self.user_dim = int(user_dim)
        self.microsteps = 0  # only training forwards increment; no optimizer state
        self._grad_seen = set()
        self._grad_nonfinite = set()
        if mode in LEARNED:
            # One user-conditioned router shared across layer groups.
            # S6's first expert is always shared, while personal logits
            # control selection among expert slots 1..3.
            logits_out = 3 if mode == "shared_c2f" else EXPERTS
            self.router = nn.Sequential(
                nn.LayerNorm(self.user_dim),
                nn.Linear(self.user_dim, HIDDEN),
                nn.GELU(),
                nn.Linear(HIDDEN, logits_out),
            )
            nn.init.zeros_(self.router[-1].weight)
            nn.init.zeros_(self.router[-1].bias)
        if mode in PROTOTYPE:
            if prototype_labels is None or prototypes is None:
                raise RuntimeError("Similarity Top2 requires train-only user KMeans")
            self.register_buffer(
                "train_cluster_labels", prototype_labels.long(), persistent=False
            )
            self.register_buffer(
                "prototype_centers", prototypes.float(), persistent=False
            )
        for name, param in self.named_parameters():
            param.register_hook(self._observe(name))

    def _observe(self, name):
        def hook(grad):
            self._grad_seen.add(name)
            if not bool(torch.isfinite(grad).all()):
                self._grad_nonfinite.add(name)
            return grad
        return hook

    def _fraction(self):
        # Validation uses the final intended architecture, even before the
        # warmup completes, so checkpoints are reloadable without schedule state.
        if not self.training:
            return 1.0
        return min(1.0, max(0.0, (self.microsteps - 1) / WARMUP_MICROSTEPS))

    def forward(self, user_embeddings, user_ids):
        batch = int(user_embeddings.shape[0])
        if tuple(user_embeddings.shape) != (batch, self.user_dim):
            raise RuntimeError(
                f"Expected MF users [B,{self.user_dim}], got {tuple(user_embeddings.shape)}"
            )
        ids = user_ids.detach().long().reshape(-1).to(user_embeddings.device)
        if ids.numel() != batch:
            raise RuntimeError("MF UserIDs do not match expert batch")
        zeros = user_embeddings.new_zeros((), dtype=torch.float32)
        mode = self.mode
        cold_count = 0
        summary = []

        if mode == "split_all":
            dense = user_embeddings.new_ones((batch, EXPERTS), dtype=torch.float32)
            groups = [dense] * 4
            summary = [torch.arange(EXPERTS, device=ids.device).repeat(batch)] * 4
            aux = zeros

        elif mode == "fixed_top2":
            first = torch.remainder(ids, EXPERTS)
            second = torch.remainder(first + 1, EXPERTS)
            choices = torch.stack((first, second), dim=1)
            selected = _hard_mask(choices, 2)
            groups = [selected] * 4
            summary = [choices.reshape(-1)] * 4
            aux = zeros

        elif mode == "prototype":
            labels = self.train_cluster_labels
            if bool((ids < 0).any()) or bool((ids >= labels.numel()).any()):
                raise RuntimeError("UserID outside MF user table/prototype labels")
            cold_count = int((labels.index_select(0, ids) < 0).sum().item())
            users = F.normalize(user_embeddings.detach().float(), dim=-1, eps=1e-8)
            centers = F.normalize(self.prototype_centers.float(), dim=-1, eps=1e-8)
            choices = (users @ centers.T).topk(2, dim=-1).indices
            selected = _hard_mask(choices, 2)
            groups = [selected] * 4
            summary = [choices.reshape(-1)] * 4
            aux = zeros

        else:
            users = F.normalize(
                user_embeddings.detach().float(), p=2, dim=-1, eps=1e-8
            )
            raw = self.router(users).float()
            if mode == "soft4":
                probs = torch.softmax(raw, dim=-1)
                all_soft = _soft4(probs)
                groups = [all_soft] * 4
                # Argmax is only diagnostic; all 4 experts are actually active.
                all_ids = torch.arange(EXPERTS, device=ids.device).repeat(batch)
                summary = [all_ids] * 4
                aux = _balance(probs)

            elif mode == "warmup_top2":
                probs = torch.softmax(raw, dim=-1)
                soft = _soft4(probs)
                hard, choices = _straight_topk(probs, 2)
                fraction = self._fraction()
                selected = (1 - fraction) * soft + fraction * hard
                groups = [selected] * 4
                summary = [choices.reshape(-1)] * 4
                aux = _balance(probs)

            elif mode == "shared_c2f":
                # shared expert index 0 and personalized indices 1,2,3.
                personal_probs = torch.softmax(raw, dim=-1)
                soft = torch.cat(
                    (torch.ones_like(personal_probs[:, :1]),
                     3.0 * personal_probs), dim=-1
                )
                # Middle: shared + top2 among 3 personal => active rank12.
                mid_ids = personal_probs.topk(2, dim=-1).indices + 1
                shared = torch.zeros((batch, 1), device=ids.device, dtype=torch.long)
                mid_hard = _hard_mask(torch.cat((shared, mid_ids), dim=1), 3)
                mid_soft_grad = torch.cat(
                    (torch.ones_like(personal_probs[:, :1]),
                     2.0 * personal_probs), dim=-1
                )
                mid = mid_hard + mid_soft_grad - mid_soft_grad.detach()
                # Deep: shared + top1 among 3 personal => active rank8.
                deep_ids = personal_probs.topk(1, dim=-1).indices + 1
                deep_hard = _hard_mask(torch.cat((shared, deep_ids), dim=1), 2)
                deep_soft_grad = torch.cat(
                    (torch.ones_like(personal_probs[:, :1]), personal_probs), dim=-1
                )
                deep = deep_hard + deep_soft_grad - deep_soft_grad.detach()

                fraction = self._fraction()
                mid = (1 - fraction) * soft + fraction * mid
                deep = (1 - fraction) * soft + fraction * deep
                # Four 8-layer groups: low+low, middle, deep.
                groups = [soft, soft, mid, deep]
                dense_ids = torch.arange(EXPERTS, device=ids.device).repeat(batch)
                summary = [
                    dense_ids, dense_ids,
                    torch.cat((shared, mid_ids), dim=1).reshape(-1),
                    torch.cat((shared, deep_ids), dim=1).reshape(-1),
                ]
                # Balance personal-expert demand, not the forced shared slot.
                aux = BALANCE_WEIGHT * (
                    personal_probs.mean(dim=0) - (1.0 / 3.0)
                ).square().sum()
            else:
                raise RuntimeError(f"Unsupported split-experts mode: {mode}")

        masks = [_to_rank(ch) for ch in groups]
        # Aggregated by inherited trainer over 100 microsteps.
        counts = [
            torch.bincount(choice.detach().long().reshape(-1),
                           minlength=EXPERTS).cpu().tolist()
            for choice in summary
        ]
        stats = {
            "expert_counts": counts,
            "cold_users_in_batch": cold_count,
        }
        return masks, aux, stats


_ORIGINAL_INSTALL_AUDIT = base.install_loss_audit


def _install_loss_and_progress(trial):
    """Increment schedule on train forwards only, before CoRA calls generator."""
    _ORIGINAL_INSTALL_AUDIT(trial)
    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx
    previous = MiniGPT4Rec_vx.forward_v2

    def forward_with_step(self, samples, *args, **kwargs):
        expert = (
            self.llama_model_lora.base_model.weight_generator.rank4_experts
        )
        if self.training and expert.training:
            expert.microsteps += 1
        return previous(self, samples, *args, **kwargs)

    MiniGPT4Rec_vx.forward_v2 = forward_with_step


def _setup(protocol):
    # Require the existing first-batch nn.Module membership fix.
    base_text = (EXP_DIR / "cora_rank4_experts5.py").read_text(
        encoding="utf-8", errors="replace"
    )
    if 'if not hasattr(gen, "_r4e_aux"):' not in base_text:
        raise RuntimeError(
            "Existing cora_rank4_experts5.py is outdated. Download the "
            "fixed base script from this experiment's GitHub branch first."
        )

    base.TRIALS = TRIALS
    base.TRIAL_BY_NAME = TRIAL_BY_NAME
    base.ORDER = ORDER
    base.Rank4Experts = SplitExperts6
    base.LEARNED = LEARNED
    base.SCRIPT = SCRIPT
    base.install_loss_audit = _install_loss_and_progress
    base.configure(protocol)


def plan(protocol):
    _setup(protocol)
    base.plan(protocol)
    manifest_file = base.late.base.core.LOG_DIR / "rank4_experts_manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["experiment"] = (
        "six variants of 4xrank4 contiguous experts within original CoRA rank16; "
        "S1 identity, S2 fixed Top2, S3 similar-user Top2, S4 user soft4, "
        "S5 warmup Top2, S6 shared + personalized coarse-to-fine"
    )
    manifest["non_destructive_cold_user"] = (
        "S3 uses four fixed train-only user centroid prototypes; "
        "unseen users use nearest two centroid experts with no parameter overwrite"
    )
    manifest["reference"] = (
        "RR1_L32_BASE prior run: Valid AUC=0.732387 UAUC=0.701999 "
        "(historical, not a same-checkpoint rerun)"
    )
    manifest["routing_schedule"] = {
        "S5": "train microsteps 1..800 interpolate soft4 to STE Top2",
        "S6": "train microsteps 1..800 interpolate soft4 to "
              "shared+2-personal middle and shared+1-personal deep",
        "validation": "always uses final hard architecture; independent of warmup",
        "deep_layers": "L25-L32",
        "middle_layers": "L17-L24",
        "shallow_layers": "L1-L16",
    }
    manifest["loss"] = (
        "C6: BCE Yes + CTR=.005 + KD=.010, plus learned-router "
        "mean expert-utilization penalty 1e-3"
    )
    manifest["trials"] = TRIALS
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[S6 PLAN PASS] six independent Valid-only configs; no GPU model loaded")


def cpu_test():
    torch.manual_seed(42)
    batch, udim, hidden = 8, 16, 32
    users = torch.randn(batch, udim)
    ids = torch.arange(batch, dtype=torch.long)
    W = torch.randn(batch, hidden, RANK, requires_grad=True)
    P = torch.randn(24, RANK, requires_grad=True)
    x = torch.randn(batch, hidden, requires_grad=True)
    dense = F.linear(torch.bmm(x.unsqueeze(1), W).squeeze(1), P)
    pieces = sum(
        F.linear(
            torch.bmm(x.unsqueeze(1), Wi).squeeze(1), Pi
        )
        for Wi, Pi in zip(W.split(RANK_PER, -1), P.split(RANK_PER, -1))
    )
    torch.testing.assert_close(dense, pieces, atol=1e-4, rtol=1e-5)
    labels = torch.tensor([0, 1, 2, 3, -1, 0, 1, -1])
    centroids = F.normalize(torch.randn(EXPERTS, udim), dim=-1)

    for trial in TRIALS:
        mode = trial["expert_mode"]
        kwargs = {}
        if mode == "prototype":
            kwargs = dict(prototype_labels=labels, prototypes=centroids)
        module = SplitExperts6(mode, udim, **kwargs)
        masks, aux, stats = module(users, ids)
        assert len(masks) == 4
        assert all(tuple(m.shape) == (batch, RANK) for m in masks)
        assert all(bool(torch.isfinite(m).all()) for m in masks)
        if mode in ("split_all", "soft4", "warmup_top2", "shared_c2f"):
            # At init: all masks equal dense CoRA (except warmup none yet).
            for mask in masks:
                torch.testing.assert_close(
                    mask, torch.ones_like(mask), atol=1e-6, rtol=0
                )
        if mode in ("fixed_top2", "prototype"):
            assert all(
                bool((m.view(batch, EXPERTS, RANK_PER).abs() > 1e-6)
                     .any(dim=-1).sum(dim=-1).eq(2).all())
                for m in masks
            )
        if mode == "prototype":
            assert stats["cold_users_in_batch"] == 2
        if mode in WARMUP:
            module.microsteps = WARMUP_MICROSTEPS + 1
            masks_hard, _, _ = module(users, ids)
            active = [2] * 4 if mode == "warmup_top2" else [4, 4, 3, 2]
            for mask, k in zip(masks_hard, active):
                seen = mask.reshape(batch, EXPERTS, RANK_PER)
                assert bool((seen.abs() > 1e-5).any(dim=-1).sum(dim=-1).eq(k).all())
            # Eval is always final architecture, independent of training schedule.
            module.eval()
            module.microsteps = 0
            masks_eval, _, _ = module(users, ids)
            for ma, mb in zip(masks_hard, masks_eval):
                torch.testing.assert_close(ma, mb, atol=1e-6, rtol=0)

        fused = F.linear(
            torch.bmm(x.unsqueeze(1), W * masks[-1].unsqueeze(1)).squeeze(1), P
        )
        loss = fused.square().mean() + aux
        loss.backward(retain_graph=True)
        if mode in LEARNED:
            params = list(module.parameters())
            assert params and all(p.grad is not None for p in params)
            assert all(bool(torch.isfinite(p.grad).all()) for p in params)
        if mode == "split_all":
            torch.testing.assert_close(fused, dense, atol=1e-5, rtol=1e-5)

    assert len(TRIALS) == 6
    print(
        "[S6 CPU PASS] rank16=4xrank4 dense equivalence; six masks; "
        "Top2; train-only prototypes; soft4 identity; 2-epoch warmup; "
        "shared coarse-to-fine; finite user-router gradients",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-test", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--only", default="")
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--patience", type=int, default=base.PATIENCE)
    parser.add_argument("--screen-epochs", type=int, default=0)
    parser.add_argument("--_worker", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(base.PROJECT_ROOT)

    if args.cpu_test:
        cpu_test()
        return
    if args.patience < 1 or args.screen_epochs < 0:
        parser.error("patience >= 1 and screen-epochs >= 0 required")
    _setup(args.protocol)
    if args._worker:
        base.worker(args._worker, args.protocol, args.patience, args.screen_epochs)
        return
    if args.plan:
        plan(args.protocol)
        return
    if args.summary:
        from cora_split_experts6_recover import recover, display, write_csv
        rows = recover(args.protocol)
        display(rows, debug=False)
        write_csv(rows, args.protocol)
        return
    if args.run:
        selected = [x.strip() for x in args.only.split(",") if x.strip()] if args.only else ORDER
        if len(selected) != len(set(selected)) or any(name not in TRIAL_BY_NAME for name in selected):
            parser.error("--only contains invalid or duplicate trial names")
        if not base.late.base.core.CONFIG_DIR.is_dir():
            plan(args.protocol)
        base.run_selected(selected, args.protocol, args.patience, args.screen_epochs)
        return
    parser.error("Specify --cpu-test, --plan, --run, or --summary")


if __name__ == "__main__":
    main()
