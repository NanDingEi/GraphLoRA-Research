#!/usr/bin/env python3
"""Six end-to-end collaborative-aware CoRA LoRA-MoE experiments, VALID only.

Rank16 is split into 4xrank4 contiguous experts INSIDE existing CoRA dynamic
W_ui and P. ALL four experts stay active; a soft router changes their weight:
  mask_k = 1 + 0.2 * (4 * softmax(router(context))_k - 1).
At router initialization all masks equal 1, exactly matching original CoRA.

M1 GLOBAL:          global learned expert weighting, no user/item.
M2 USER CLUSTER:    four train-only MF user centroids, shared cluster router.
M3 ITEM CLUSTER:    four train-only MF item centroids, shared cluster router.
M4 PAIR:            user MF + target MF + elementwise user*item.
M5 MULTI INTEREST:  three trainable interests attend to the user's real sas_seq
                    history, target-item attention chooses context for routing.
M6 LAYER PAIR:      four independent group routes from the user-item pair,
                    applied to L1-8, L9-16, L17-24, L25-32.

Reuses the existing tested CoRA C6/D4 training and checkpoint infrastructure.
Independent fresh init, seed42, only GPU0, serial foreground, VALID only.
No copies of LoRA P matrices, no hard Top-K rank removal, no TEST.

Commands:
 python experiments/no31_cora_original/cora_collab_moe6.py --cpu-test
 python experiments/no31_cora_original/cora_collab_moe6.py --plan
 python -u experiments/no31_cora_original/cora_collab_moe6.py --run
 python experiments/no31_cora_original/cora_collab_moe6.py --summary
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cora_rank4_experts5 as base  # noqa: E402


DEFAULT_PROTOCOL = "CORA_COLLAB_MOE6_20261104"
SEED = 42
EXPERTS = 4
RANK_PER = 4
RANK = EXPERTS * RANK_PER
MLP_HIDDEN = 64
INTERESTS = 3
TEMPERATURE = 0.2
RESIDUAL_LAMBDA = 0.2
BALANCE_WEIGHT = 1e-3

TRIALS = [
    dict(name="M1_GLOBAL", expert_mode="global",
         mode="neighbor_mix", personal_share=0.25),
    dict(name="M2_USER_CLUSTER", expert_mode="user_cluster",
         mode="neighbor_mix", personal_share=0.25),
    dict(name="M3_ITEM_CLUSTER", expert_mode="item_cluster",
         mode="neighbor_mix", personal_share=0.25),
    dict(name="M4_PAIR", expert_mode="pair",
         mode="neighbor_mix", personal_share=0.25),
    dict(name="M5_MULTI_INTEREST", expert_mode="multi_interest",
         mode="neighbor_mix", personal_share=0.25),
    dict(name="M6_LAYER_PAIR", expert_mode="layer_pair",
         mode="neighbor_mix", personal_share=0.25),
]
TRIAL_BY_NAME = {t["name"]: t for t in TRIALS}
ORDER = [t["name"] for t in TRIALS]
CLUSTERS = {"user_cluster", "item_cluster"}


def _build_mlp(input_dim, output_dim):
    network = nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, MLP_HIDDEN),
        nn.GELU(),
        nn.Linear(MLP_HIDDEN, output_dim),
    )
    # Exactly uniform expert masks at start, but hidden features are random
    # and the final layer acquires gradient on the very first backward pass.
    nn.init.zeros_(network[-1].weight)
    nn.init.zeros_(network[-1].bias)
    return network


def _train_centroids(rec, kind):
    """Build deterministic K4 from positive/history train-only OOD2 IDs."""
    if kind not in CLUSTERS:
        raise ValueError(f"Unexpected centroid kind: {kind}")
    users, items, train_path = base.late.base._load_train_entity_ids()
    is_user = kind == "user_cluster"
    seen = users if is_user else items
    table = (
        rec.user_embedding.weight if is_user else rec.item_embedding.weight
    )
    if len(seen) <= EXPERTS:
        raise RuntimeError(f"Insufficient training entities for 4 centroids: {len(seen)}")
    clusters = base.late.base._kmeans_cosine(
        table, EXPERTS, seed=SEED + (19 if is_user else 23),
        seen_ids=seen,
    )
    centers = []
    for ids in clusters["members"]:
        if ids.numel() == 0:
            raise RuntimeError("Encountered empty train-only expert cluster")
        mean = table.detach().float().cpu().index_select(
            0, ids.long()
        ).mean(dim=0)
        centers.append(F.normalize(mean, dim=-1, eps=1e-8))
    print(
        f"[MOE6 TRAIN CLUSTERS] kind={kind} source={train_path} "
        f"n_train_seen={len(seen)} sizes={[int(x.numel()) for x in clusters['members']]} "
        "valid/test entity IDs not used for construction",
        flush=True,
    )
    return clusters["labels"], torch.stack(centers)


class CollaborativeMoE(nn.Module):
    """Four rank4 soft experts, routed by a selected recommendation signal.

    Input to router is detached MF data; router and original CoRA generator/P
    still receive gradients through masked dynamic weights and the main task.
    """

    def __init__(self, mode, dim, labels=None, centers=None):
        super().__init__()
        self.mode = str(mode)
        self.dim = int(dim)
        self._grad_seen = set()
        self._grad_nonfinite = set()

        if mode == "global":
            self.global_logits = nn.Parameter(torch.zeros(EXPERTS))
        elif mode in CLUSTERS:
            if labels is None or centers is None:
                raise RuntimeError(f"{mode} requires train-only centroids")
            self.register_buffer("cluster_labels", labels.long(), persistent=False)
            self.register_buffer("cluster_centers", centers.float(), persistent=False)
            self.cluster_logits = nn.Parameter(torch.zeros(EXPERTS, EXPERTS))
        elif mode == "pair":
            self.router = _build_mlp(3 * dim, EXPERTS)
        elif mode == "multi_interest":
            self.router = _build_mlp(4 * dim, EXPERTS)
            self.interest_queries = nn.Parameter(
                torch.empty(INTERESTS, dim).normal_(mean=0.0, std=0.02)
            )
        elif mode == "layer_pair":
            self.router = _build_mlp(3 * dim, EXPERTS * 4)
        else:
            raise ValueError(f"Unknown CollaborativeMoE type: {mode}")

        for name, param in self.named_parameters():
            if param.requires_grad:
                param.register_hook(self._grad_hook(name))

    def _grad_hook(self, name):
        def observe(g):
            self._grad_seen.add(name)
            if not bool(torch.isfinite(g).all()):
                self._grad_nonfinite.add(name)
            return g
        return observe

    def _route_cluster(self, ids, vectors):
        labels = self.cluster_labels
        centers = self.cluster_centers.to(vectors.device).float()
        ids = ids.long().to(labels.device).reshape(-1)
        if bool((ids < 0).any()) or bool((ids >= labels.numel()).any()):
            raise RuntimeError("MF user/item ID is outside cluster label table")
        assigned = labels.index_select(0, ids).long()
        unknown = assigned.lt(0)
        if bool(unknown.any()):
            query = F.normalize(
                vectors.detach().float()[unknown], p=2, dim=-1, eps=1e-8
            )
            nearest = (query @ F.normalize(
                centers, p=2, dim=-1, eps=1e-8
            ).T).argmax(dim=-1)
            assigned = assigned.clone()
            assigned[unknown] = nearest
        logits = self.cluster_logits.index_select(0, assigned)
        return logits, int(unknown.sum().item())

    def _multi_interest(self, candidate, history, item_table):
        if history is None:
            raise RuntimeError(
                "M5_MULTI_INTEREST requires samples['sas_seq']; "
                "will not silently replace multi-interest routing with user-only routing"
            )
        if not torch.is_tensor(history) or history.ndim != 2:
            raise RuntimeError(
                "M5 expects the real [batch,history_len] sas_seq integer Tensor"
            )
        if history.shape[0] != candidate.shape[0] or history.shape[1] < 1:
            raise RuntimeError(
                f"sas_seq shape {tuple(history.shape)} is incompatible with "
                f"batch={candidate.shape[0]}"
            )
        h = history.to(item_table.device).long()
        valid = h.gt(0) & h.lt(item_table.shape[0])
        if not bool((h.ge(0) & h.lt(item_table.shape[0])).all()):
            # -1 padding may be used; permit negative padding only, never a
            # positive invalid item ID.
            if bool((h >= item_table.shape[0]).any()):
                raise RuntimeError("M5 history has an item ID beyond the MF item table")
        safe_ids = h.clamp(min=0, max=item_table.shape[0]-1)
        history_vectors = item_table.index_select(
            0, safe_ids.reshape(-1)
        ).reshape(h.shape[0], h.shape[1], self.dim).detach().float()
        history_vectors = F.normalize(
            history_vectors, p=2, dim=-1, eps=1e-8
        )
        q = F.normalize(
            self.interest_queries.float(), p=2, dim=-1, eps=1e-8
        )
        attn_logits = torch.einsum("kd,btd->bkt", q, history_vectors) / TEMPERATURE
        attn_logits = attn_logits.masked_fill(
            ~valid.unsqueeze(1), -1e4
        )
        # All-empty histories: zero out the resulting attention and fall
        # back to a zero interest context, rather than creating NaNs.
        attention = torch.softmax(attn_logits, dim=-1)
        attention = attention * valid.unsqueeze(1).float()
        attention = attention / attention.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        interests = torch.einsum("bkt,btd->bkd", attention, history_vectors)
        target = F.normalize(
            candidate.detach().float(), p=2, dim=-1, eps=1e-8
        )
        match_logits = torch.einsum(
            "bkd,bd->bk",
            F.normalize(interests, p=2, dim=-1, eps=1e-8),
            target,
        ) / TEMPERATURE
        chosen = torch.softmax(match_logits, dim=-1)
        interest_context = torch.einsum("bk,bkd->bd", chosen, interests)
        return interest_context, int((~valid.any(dim=-1)).sum().item())

    def forward(self, rec, user_ids, item_ids, samples):
        user_table = rec.user_embedding.weight
        item_table = rec.item_embedding.weight
        uids = user_ids.long().to(user_table.device).reshape(-1)
        iids = item_ids.long().to(item_table.device).reshape(-1)
        if uids.numel() != iids.numel():
            raise RuntimeError("User-item paired IDs have mismatched batch")
        if (
            bool((uids < 0).any()) or bool((uids >= user_table.shape[0]).any())
            or bool((iids < 0).any()) or bool((iids >= item_table.shape[0]).any())
        ):
            raise RuntimeError("User/item ID outside MF embedding tables")
        user = user_table.index_select(0, uids).detach().float()
        item = item_table.index_select(0, iids).detach().float()
        user = F.normalize(user, p=2, dim=-1, eps=1e-8)
        item = F.normalize(item, p=2, dim=-1, eps=1e-8)
        batch = user.shape[0]
        cold = 0
        empty_hist = 0

        if self.mode == "global":
            logits = self.global_logits.view(1, EXPERTS).expand(batch, -1)
        elif self.mode == "user_cluster":
            logits, cold = self._route_cluster(uids, user)
        elif self.mode == "item_cluster":
            logits, cold = self._route_cluster(iids, item)
        elif self.mode == "pair":
            logits = self.router(torch.cat([user, item, user * item], dim=-1))
        elif self.mode == "multi_interest":
            context, empty_hist = self._multi_interest(
                item, samples.get("sas_seq"), item_table
            )
            logits = self.router(
                torch.cat([user, item, user * item, context], dim=-1)
            )
        elif self.mode == "layer_pair":
            logits = self.router(
                torch.cat([user, item, user * item], dim=-1)
            ).reshape(batch, 4, EXPERTS)
        else:
            raise RuntimeError("Unrecognized MoE mode")

        probs = torch.softmax(logits.float(), dim=-1)
        if probs.ndim == 2:
            probs = probs.unsqueeze(1).expand(-1, 4, -1)
        if tuple(probs.shape) != (batch, 4, EXPERTS):
            raise RuntimeError("MoE probability tensor must be [B,4,4]")
        # Exact identity for uniform probabilities; all rank channels
        # remain active and strength varies smoothly within [0.8,1.6].
        masks = 1.0 + RESIDUAL_LAMBDA * (EXPERTS * probs - 1.0)
        if not bool(torch.isfinite(masks).all()):
            raise FloatingPointError("Nonfinite soft expert masks")

        route_average = probs.mean(dim=(0, 1))
        balance = BALANCE_WEIGHT * (
            route_average - 1.0 / EXPERTS
        ).square().sum()
        if self.mode == "layer_pair":
            # Every layer-group participates; groups are not forced equal.
            # The balance term only controls their overall utilization.
            pass

        rank_masks = [
            masks[:, idx, :].repeat_interleave(RANK_PER, dim=-1)
            for idx in range(4)
        ]
        # All 4 soft experts are active; counts record actual usage capacity,
        # separate from diagnostic probability distribution logged below.
        counts = [[batch] * EXPERTS for _ in range(4)]
        stats = {
            "expert_counts": counts,
            "cold_users_in_batch": cold,
            "empty_histories": empty_hist,
        }
        return rank_masks, balance, stats, probs.detach()


_ORIGINAL_AUDIT = base.install_loss_audit


def install_moe_model(trial):
    """Register new routers BEFORE optimizer creation, and attach sample data."""
    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx
    from minigpt4.models.ppeft.weight_generator import PerceptualWeightsGenerator

    if getattr(MiniGPT4Rec_vx, "_collab_moe6_installed", False):
        raise RuntimeError("MoE6 model patch already installed; worker must be fresh")
    original_init = MiniGPT4Rec_vx.__init__
    original_generator_forward = PerceptualWeightsGenerator.forward
    mode = trial["expert_mode"]

    def init_with_moe(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        gen = self.llama_model_lora.base_model.weight_generator
        if int(gen.r) != RANK:
            raise RuntimeError(f"Expected original CoRA rank16, got {gen.r}")
        rec = self.rec_encoder
        if not hasattr(rec, "user_embedding") or not hasattr(rec, "item_embedding"):
            raise RuntimeError("Collaborative MoE requires MF embedding tables")
        user_dim = int(rec.user_embedding.weight.shape[-1])
        item_dim = int(rec.item_embedding.weight.shape[-1])
        if user_dim != item_dim:
            raise RuntimeError("MoE interaction routers require same-sized MF user/item vectors")
        if hasattr(gen, "rank4_experts"):
            raise RuntimeError("MoE experts already registered on weight generator")
        labels = centers = None
        if mode in CLUSTERS:
            labels, centers = _train_centroids(rec, mode)
        # Keep the original CoRA weight init AND the subsequent data-loader
        # RNG stream matched across all six trials. The additional MLP/query
        # random draws live inside a restored CPU RNG scope.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED + 104)
            gen.rank4_experts = CollaborativeMoE(
                mode, user_dim, labels=labels, centers=centers
            )
        n = sum(p.numel() for p in gen.rank4_experts.parameters() if p.requires_grad)
        print(
            f"[MOE6 INIT] name={trial['name']} mode={mode} "
            f"rank={RANK}=4x4 lambda={RESIDUAL_LAMBDA} "
            f"MF_dim={user_dim} router_trainable={n}",
            flush=True,
        )

    def routed_generator(self, ui_embedding):
        weights = original_generator_forward(self, ui_embedding)
        ctx = getattr(self, "_rankgran_ctx", None)
        if ctx is None:
            return weights
        if not hasattr(self, "rank4_experts"):
            raise RuntimeError("MoE6 module missing during weight generator forward")
        if not hasattr(self, "_moe_samples"):
            raise RuntimeError(
                "MoE6 sample context not installed; refusing to fall back to user-only routing"
            )
        if not isinstance(weights, (list, tuple)) or len(weights) != 32:
            raise RuntimeError(
                "D4 grouped generator must produce 32 layer-indexed weights"
            )
        masks, aux, stats, probs = self.rank4_experts(
            ctx["rec_encoder"], ctx["user_ids"], ctx["item_ids"],
            self._moe_samples
        )
        self._r4e_aux = aux
        self._r4e_stats = stats
        self._r4e_calls = getattr(self, "_r4e_calls", 0) + 1

        if self.training:
            p = probs.detach().float().cpu()
            n = int(p.shape[0])
            if not hasattr(self, "_moe_stats_window"):
                self._moe_stats_window = {
                    "sum": torch.zeros(4, EXPERTS),
                    "sumsq": torch.zeros(4, EXPERTS),
                    "entropy": 0.0,
                    "batch_samples": 0,
                    "empty_hist": 0,
                }
            window = self._moe_stats_window
            window["sum"] += p.sum(dim=0)
            window["sumsq"] += p.square().sum(dim=0)
            window["entropy"] += float(
                (-(p.clamp_min(1e-8)*p.clamp_min(1e-8).log()).sum(dim=-1)).sum()
            )
            window["batch_samples"] += n
            window["empty_hist"] += stats["empty_histories"]

        cache = {}
        routed = []
        for layer_idx, weight in enumerate(weights):
            group = layer_idx // 8
            if not torch.is_tensor(weight) or weight.ndim != 3 or weight.shape[-1] != RANK:
                raise RuntimeError(
                    f"Invalid CoRA dynamic weight for layer={layer_idx}: "
                    f"{getattr(weight,'shape',None)}"
                )
            key = (id(weight), group)
            if key not in cache:
                m = masks[group].unsqueeze(1).to(
                    weight.device, dtype=weight.dtype
                )
                if m.shape[0] != weight.shape[0]:
                    raise RuntimeError("MoE6 mask batch dimension mismatch")
                cache[key] = weight * m
            routed.append(cache[key])
        return routed

    MiniGPT4Rec_vx.__init__ = init_with_moe
    PerceptualWeightsGenerator.forward = routed_generator

    # RankGran already wraps both methods with its user/item ID context.
    # Supply the real model's sequence field for M5; all methods can inspect
    # the same sample metadata without altering the official CoRA source.
    for method in ("forward_v2", "generate_for_samples_v2"):
        previous_method = getattr(MiniGPT4Rec_vx, method)
        def wrap(fn):
            def with_moe_samples(self, samples, *args, **kwargs):
                generator = self.llama_model_lora.base_model.weight_generator
                if hasattr(generator, "_moe_samples"):
                    raise RuntimeError("Nested MoE6 samples context not supported")
                generator._moe_samples = samples
                try:
                    return fn(self, samples, *args, **kwargs)
                finally:
                    if hasattr(generator, "_moe_samples"):
                        delattr(generator, "_moe_samples")
            return with_moe_samples
        setattr(MiniGPT4Rec_vx, method, wrap(previous_method))

    MiniGPT4Rec_vx._collab_moe6_installed = True
    print(
        f"[MOE6 FORWARD] {trial['name']} group-aware, dense soft masks, "
        "C6/D4 paths unchanged; context includes actual sas_seq for M5",
        flush=True,
    )


def install_audit(trial):
    """Keep parent optimizer/gradient validation and add true soft route stats."""
    _ORIGINAL_AUDIT(trial)
    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx
    previous_forward = MiniGPT4Rec_vx.forward_v2

    def with_moe_logging(self, samples, *args, **kwargs):
        result = previous_forward(self, samples, *args, **kwargs)
        step = getattr(self, "_r4e_train_steps", 0)
        if step == 1 or step % 100 == 0:
            gen = self.llama_model_lora.base_model.weight_generator
            win = getattr(gen, "_moe_stats_window", None)
            if win is None or win["batch_samples"] == 0:
                raise RuntimeError("MoE6 expected nonempty routing statistics")
            n = win["batch_samples"]
            mean = win["sum"] / n
            var = (win["sumsq"] / n - mean.square()).clamp_min(0)
            entropy = win["entropy"] / (n * 4)
            print(
                f"[MOE6 ROUTE] {trial['name']} step={step} "
                f"prob_mean={mean.tolist()} "
                f"prob_std={var.sqrt().tolist()} "
                f"entropy={entropy:.6f} "
                f"history_empty={win['empty_hist']}/{n}",
                flush=True,
            )
            delattr(gen, "_moe_stats_window")
        return result

    MiniGPT4Rec_vx.forward_v2 = with_moe_logging


def configure(protocol):
    base.TRIALS = TRIALS
    base.TRIAL_BY_NAME = TRIAL_BY_NAME
    base.ORDER = ORDER
    base.SCRIPT = SCRIPT
    base.install_model = install_moe_model
    base.install_loss_audit = install_audit
    base.configure(protocol)


def plan(protocol):
    configure(protocol)
    base.plan(protocol)
    path = base.late.base.core.LOG_DIR / "collab_moe6_manifest.json"
    manifest = {
        "protocol": protocol, "gpu": "physical GPU0 serial", "seed": SEED,
        "runs": TRIALS,
        "routing": "four always-active nonoverlapping rank4 experts within existing CoRA rank16",
        "mask": "1+lambda*(4*softmax(router(context))-1)",
        "lambda": RESIDUAL_LAMBDA,
        "zero_initialized_router_output": True,
        "main_CoRA": "D4_TOPK_RES025 neighbor_mix personal_share=.25",
        "auxiliary": "C6_KD_BOTH CTR=.005 KD=.01",
        "user_cluster": "K4 trained ONLY from positive/history train_ood2.pkl user IDs",
        "item_cluster": "K4 trained ONLY from positive/history train_ood2.pkl item IDs",
        "multi_interest": "three learned latent queries over actual sas_seq MF item embeddings; target-aware interest attention",
        "layer_pair": "four independent user-item routes on layer blocks 1-8/9-16/17-24/25-32",
        "loss_extra": "1e-3 * mean_probability_balance",
        "no_test": True,
        "controls": {
            "seed": SEED, "batch_size": 2, "accum_grad_iters": 4,
            "iters_per_epoch": 400, "patience": 10,
            "cora_lr": 3e-4, "mf_lr": 3e-5,
            "pretrained_mf_and_cora_reinit_per_trial": True,
        },
        "historical_reference": {
            "S1_valid_auc": 0.732387, "S1_valid_uauc": 0.701999,
            "S4_soft4_auc": 0.733476, "S4_soft4_uauc": 0.697128,
        },
    }
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[MOE6 PLAN PASS] six independent configs; manifest={path}; no GPU work")


def cpu_test():
    torch.manual_seed(SEED)
    batch, dim = 8, 16
    from types import SimpleNamespace
    rec = SimpleNamespace(
        user_embedding=nn.Embedding(40, dim),
        item_embedding=nn.Embedding(50, dim),
    )
    users = torch.arange(1, batch + 1)
    items = torch.arange(2, batch + 2)
    hist = torch.tensor([
        [1, 3, 0, 0],
        [4, 7, 8, 0],
        [2, 9, 11, 12],
        [0, 0, 0, 0],
        [3, 8, 0, 0],
        [4, 6, 8, 0],
        [2, 10, 0, 0],
        [5, 7, 1, 0],
    ])
    samples = {"sas_seq": hist}
    labels_u = torch.full((40,), -1, dtype=torch.long)
    labels_i = torch.full((50,), -1, dtype=torch.long)
    labels_u[1:9] = torch.arange(8) % EXPERTS
    labels_i[2:10] = torch.arange(8) % EXPERTS
    centers = F.normalize(torch.randn(EXPERTS, dim), dim=-1)
    x = torch.randn(batch, 32, requires_grad=True)
    w = torch.randn(batch, 32, RANK, requires_grad=True)
    P = torch.randn(24, RANK, requires_grad=True)
    dense = F.linear(torch.bmm(x.unsqueeze(1), w).squeeze(1), P)
    assert len(TRIALS) == 6

    for trial in TRIALS:
        kind = trial["expert_mode"]
        extra = (
            dict(labels=(labels_u if kind == "user_cluster" else labels_i),
                 centers=centers)
            if kind in CLUSTERS else {}
        )
        module = CollaborativeMoE(kind, dim, **extra)
        masks, aux, stats, probabilities = module(rec, users, items, samples)
        assert len(masks) == 4
        assert tuple(probabilities.shape) == (batch, 4, EXPERTS)
        assert all(tuple(m.shape) == (batch, RANK) for m in masks)
        assert all(bool((m >= .8-1e-6).all()) for m in masks)
        assert all(bool((m <= 1.6+1e-6).all()) for m in masks)
        assert all(torch.isfinite(m).all() for m in masks)
        for mask in masks:
            torch.testing.assert_close(
                mask, torch.ones_like(mask), atol=1e-6, rtol=0
            )
        assert stats["empty_histories"] == (1 if kind == "multi_interest" else 0)
        if kind == "layer_pair":
            assert len({p.data_ptr() for p in probabilities.unbind(dim=1)}) == 4
        group_outputs = [
            F.linear(
                torch.bmm(
                    x.unsqueeze(1), w * mask.unsqueeze(1)
                ).squeeze(1), P
            )
            for mask in masks
        ]
        for output in group_outputs:
            torch.testing.assert_close(output, dense, atol=1e-5, rtol=1e-5)
        # Exercise every group, including all four M6 heads, in backward.
        loss = sum(out.square().mean() for out in group_outputs) / 4.0 + aux
        loss.backward(retain_graph=True)
        expected = list(module.parameters())
        assert expected and all(p.grad is not None for p in expected)
        assert all(bool(torch.isfinite(p.grad).all()) for p in expected)
    print(
        "[MOE6 CPU PASS] six identity-initialized full-rank soft MoE routes; "
        "user/item clusters; pair; three-interest history attention; "
        "four layer-group logits; no rank dropout; finite router gradients",
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
    parser.add_argument("--patience", type=int, default=base.PATIENCE)
    parser.add_argument("--screen-epochs", type=int, default=0)
    parser.add_argument("--_worker", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(base.PROJECT_ROOT)
    if args.cpu_test:
        cpu_test()
        return
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.protocol):
        parser.error("Protocol accepts letters, digits, '_' and '-' only")
    if args.patience < 1 or args.screen_epochs < 0:
        parser.error("patience >=1 and screen-epochs >=0 required")
    configure(args.protocol)
    if args._worker:
        base.worker(args._worker, args.protocol, args.patience, args.screen_epochs)
        return
    if args.plan:
        plan(args.protocol)
        return
    if args.summary:
        from cora_collab_moe6_summary import run_summary
        run_summary(args.protocol, debug=args.debug)
        return
    if args.run:
        selected = (
            [s.strip() for s in args.only.split(",") if s.strip()]
            if args.only else ORDER
        )
        if len(selected) != len(set(selected)) or any(s not in TRIAL_BY_NAME for s in selected):
            parser.error("--only contains unknown or duplicate trial names")
        if not base.late.base.core.CONFIG_DIR.is_dir():
            plan(args.protocol)
        base.run_selected(selected, args.protocol, args.patience, args.screen_epochs)
        print("[MOE6 COMPLETE] selected variants completed; VALID only", flush=True)
        return
    parser.error("Specify --cpu-test, --plan, --run or --summary")


if __name__ == "__main__":
    main()
