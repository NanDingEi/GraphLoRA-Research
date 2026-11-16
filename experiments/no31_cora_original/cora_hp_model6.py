#!/usr/bin/env python3
"""Six single-model architecture improvements on the hierarchical CoRA stack.

Ports the HCP idea (3 hierarchical group experts + 1 raw-personal expert)
onto the proven D4_TOPK_RES025 + C6_KD_BOTH + rank16-generator runtime that
produced the historical bests (E2_USER_L10 valid 0.7316 / RR1 test 0.7397).

Six architecture variants share one pretrained 7B base and one fresh
rank16 CoRA model per trial:
    M1: user-conditioned mixture of T=0.25 and T=0.50 routing distributions.
    M2: user-conditioned temperature in [0.25, 0.50].
    M3: user-conditioned coarse/fine blend beta in [0.50, 0.90].
    M4: candidate-aware personal-expert logit from [user, item, user*item].
    M5: one personal-expert logit per 8-layer group.
    M6: expert-specific affine adapters on the four Rank4 slices.

The shared routing output remains four expert probabilities per sample and
four layer groups. D4 grouped inputs, C6 losses, Joint8 LR split, seed42,
GPU0 serial execution, and VALID-only model selection are inherited.

Run from the CoRA_official root:
  python experiments/no31_cora_original/cora_hp_model6.py --cpu-test
  python experiments/no31_cora_original/cora_hp_model6.py --plan
  CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/cora_hp_model6.py --run
  python experiments/no31_cora_original/cora_hp_model6.py --summary
"""

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cora_collab_moe6 as moe
import cora_route_factorial6 as fac
import cora_residual_routing6 as r6mod

DEFAULT_PROTOCOL = "CORA_HP_MODEL6_20261116"
SEED = 42
RANK = 16
EXPERTS = 4
RANK_PER = 4
K1 = 3
K2_PER_GROUP = 8
ROUTE_TEMP = 0.25
PERSONAL_INIT_LOGIT = math.log(2.0)
FINE_TEMP = 0.2
COARSE_BETA = 0.5
BALANCE_WEIGHT = 1e-3

TRIALS = [
    dict(name="M1_DUAL_TEMP", expert_mode="hier_personal",
         model_variant="dual_temp", route="hier_personal", route_lambda=0.2,
         route_temp=0.50, init_temp=0.50, personal_target=0.40,
         route_fine=True, beta=0.70, mode="neighbor_mix", personal_share=0.25),
    dict(name="M2_USER_TEMP", expert_mode="hier_personal",
         model_variant="adaptive_temp", route="hier_personal", route_lambda=0.2,
         route_temp=0.50, init_temp=0.47, personal_target=0.40,
         route_fine=True, beta=0.70, mode="neighbor_mix", personal_share=0.25),
    dict(name="M3_USER_BETA", expert_mode="hier_personal",
         model_variant="adaptive_beta", route="hier_personal", route_lambda=0.2,
         route_temp=0.50, init_temp=0.50, personal_target=0.40,
         route_fine=True, beta=0.70, mode="neighbor_mix", personal_share=0.25),
    dict(name="M4_PAIR_PERSONAL", expert_mode="hier_personal",
         model_variant="pair_personal", route="hier_personal", route_lambda=0.2,
         route_temp=0.50, init_temp=0.50, personal_target=0.40,
         route_fine=True, beta=0.70, mode="neighbor_mix", personal_share=0.25),
    dict(name="M5_LAYER_PERSONAL", expert_mode="hier_personal",
         model_variant="layer_personal", route="hier_personal", route_lambda=0.2,
         route_temp=0.50, init_temp=0.50, personal_target=0.40,
         route_fine=True, beta=0.70, route_variant="group_personal",
         mode="neighbor_mix", personal_share=0.25),
    dict(name="M6_HETERO_ADAPTER", expert_mode="hier_personal",
         model_variant="hetero_adapter", route="hier_personal", route_lambda=0.2,
         route_temp=0.50, init_temp=0.50, personal_target=0.40,
         route_fine=True, beta=0.70, mode="neighbor_mix", personal_share=0.25),
]
ORDER = [t["name"] for t in TRIALS]
TRIAL_BY_NAME = {t["name"]: t for t in TRIALS}
LOG_LINE = re.compile(r"\*\*\*auc:\s*([0-9.]+)\s+\*\*\*uauc:\s*\(?([0-9.]+)")


def _spherical_kmeans(x, k, seed, iterations=15):
    n = int(x.shape[0])
    if not 1 <= k < n:
        raise ValueError(f"k={k} invalid for n={n}")
    gen = torch.Generator().manual_seed(seed)
    centers = [x[int(torch.randint(n, (1,), generator=gen).item())].clone()]
    closest = (1.0 - x @ centers[0]).clamp_min(0)
    for _ in range(1, k):
        probs = closest.square()
        if not bool(torch.isfinite(probs).all()) or float(probs.sum()) <= 1e-12:
            idx = int(torch.randint(n, (1,), generator=gen).item())
        else:
            idx = int(torch.multinomial(probs, 1, generator=gen).item())
        centers.append(x[idx].clone())
        closest = torch.minimum(closest, (1.0 - x @ centers[-1]).clamp_min(0))
    centers = torch.stack(centers, dim=0)
    for _ in range(iterations):
        labels = (x @ centers.T).argmax(dim=1)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, labels, x)
        counts = torch.bincount(labels, minlength=k).to(x.dtype)
        updated = sums / counts.clamp_min(1.0).unsqueeze(-1)
        norm = updated.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        centers = torch.where((counts > 0).unsqueeze(-1), updated / norm, centers)
    labels = (x @ centers.T).argmax(dim=1)
    return centers, labels


def hierarchical_centers(user_table, uid_lo=1):
    rows = user_table[uid_lo:].detach().float()
    if rows.shape[0] < 256:
        raise RuntimeError("Too few user rows for hierarchical clustering")
    x = F.normalize(rows, p=2, dim=-1, eps=1e-8)
    coarse, coarse_labels = _spherical_kmeans(x, K1, SEED + 501)
    fine_list, parents = [], []
    for g in range(K1):
        members = x[coarse_labels == g]
        if int(members.shape[0]) < K2_PER_GROUP:
            fine = members.clone()
            while fine.shape[0] < K2_PER_GROUP:
                fine = torch.cat([fine, members[:1]], dim=0)
        else:
            fine, _ = _spherical_kmeans(members, K2_PER_GROUP, SEED + 601 + g)
        fine_list.append(fine)
        parents.append(torch.full((int(fine.shape[0]),), g, dtype=torch.long))
    fine = torch.cat(fine_list, dim=0)
    parent = torch.cat(parents, dim=0)
    if bool((torch.bincount(parent, minlength=K1) == 0).any()):
        raise RuntimeError("Empty coarse group")
    return coarse, fine, parent


class HierPersonalMoE(moe.CollaborativeMoE):
    """Hierarchical cluster/personal routing with optional learned model variants.

    Non-hier modes delegate to the original CollaborativeMoE unchanged, so
    every historical trial remains reproducible inside this worker.
    """

    def __init__(self, mode, dim, labels=None, centers=None,
                 coarse=None, fine=None, parent=None,
                 personal_logit_init=PERSONAL_INIT_LOGIT,
                 route_temp=None, route_fine=False, beta=COARSE_BETA,
                 hybrid_residual=False, route_variant="shared",
                 model_variant="baseline"):
        if mode != "hier_personal":
            super().__init__(mode, dim, labels=labels, centers=centers)
            return
        nn.Module.__init__(self)
        self.mode = "hier_personal"
        self.dim = int(dim)
        self.route_temp = float(route_temp if route_temp is not None else ROUTE_TEMP)
        self.route_fine = bool(route_fine)
        self.beta = float(beta)
        self.model_variant = str(model_variant)
        allowed = {"baseline", "dual_temp", "adaptive_temp", "adaptive_beta",
                   "pair_personal", "layer_personal", "hetero_adapter"}
        if self.model_variant not in allowed:
            raise ValueError(f"Unknown model_variant: {self.model_variant}")
        self._grad_seen = set()
        self._grad_nonfinite = set()
        if coarse is None or fine is None or parent is None:
            raise RuntimeError("hier_personal requires precomputed centers")
        if int(coarse.shape[0]) != K1 or int(fine.shape[0]) != K1 * K2_PER_GROUP:
            raise RuntimeError("Cluster geometry mismatch")
        self.register_buffer("coarse_centers", coarse.detach().clone().float())
        self.register_buffer("fine_centers", fine.detach().clone().float())
        self.register_buffer("fine_parent", parent.detach().clone())
        self.route_variant = str(route_variant)
        if self.route_variant not in ("shared", "group_personal"):
            raise ValueError(f"Unknown route_variant: {self.route_variant}")
        if self.model_variant == "dual_temp":
            if not isinstance(personal_logit_init, (tuple, list, torch.Tensor)) or len(personal_logit_init) != 2:
                raise ValueError("dual_temp requires [personal_logit_T025, personal_logit_T050]")
            self.personal_logit = nn.Parameter(torch.as_tensor(
                personal_logit_init, dtype=torch.float32).clone())
        elif self.route_variant == "group_personal":
            init = float(personal_logit_init)
            self.personal_logit = nn.Parameter(torch.full((4,), init))
        else:
            self.personal_logit = nn.Parameter(torch.tensor(float(personal_logit_init)))
        self.hybrid_residual = bool(hybrid_residual)

        # Small user-conditioned routing heads. Zero/controlled initialization
        # keeps each trial close to its stated HP24-derived starting point.
        if self.model_variant == "dual_temp":
            self.dual_alpha = nn.Sequential(
                nn.Linear(self.dim, 64), nn.GELU(), nn.Linear(64, 1))
            nn.init.zeros_(self.dual_alpha[-1].weight)
            nn.init.zeros_(self.dual_alpha[-1].bias)
        if self.model_variant == "adaptive_temp":
            self.temp_gate = nn.Sequential(
                nn.Linear(self.dim, 64), nn.GELU(), nn.Linear(64, 1))
            nn.init.zeros_(self.temp_gate[-1].weight)
            nn.init.constant_(self.temp_gate[-1].bias, 2.0)
        if self.model_variant == "adaptive_beta":
            self.beta_gate = nn.Sequential(
                nn.Linear(self.dim, 64), nn.GELU(), nn.Linear(64, 1))
            nn.init.zeros_(self.beta_gate[-1].weight)
            nn.init.zeros_(self.beta_gate[-1].bias)
        if self.model_variant == "pair_personal":
            self.pair_gate = nn.Sequential(
                nn.Linear(3 * self.dim, 64), nn.GELU(), nn.Linear(64, 1))
            nn.init.zeros_(self.pair_gate[-1].weight)
            nn.init.zeros_(self.pair_gate[-1].bias)
        if self.model_variant == "hetero_adapter":
            # Four expert-specific affine adapters over the four Rank4 slices,
            # separately parameterized for each of the four layer groups.
            self.expert_gain = nn.Parameter(torch.ones(4, EXPERTS, RANK_PER))
            self.expert_bias = nn.Parameter(torch.zeros(4, EXPERTS, RANK_PER))
        if self.hybrid_residual:
            self.route_residual = nn.Linear(self.dim, K1 + 1)
            nn.init.zeros_(self.route_residual.weight)
            nn.init.zeros_(self.route_residual.bias)
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.register_hook(self._grad_hook(name))

    def apply_rank_adapter(self, weight, layer_group):
        if self.model_variant != "hetero_adapter":
            return weight
        group = int(layer_group)
        if not 0 <= group < 4 or weight.shape[-1] != RANK:
            raise RuntimeError("Invalid heterogeneous rank-adapter input")
        chunks = []
        for expert in range(EXPERTS):
            lo, hi = expert * RANK_PER, (expert + 1) * RANK_PER
            gain = self.expert_gain[group, expert].to(
                device=weight.device, dtype=weight.dtype)
            bias = self.expert_bias[group, expert].to(
                device=weight.device, dtype=weight.dtype)
            chunks.append(weight[..., lo:hi] * gain + bias)
        return torch.cat(chunks, dim=-1)

    def forward(self, rec, user_ids, item_ids, samples):
        if self.mode != "hier_personal":
            return super().forward(rec, user_ids, item_ids, samples)
        user_table = rec.user_embedding.weight
        item_table = rec.item_embedding.weight
        uids = user_ids.long().to(user_table.device).reshape(-1)
        iids = item_ids.long().to(item_table.device).reshape(-1)
        if bool((uids < 0).any()) or bool((uids >= user_table.shape[0]).any()):
            raise RuntimeError("User ID outside MF embedding table")
        if uids.numel() != iids.numel():
            raise RuntimeError("User/item paired IDs have mismatched batch")
        user = F.normalize(user_table.index_select(0, uids).detach().float(),
                           p=2, dim=-1, eps=1e-8)
        batch = user.shape[0]

        beta_user = None
        if self.model_variant == "adaptive_beta":
            beta_user = 0.5 + 0.4 * torch.sigmoid(self.beta_gate(user))
        if self.route_fine:
            cos2 = user @ self.fine_centers.T
            vecs = []
            for g in range(int(self.coarse_centers.shape[0])):
                member = (self.fine_parent == g).float()
                s2 = F.softmax(cos2 / FINE_TEMP + (member - 1.0) * 1e4, dim=-1)
                v_fine = s2 @ self.fine_centers
                if beta_user is None:
                    beta_value = self.beta
                else:
                    beta_value = beta_user
                v_g = beta_value * self.coarse_centers[g].unsqueeze(0) + \
                    (1.0 - beta_value) * v_fine
                vecs.append(F.normalize(v_g, p=2, dim=-1, eps=1e-8))
            V = torch.stack(vecs, dim=1)
            cos1 = (user.unsqueeze(1) * V).sum(dim=-1)
        else:
            cos1 = user @ self.coarse_centers.T

        if self.model_variant == "dual_temp":
            p25 = torch.softmax(torch.cat((
                cos1 / 0.25, self.personal_logit[0].expand(batch, 1)), dim=-1).float(), dim=-1)
            p50 = torch.softmax(torch.cat((
                cos1 / 0.50, self.personal_logit[1].expand(batch, 1)), dim=-1).float(), dim=-1)
            alpha = torch.sigmoid(self.dual_alpha(user))
            probs = alpha * p25 + (1.0 - alpha) * p50
        elif self.route_variant == "group_personal":
            pl = self.personal_logit.unsqueeze(0).expand(batch, -1)
            group_logits = torch.cat((
                torch.stack([cos1 / self.route_temp] * 4, dim=1),
                pl.unsqueeze(-1)), dim=-1)
            if self.hybrid_residual:
                group_logits = group_logits + self.route_residual(user)
            probs = torch.softmax(group_logits.float(), dim=-1)
        else:
            if self.model_variant == "adaptive_temp":
                route_temp = 0.25 + 0.25 * torch.sigmoid(self.temp_gate(user))
            else:
                route_temp = self.route_temp
            personal_logit = self.personal_logit.expand(batch, 1)
            if self.model_variant == "pair_personal":
                if bool((iids < 0).any()) or bool((iids >= item_table.shape[0]).any()):
                    raise RuntimeError("Item ID outside MF embedding table")
                item = F.normalize(item_table.index_select(0, iids).detach().float(),
                                   p=2, dim=-1, eps=1e-8)
                pair = torch.cat((user, item, user * item), dim=-1)
                personal_logit = personal_logit + 0.1 * torch.tanh(self.pair_gate(pair))
            logits = torch.cat((cos1 / route_temp, personal_logit), dim=-1)
            if self.hybrid_residual:
                logits = logits + self.route_residual(user)
            probs = torch.softmax(logits.float(), dim=-1)

        if self.route_variant == "group_personal":
            if tuple(probs.shape) != (batch, 4, EXPERTS):
                raise RuntimeError("group_personal router must produce [B,4,4]")
            probs4 = probs
        else:
            if tuple(probs.shape) != (batch, EXPERTS):
                raise RuntimeError("Router must produce [B,4] probabilities")
            probs4 = probs.unsqueeze(1).expand(-1, 4, -1)
        masks = 1.0 + moe.RESIDUAL_LAMBDA * (EXPERTS * probs4 - 1.0)
        if not bool(torch.isfinite(masks).all()):
            raise FloatingPointError("Nonfinite soft expert masks")
        route_average = (probs4.mean(dim=0) if self.route_variant == "group_personal"
                         else probs.mean(dim=0))
        balance = moe.BALANCE_WEIGHT * (route_average - 1.0 / EXPERTS).square().sum()
        rank_masks = [
            masks[:, idx, :].repeat_interleave(RANK_PER, dim=-1) for idx in range(4)
        ]
        stats = {"expert_counts": [[batch] * EXPERTS for _ in range(4)],
                 "cold_users_in_batch": 0, "empty_histories": 0}
        return rank_masks, balance, stats, probs4.detach()



def install_hier_model(trial):
    """Same construction order/patch sites as collab_moe6.install_moe_model,
    with the expert module replaced by HierPersonalMoE."""
    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx
    from minigpt4.models.ppeft.weight_generator import PerceptualWeightsGenerator

    if getattr(MiniGPT4Rec_vx, "_collab_moe6_installed", False):
        raise RuntimeError("MoE model patch already installed; worker must be fresh")
    original_init = MiniGPT4Rec_vx.__init__
    original_generator_forward = PerceptualWeightsGenerator.forward
    mode = trial["expert_mode"]
    if mode != "hier_personal":
        raise RuntimeError("This installer only handles hier_personal")

    def init_with_hier(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        gen = self.llama_model_lora.base_model.weight_generator
        if int(gen.r) != RANK:
            raise RuntimeError(f"Expected original CoRA rank16, got {gen.r}")
        rec = self.rec_encoder
        if not hasattr(rec, "user_embedding") or not hasattr(rec, "item_embedding"):
            raise RuntimeError("Hier MoE requires MF embedding tables")
        user_dim = int(rec.user_embedding.weight.shape[-1])
        item_dim = int(rec.item_embedding.weight.shape[-1])
        if user_dim != item_dim:
            raise RuntimeError("MF user/item dims differ")
        if hasattr(gen, "rank4_experts"):
            raise RuntimeError("MoE experts already registered")
        route_temp = float(trial.get("route_temp", ROUTE_TEMP))
        init_temp = float(trial.get("init_temp", route_temp))
        personal_target = float(trial.get("personal_target", 0.40))
        route_fine = bool(trial.get("route_fine", False))
        beta = float(trial.get("beta", COARSE_BETA))
        model_variant = str(trial.get("model_variant", "baseline"))
        route_variant = str(trial.get("route_variant", "shared"))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED + 104)
            table_cpu = rec.user_embedding.weight.detach().cpu()
            coarse, fine, parent = hierarchical_centers(table_cpu)
            user_n = F.normalize(table_cpu[1:], p=2, dim=-1, eps=1e-8)
            cos_all = user_n @ coarse.T
            ps = min(max(personal_target, 0.05), 0.95)
            log_odds = math.log(ps / (1.0 - ps))
            if model_variant == "dual_temp":
                z25 = float((cos_all / 0.25).exp().sum(dim=-1).mean())
                z50 = float((cos_all / 0.50).exp().sum(dim=-1).mean())
                pl_init = [log_odds + math.log(max(z25, 1e-12)),
                           log_odds + math.log(max(z50, 1e-12))]
            else:
                z_mean = float((cos_all / init_temp).exp().sum(dim=-1).mean())
                pl_init = log_odds + math.log(max(z_mean, 1e-12))
            gen.rank4_experts = HierPersonalMoE(
                "hier_personal", user_dim, coarse=coarse, fine=fine,
                parent=parent, personal_logit_init=pl_init,
                route_temp=route_temp, route_fine=route_fine, beta=beta,
                hybrid_residual=bool(trial.get("hybrid_residual", False)),
                route_variant=route_variant, model_variant=model_variant)
        n = sum(p.numel() for p in gen.rank4_experts.parameters() if p.requires_grad)
        print(f"[HP MODEL6 INIT] {trial['name']} variant={model_variant} "
              f"lambda={moe.RESIDUAL_LAMBDA} K1={K1} K2={K2_PER_GROUP} "
              f"T={route_temp} init_T={init_temp} ps={personal_target} "
              f"fine={route_fine} beta={beta} route_variant={route_variant} "
              f"router_trainable={n}", flush=True)

    def routed_generator(self, ui_embedding):
        weights = original_generator_forward(self, ui_embedding)
        ctx = getattr(self, "_rankgran_ctx", None)
        if ctx is None:
            return weights
        if not hasattr(self, "rank4_experts"):
            raise RuntimeError("MoE module missing during generator forward")
        if not hasattr(self, "_moe_samples"):
            raise RuntimeError("MoE sample context not installed")
        if not isinstance(weights, (list, tuple)) or len(weights) != 32:
            raise RuntimeError("D4 grouped generator must produce 32 weights")
        masks, aux, stats, probs = self.rank4_experts(
            ctx["rec_encoder"], ctx["user_ids"], ctx["item_ids"],
            self._moe_samples)
        self._r4e_aux = aux
        self._r4e_stats = stats
        self._r4e_calls = getattr(self, "_r4e_calls", 0) + 1
        if self.training:
            # Same routing-statistics window as collab_moe6's generator; the
            # inherited loss audit requires it at step 1/100.
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
                (-(p.clamp_min(1e-8) * p.clamp_min(1e-8).log()).sum(dim=-1)).sum()
            )
            window["batch_samples"] += n
            window["empty_hist"] += stats["empty_histories"]
        cache = {}
        routed = []
        for layer_idx, weight in enumerate(weights):
            group = layer_idx // 8
            if not torch.is_tensor(weight) or weight.ndim != 3 or \
                    weight.shape[-1] != RANK:
                raise RuntimeError(f"Invalid dynamic weight layer={layer_idx}")
            key = (id(weight), group)
            if key not in cache:
                routed_weight = weight
                if trial.get("model_variant") == "hetero_adapter":
                    routed_weight = self.rank4_experts.apply_rank_adapter(weight, group)
                m = masks[group].unsqueeze(1).to(weight.device, dtype=weight.dtype)
                if m.shape[0] != weight.shape[0]:
                    raise RuntimeError("Mask batch mismatch")
                cache[key] = routed_weight * m
            routed.append(cache[key])
        return routed

    MiniGPT4Rec_vx.__init__ = init_with_hier
    PerceptualWeightsGenerator.forward = routed_generator

    for method in ("forward_v2", "generate_for_samples_v2"):
        previous_method = getattr(MiniGPT4Rec_vx, method)

        def wrap(fn):
            def with_moe_samples(self, samples, *args, **kwargs):
                generator = self.llama_model_lora.base_model.weight_generator
                if hasattr(generator, "_moe_samples"):
                    raise RuntimeError("Nested MoE samples context not supported")
                generator._moe_samples = samples
                try:
                    return fn(self, samples, *args, **kwargs)
                finally:
                    if hasattr(generator, "_moe_samples"):
                        delattr(generator, "_moe_samples")
            return with_moe_samples

        setattr(MiniGPT4Rec_vx, method, wrap(previous_method))
    MiniGPT4Rec_vx._collab_moe6_installed = True
    print(f"[HIER FORWARD] {trial['name']} cluster+personal soft masks; "
          "D4/C6 paths unchanged", flush=True)


def install_model(trial):
    lam = float(trial["route_lambda"])
    if lam not in (0.2, 0.35, 1.0):
        raise RuntimeError(f"Unapproved lambda: {lam}")
    moe.RESIDUAL_LAMBDA = lam
    fac._install_grad_norm_hooks()
    install_hier_model(trial)
    if trial.get("logit_gate"):
        # L2_DUALGATE: stack the proven RR5 logit-space user gate
        # (s = z32 + sigmoid(g(u))*(z31-z32)) on top of the expert-mask
        # routing. The fusion module is created at model init (inside this
        # call chain) so its parameters exist before optimizer construction.
        from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx
        if getattr(MiniGPT4Rec_vx, "_r6_init_patched", False) or \
                getattr(MiniGPT4Rec_vx, "_r6_forward_patched", False):
            raise RuntimeError("R6 fusion already installed in this worker")
        r6_trial = dict(name=trial["name"], kind="user_gate")
        r6mod.install_trial_module(r6_trial)
        r6mod.install_optimizer_audit(r6_trial)
        r6mod.install_trial_forward(r6_trial)
        print(f"[HIER MODEL] {trial['name']} lambda={lam} rank=16=4x4 + "
              "LOGIT user_gate (dual-gate); D4/C6 unchanged", flush=True)
    else:
        print(f"[HIER MODEL] {trial['name']} lambda={lam} rank=16=4x4 "
              "D4/C6 unchanged", flush=True)


def configure(protocol):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", protocol):
        raise ValueError("Unsafe protocol name")
    runner = moe.base
    runner.TRIALS = TRIALS
    runner.TRIAL_BY_NAME = TRIAL_BY_NAME
    runner.ORDER = ORDER
    runner.SCRIPT = SCRIPT
    runner.install_model = install_model
    runner.install_loss_audit = fac.install_audit
    runner.configure(protocol)


def plan(protocol):
    configure(protocol)
    moe.base.plan(protocol)
    root = HERE / "logs" / protocol
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": protocol, "seed": SEED,
        "hypothesis": "Six single-model architecture variants: dual/adaptive "
                      "temperature, adaptive coarse-fine fusion, pair-aware "
                      "personal routing, layer-specific personal routing, and "
                      "expert-specific rank-slice adapters.",
        "trials": TRIALS,
        "clustering": {"K1": K1, "K2_per_group": K2_PER_GROUP,
                       "source": "initial pretrained MF user table; frozen buffers"},
        "routing": {"variants": [t["model_variant"] for t in TRIALS],
                    "base_temperature": 0.50, "dual_temperatures": [0.25, 0.50],
                    "adaptive_temperature_range": [0.25, 0.50],
                    "adaptive_beta_range": [0.50, 0.90],
                    "personal_init_share": 0.40, "balance": BALANCE_WEIGHT},
        "controls": {"D4": "TOPK_RES025; personal_share=.25",
                     "C6": "CTR=.005 KD=.010", "cora_lr": 3e-4,
                     "mf_lr_ratio": 0.1, "batch": 2, "accum": 4,
                     "iters": 400, "patience": 10, "valid_only": True},
    }
    (root / "hp_model6_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[HP MODEL6 PLAN PASS] manifest={root / 'hp_model6_manifest.json'}",
          flush=True)


def cpu_test():
    torch.manual_seed(SEED)
    dim, batch = 16, 8
    rec = type("MF", (), {"user_embedding": nn.Embedding(300, dim),
                          "item_embedding": nn.Embedding(400, dim)})()
    with torch.no_grad():
        rec.user_embedding.weight.normal_()
        rec.item_embedding.weight.normal_()
    coarse, fine, parent = hierarchical_centers(rec.user_embedding.weight)
    user_ids = torch.arange(1, batch + 1)
    item_ids = torch.arange(2, batch + 2)
    variants = [
        ("M1_DUAL_TEMP", "dual_temp", "shared", [math.log(2.0), math.log(2.0)]),
        ("M2_USER_TEMP", "adaptive_temp", "shared", math.log(2.0)),
        ("M3_USER_BETA", "adaptive_beta", "shared", math.log(2.0)),
        ("M4_PAIR_PERSONAL", "pair_personal", "shared", math.log(2.0)),
        ("M5_LAYER_PERSONAL", "layer_personal", "group_personal", math.log(2.0)),
        ("M6_HETERO_ADAPTER", "hetero_adapter", "shared", math.log(2.0)),
    ]
    for name, variant, route_variant, pl_init in variants:
        moe.RESIDUAL_LAMBDA = 0.2
        model = HierPersonalMoE(
            "hier_personal", dim, coarse=coarse, fine=fine, parent=parent,
            personal_logit_init=pl_init, route_temp=0.50, route_fine=True,
            beta=0.70, route_variant=route_variant, model_variant=variant)
        masks, aux, stats, probs = model(rec, user_ids, item_ids, {})
        assert len(masks) == 4 and tuple(masks[0].shape) == (batch, RANK)
        assert tuple(probs.shape) == (batch, 4, 4), (name, tuple(probs.shape))
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(batch, 4),
                                   atol=1e-5, rtol=1e-5)
        signal = torch.randn(batch, RANK)
        loss = (signal * masks[0]).sum() + aux
        loss.backward()
        assert torch.isfinite(loss)
        if variant == "hetero_adapter":
            weight = torch.randn(batch, 2, RANK, requires_grad=True)
            adapted = model.apply_rank_adapter(weight, 0)
            torch.testing.assert_close(adapted, weight, atol=0, rtol=0)
            adapted.square().mean().backward()
            assert model.expert_gain.grad is not None
            assert model.expert_bias.grad is not None
        print(f"[HP MODEL6 CPU] {name} shape={tuple(probs.shape)} "
              f"router_params={sum(p.numel() for p in model.parameters())} PASS",
              flush=True)
    print("[HP MODEL6 CPU PASS] six routing variants, paired IDs, layer routing, "
          "rank adapters, probability normalization and gradients", flush=True)


def summary(protocol, debug=False):
    log_dir = HERE / "logs" / protocol
    out_dir = HERE / "out" / protocol
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "hp_model6_valid_summary.csv"
    rows = []
    for name in ORDER:
        log = log_dir / f"{name}.log"
        if not log.is_file():
            rows.append(dict(trial=name, valid_auc="", valid_uauc="",
                             best_epoch="", total_epochs="", status="NO_LOG"))
            continue
        text = log.read_text(encoding="utf-8", errors="replace")
        pairs = [(float(a), float(u)) for a, u in LOG_LINE.findall(text)]
        if not pairs:
            rows.append(dict(trial=name, valid_auc="", valid_uauc="",
                             best_epoch="", total_epochs="", status="NO_METRICS"))
            continue
        best_i = max(range(len(pairs)), key=lambda i: pairs[i][0])
        stopped = "[JOINT8 EARLY STOP]" in text
        rows.append(dict(trial=name,
                         valid_auc=f"{pairs[best_i][0]:.9f}",
                         valid_uauc=f"{pairs[best_i][1]:.9f}",
                         best_epoch=best_i, total_epochs=len(pairs),
                         status="EARLY_STOP" if stopped else "CHECK"))
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[HP MODEL6 SUMMARY] {target}")
    for row in rows:
        print(row["trial"], row["valid_auc"], row["valid_uauc"],
              f"ep={row['best_epoch']}/{row['total_epochs']}", row["status"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-test", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summary", action="store_true")
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
        parser.error("patience>=1 and screen-epochs>=0 required")
    configure(args.protocol)
    if args._worker:
        moe.base.worker(args._worker, args.protocol, args.patience,
                        args.screen_epochs)
        return
    if args.plan:
        plan(args.protocol)
        return
    if args.summary:
        summary(args.protocol)
        return
    if args.run:
        chosen = ([x.strip() for x in args.only.split(",") if x.strip()]
                  if args.only else ORDER)
        if len(chosen) != len(set(chosen)) or any(
                x not in TRIAL_BY_NAME for x in chosen):
            parser.error("--only has unknown or duplicated names")
        if not moe.base.late.base.core.CONFIG_DIR.is_dir():
            plan(args.protocol)
        moe.base.run_selected(chosen, args.protocol, args.patience,
                              args.screen_epochs)
        print("[HP MODEL6 COMPLETE] selected trials finished. Valid only.", flush=True)
        return
    parser.error("Specify --cpu-test, --plan, --run or --summary")


if __name__ == "__main__":
    main()
