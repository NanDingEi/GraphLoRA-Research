#!/usr/bin/env python3
"""Hierarchical-cluster + personal expert routing on the FULL CoRA stack.

Ports the HCP idea (3 hierarchical group experts + 1 raw-personal expert)
onto the proven D4_TOPK_RES025 + C6_KD_BOTH + rank16-generator runtime that
produced the historical bests (E2_USER_L10 valid 0.7316 / RR1 test 0.7397).

Heterogeneous router (user-only, same output contract as user_soft4):
    coarse centers c1_g (K1=3, spherical kmeans on INITIAL pretrained MF
    user table, frozen buffers) and fine centers c2 per group (K2=8):
        logits = [cos(u, c1_g)/T for g=0..2, personal_logit]
        probs  = softmax(logits)                      # [B, 4]
        masks  = 1 + lambda*(4*probs - 1)             # shared across layer groups
        rank_masks = repeat_interleave(probs, 4)      # [B, 16] per group
    personal_logit init log(2) -> p_personal ~ 0.40 (G1_CLUSTER_K32 lesson:
    pure cluster routing collapsed UAUC; the personal expert keeps a share).
    balance = 1e-3 * sum((mean probs - 0.25)^2).

Everything else (D4 grouped inputs, C6 losses, Joint8 LR split, compact
atomic checkpoints, seed42, GPU0 serial, VALID only) is inherited unchanged.

Run from the CoRA_official root:
  python experiments/no31_cora_original/cora_hier_fullstack.py --cpu-test
  python experiments/no31_cora_original/cora_hier_fullstack.py --plan
  CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/cora_hier_fullstack.py --run
  python experiments/no31_cora_original/cora_hier_fullstack.py --summary
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

DEFAULT_PROTOCOL = "CORA_HIER_FULLSTACK_20261113"
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
    dict(name="HP1_HIER_L10", expert_mode="hier_personal",
         route="hier_personal", route_lambda=1.0,
         route_temp=0.25, personal_target=0.40, route_fine=False, beta=0.5,
         mode="neighbor_mix", personal_share=0.25),
    dict(name="HP2_HIER_L02", expert_mode="hier_personal",
         route="hier_personal", route_lambda=0.2,
         route_temp=0.25, personal_target=0.40, route_fine=False, beta=0.5,
         mode="neighbor_mix", personal_share=0.25),
    dict(name="HP3_T05_L10", expert_mode="hier_personal",
         route="hier_personal", route_lambda=0.2,
         route_temp=0.5, personal_target=0.40, route_fine=False, beta=0.5,
         mode="neighbor_mix", personal_share=0.25),
    dict(name="HP4_P55_L10", expert_mode="hier_personal",
         route="hier_personal", route_lambda=0.2,
         route_temp=0.25, personal_target=0.55, route_fine=False, beta=0.5,
         mode="neighbor_mix", personal_share=0.25),
    dict(name="HP5_VROUTE_L10", expert_mode="hier_personal",
         route="hier_personal", route_lambda=0.2,
         route_temp=0.25, personal_target=0.40, route_fine=True, beta=0.5,
         mode="neighbor_mix", personal_share=0.25),
    dict(name="HP7_HYBRID_L02", expert_mode="hier_personal",
         route="hier_personal", route_lambda=0.2,
         route_temp=0.25, personal_target=0.40, route_fine=False,
         hybrid_residual=True,
         mode="neighbor_mix", personal_share=0.25),
    dict(name="HP11_FINE_B03", expert_mode="hier_personal",
         route="hier_personal", route_lambda=0.2,
         route_temp=0.25, personal_target=0.40, route_fine=True, beta=0.3,
         mode="neighbor_mix", personal_share=0.25),
    dict(name="HP12_FINE_B07", expert_mode="hier_personal",
         route="hier_personal", route_lambda=0.2,
         route_temp=0.25, personal_target=0.40, route_fine=True, beta=0.7,
         mode="neighbor_mix", personal_share=0.25),
    dict(name="HP13_FINE_HYBRID", expert_mode="hier_personal",
         route="hier_personal", route_lambda=0.2,
         route_temp=0.25, personal_target=0.40, route_fine=True, beta=0.5,
         hybrid_residual=True,
         mode="neighbor_mix", personal_share=0.25),
    dict(name="HP15_FINE_T05", expert_mode="hier_personal",
         route="hier_personal", route_lambda=0.2,
         route_temp=0.5, personal_target=0.40, route_fine=True, beta=0.5,
         mode="neighbor_mix", personal_share=0.25),
    dict(name="HP17_FINE_B08", expert_mode="hier_personal",
         route="hier_personal", route_lambda=0.2,
         route_temp=0.25, personal_target=0.40, route_fine=True, beta=0.8,
         mode="neighbor_mix", personal_share=0.25),
    dict(name="HP18_FINE_B06", expert_mode="hier_personal",
         route="hier_personal", route_lambda=0.2,
         route_temp=0.25, personal_target=0.40, route_fine=True, beta=0.6,
         mode="neighbor_mix", personal_share=0.25),
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
    """3 hierarchical group experts + 1 raw-personal expert (routing only).

    Non-hier modes delegate to the original CollaborativeMoE unchanged, so
    every historical trial remains reproducible inside this worker.
    """

    def __init__(self, mode, dim, labels=None, centers=None,
                 coarse=None, fine=None, parent=None,
                 personal_logit_init=PERSONAL_INIT_LOGIT,
                 route_temp=None, route_fine=False, beta=COARSE_BETA,
                 hybrid_residual=False):
        if mode != "hier_personal":
            super().__init__(mode, dim, labels=labels, centers=centers)
            return
        nn.Module.__init__(self)
        self.mode = "hier_personal"
        self.dim = int(dim)
        self.route_temp = float(route_temp if route_temp is not None else ROUTE_TEMP)
        self.route_fine = bool(route_fine)
        self.beta = float(beta)
        self._grad_seen = set()
        self._grad_nonfinite = set()
        if coarse is None or fine is None or parent is None:
            raise RuntimeError("hier_personal requires precomputed centers")
        if int(coarse.shape[0]) != K1 or int(fine.shape[0]) != K1 * K2_PER_GROUP:
            raise RuntimeError("Cluster geometry mismatch")
        self.register_buffer("coarse_centers", coarse.detach().clone().float())
        self.register_buffer("fine_centers", fine.detach().clone().float())
        self.register_buffer("fine_parent", parent.detach().clone())
        # Adaptive init: match typical coarse-cosine logit scale so the
        # personal expert starts at ~40% share instead of being swamped.
        self.personal_logit = nn.Parameter(torch.tensor(float(personal_logit_init)))
        self.hybrid_residual = bool(hybrid_residual)
        if self.hybrid_residual:
            # Zero-init learnable residual on top of the cluster-cosine
            # logits: routing starts EXACTLY at the hierarchical prior, then
            # adapts as MF trains (fixes stale static centers).
            self.route_residual = nn.Linear(self.dim, K1 + 1)
            nn.init.zeros_(self.route_residual.weight)
            nn.init.zeros_(self.route_residual.bias)
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.register_hook(self._grad_hook(name))

    def forward(self, rec, user_ids, item_ids, samples):
        if self.mode != "hier_personal":
            return super().forward(rec, user_ids, item_ids, samples)
        user_table = rec.user_embedding.weight
        uids = user_ids.long().to(user_table.device).reshape(-1)
        if bool((uids < 0).any()) or bool((uids >= user_table.shape[0]).any()):
            raise RuntimeError("User ID outside MF embedding table")
        user = F.normalize(user_table.index_select(0, uids).detach().float(),
                           p=2, dim=-1, eps=1e-8)
        batch = user.shape[0]
        if self.route_fine:
            cos2 = user @ self.fine_centers.T
            vecs = []
            for g in range(int(self.coarse_centers.shape[0])):
                member = (self.fine_parent == g).float()
                s2 = F.softmax(cos2 / FINE_TEMP + (member - 1.0) * 1e4, dim=-1)
                v_fine = s2 @ self.fine_centers
                v_g = self.beta * self.coarse_centers[g].unsqueeze(0) + \
                    (1.0 - self.beta) * v_fine
                vecs.append(F.normalize(v_g, p=2, dim=-1, eps=1e-8))
            V = torch.stack(vecs, dim=1)
            cos1 = (user.unsqueeze(1) * V).sum(dim=-1)
        else:
            cos1 = user @ self.coarse_centers.T
        logits = torch.cat((cos1 / self.route_temp,
                            self.personal_logit.expand(batch, 1)), dim=-1)
        if self.hybrid_residual:
            logits = logits + self.route_residual(user)
        probs = torch.softmax(logits.float(), dim=-1)
        if tuple(probs.shape) != (batch, EXPERTS):
            raise RuntimeError("Router must produce [B,4] probabilities")
        probs4 = probs.unsqueeze(1).expand(-1, 4, -1)
        masks = 1.0 + moe.RESIDUAL_LAMBDA * (EXPERTS * probs4 - 1.0)
        if not bool(torch.isfinite(masks).all()):
            raise FloatingPointError("Nonfinite soft expert masks")
        route_average = probs.mean(dim=0)
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
        personal_target = float(trial.get("personal_target", 0.40))
        route_fine = bool(trial.get("route_fine", False))
        beta = float(trial.get("beta", COARSE_BETA))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED + 104)
            table_cpu = rec.user_embedding.weight.detach().cpu()
            coarse, fine, parent = hierarchical_centers(table_cpu)
            user_n = F.normalize(table_cpu[1:], p=2, dim=-1, eps=1e-8)
            if route_fine:
                # z over blended per-group vectors (approximate scale check)
                cos_all = user_n @ coarse.T
            else:
                cos_all = user_n @ coarse.T
            z_mean = float((cos_all / route_temp).exp().sum(dim=-1).mean())
            ps = min(max(personal_target, 0.05), 0.95)
            # Exact target share for the average user:
            # exp(pl) = ps/(1-ps) * E[sum_g exp(cos_g/T)].
            pl_init = math.log(ps / (1.0 - ps)) + math.log(max(z_mean, 1e-12))
            gen.rank4_experts = HierPersonalMoE(
                "hier_personal", user_dim, coarse=coarse, fine=fine,
                parent=parent, personal_logit_init=pl_init,
                route_temp=route_temp, route_fine=route_fine, beta=beta,
                hybrid_residual=bool(trial.get("hybrid_residual", False)))
        n = sum(p.numel() for p in gen.rank4_experts.parameters() if p.requires_grad)
        print(f"[HIER INIT] {trial['name']} mode={mode} lambda={moe.RESIDUAL_LAMBDA} "
              f"K1={K1} K2={K2_PER_GROUP} T={route_temp} "
              f"ps={personal_target} fine={route_fine} beta={beta} "
              f"pl_init={pl_init:.3f} router_trainable={n}",
              flush=True)

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
                m = masks[group].unsqueeze(1).to(weight.device, dtype=weight.dtype)
                if m.shape[0] != weight.shape[0]:
                    raise RuntimeError("Mask batch mismatch")
                cache[key] = weight * m
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
    if lam not in (0.2, 1.0):
        raise RuntimeError(f"Unapproved lambda: {lam}")
    moe.RESIDUAL_LAMBDA = lam
    fac._install_grad_norm_hooks()
    install_hier_model(trial)
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
        "hypothesis": "3 hierarchical cluster group experts + 1 raw-personal "
                      "expert, routed by coarse-centroid cosine + personal logit",
        "trials": TRIALS,
        "clustering": {"K1": K1, "K2_per_group": K2_PER_GROUP,
                       "source": "initial pretrained MF user table; frozen buffers"},
        "routing": {"formula": "softmax([cos(u,c1)/0.25 x3, personal_logit])",
                    "personal_init_logit": PERSONAL_INIT_LOGIT,
                    "balance": BALANCE_WEIGHT},
        "controls": {"D4": "TOPK_RES025; personal_share=.25",
                     "C6": "CTR=.005 KD=.010", "cora_lr": 3e-4,
                     "mf_lr_ratio": 0.1, "batch": 2, "accum": 4,
                     "iters": 400, "patience": 10, "valid_only": True},
    }
    (root / "hier_fullstack_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[HIER PLAN PASS] manifest={root / 'hier_fullstack_manifest.json'}",
          flush=True)


def cpu_test():
    torch.manual_seed(SEED)
    dim, batch = 16, 8
    rec = type("MF", (), {"user_embedding": nn.Embedding(300, dim),
                          "item_embedding": nn.Embedding(300, dim)})()
    with torch.no_grad():
        rec.user_embedding.weight.normal_()
    coarse, fine, parent = hierarchical_centers(rec.user_embedding.weight)
    user_ids = torch.arange(1, batch + 1)
    item_ids = torch.arange(2, batch + 2)
    for lam in (0.2, 1.0):
        moe.RESIDUAL_LAMBDA = lam
        pl_init = (math.log(2.0 / 3.0)
                   + math.log(float(
                       (F.normalize(rec.user_embedding.weight[1:].detach(),
                                    p=2, dim=-1) @ coarse.T / ROUTE_TEMP)
                       .exp().sum(dim=-1).mean())))
        model = HierPersonalMoE("hier_personal", dim,
                                coarse=coarse, fine=fine, parent=parent,
                                personal_logit_init=pl_init)
        masks, aux, stats, probs = model(rec, user_ids, item_ids, {})
        assert len(masks) == 4 and tuple(masks[0].shape) == (batch, 16)
        assert tuple(probs.shape) == (batch, 4, 4)
        expected = 1.0 + lam * (4.0 * probs - 1.0)
        torch.testing.assert_close(
            masks[0], expected[:, 0].repeat_interleave(4, dim=-1),
            atol=1e-6, rtol=1e-6)
        # Adaptive init must give the personal expert ~40% share.
        p_pers = float(probs[0, 0, -1])
        assert 0.30 <= p_pers <= 0.50, p_pers
        # Gradient reaches personal_logit through the mask sum.
        signal = torch.randn(batch, 16)
        loss = (signal * masks[0]).sum() + aux
        loss.backward()
        assert model.personal_logit.grad is not None and \
            float(model.personal_logit.grad.abs()) > 0
        print(f"[HIER CPU] lambda={lam} p_personal={p_pers:.3f} "
              "mask_formula_ok gradient_ok", flush=True)
    # Non-hier delegation must still construct and route.
    moe.RESIDUAL_LAMBDA = 0.2
    legacy = HierPersonalMoE("user_soft4", dim)
    masks, aux, stats, probs = legacy(rec, user_ids, item_ids, {})
    assert len(masks) == 4
    print("[HIER CPU PASS] hierarchical routing, formula, gradients, "
          "legacy delegation", flush=True)


def summary(protocol, debug=False):
    log_dir = HERE / "logs" / protocol
    out_dir = HERE / "out" / protocol
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "hier_fullstack_valid_summary.csv"
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
    print(f"[HIER SUMMARY] {target}")
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
        print("[HIER COMPLETE] selected trials finished. Valid only.", flush=True)
        return
    parser.error("Specify --cpu-test, --plan, --run or --summary")


if __name__ == "__main__":
    main()
