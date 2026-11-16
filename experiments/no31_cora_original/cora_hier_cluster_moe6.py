#!/usr/bin/env python3
"""Hierarchical cluster-group + raw-personal heterogeneous expert CoRA-MoE.

Extends the proven independent-rank4-expert framework of
cora_clean_rank4_moe6.py with HETEROGENEOUS expert conditions:

  * K1=3 GROUP experts (rank4 each): expert g is conditioned on a TWO-LEVEL
    cluster vector v_g(u) built from spherical k-means on the initial
    pretrained MF user table:
        fine attention   s2_j = softmax_{j in group g}(cos(u, c2_j)/0.2)
        v_fine_g         = sum_j s2_j * c2_j          (K2=8 fine centers)
        v_g              = normalize(0.5*c1_g + 0.5*v_fine_g)
    Each group has its own MLP that MODULATES the shared QFormer pooling:
        cond_g = LayerNorm(pool + MLP_g(v_g));  A_g = head_g(cond_g)
  * 1 PERSONAL expert (rank4): conditioned on the RAW interaction features
        cond_p = LayerNorm(pool + MLP_p(normalize([u, i, u*i])))
        A_p    = head_p(cond_p)

Routing (heterogeneous, user-only):
        p = softmax([cos(u, c1_g)/0.25 for g=0..2, b_p])   # b_p init log(2)
        p_personal ~= 0.40 at init (lesson from G1_CLUSTER_K32: pure
        clustering collapses UAUC; the personal expert keeps a real share)
        mask = 1 + lambda*(4*p - 1);  warmup 200 microsteps uniform;
        balance loss 1e-3 * sum((p_bar-0.25)^2)

Total rank budget stays 16 (4 x rank4) exactly like the clean baseline,
so results compare fairly against CORA_CLEAN_R4_EARLY5_FIX1 (F3=0.7257).
Cluster centers are computed ONCE from the INITIAL pretrained MF user
table and stored as frozen buffers (same philosophy as D4 neighbor ids).
The Vicuna backbone stays frozen; MF + QFormer + heads + P + router train.

Two variants: H1 lambda=0.2, H2 lambda=1.0.

Run from CoRA_official root:
 python experiments/no31_cora_original/cora_hier_cluster_moe6.py --cpu-test
 python experiments/no31_cora_original/cora_hier_cluster_moe6.py --plan --protocol CORA_HIER_CLUSTER_MOE_20261112_H1 --route-strength 0.2
 CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/cora_hier_cluster_moe6.py --run \
     --protocol CORA_HIER_CLUSTER_MOE_20261112_H1 --route-strength 0.2
"""

import argparse
import inspect
import json
import math
import os
import re
import runpy
import subprocess
import sys
import textwrap
import types
import weakref
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if not (ROOT / "train_collm_mf_din.py").is_file():
    raise RuntimeError("Expected CoRA_official root above experiments/no31_cora_original")
for d in (str(ROOT), str(HERE)):
    if d not in sys.path:
        sys.path.insert(0, d)

import cora_joint8 as joint

SEED = 42
N_EXPERTS = 4                 # 3 group + 1 personal; total rank budget 16
N_GROUP_EXPERTS = 3
RANK_PER_EXPERT = 4
N_LAYERS = 32
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
SLOTS = N_LAYERS * len(PROJECTIONS)
STEPS = 400
GROUPS = 4
ROUTER_HIDDEN = 64
COND_HIDDEN = 256
K1 = 3                       # coarse clusters == group experts
K2_PER_GROUP = 8             # fine clusters per coarse group
COARSE_BETA = 0.5            # coarse/fine mixing in v_g
FINE_TEMP = 0.2              # fine attention temperature (D4 neighbor value)
ROUTE_TEMP = 0.25            # routing softmax temperature over cosine
PERSONAL_INIT_LOGIT = math.log(2.0)   # -> p_personal ~= 0.40 at init
DEFAULT_PROTOCOL = "CORA_HIER_CLUSTER_MOE_20261112"
RANK_RE = re.compile(r"layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$")


def _spherical_kmeans(x, k, seed, iterations=15):
    """Deterministic spherical k-means; x must already be L2-normalized."""
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


def _hierarchical_centers(user_table, uid_lo=1):
    """Two-level spherical k-means over the initial MF user table."""
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
    fine = torch.cat(fine_list, dim=0)                  # [K1*K2, D]
    parent = torch.cat(parents, dim=0)                  # [K1*K2]
    sizes = torch.bincount(parent, minlength=K1)
    if bool((sizes == 0).any()):
        raise RuntimeError("Empty coarse group in hierarchical clustering")
    return coarse, fine, parent


class HierClusterPersonalMoE(nn.Module):
    """Heterogeneous expert router + per-expert condition modulators."""

    def __init__(self, n_slots, hidden, qformer_width, user_dim, mode,
                 strength, warmup, coarse, fine, parent,
                 personal_logit_init=PERSONAL_INIT_LOGIT):
        super().__init__()
        if mode != "hier_personal":
            raise ValueError("This MoE only implements hier_personal mode")
        if not 0 <= strength <= 1:
            raise ValueError("Router strength outside [0,1]")
        self.n_slots = int(n_slots)
        self.hidden = int(hidden)
        self.qformer_width = int(qformer_width)
        self.user_dim = int(user_dim)
        self.strength = float(strength)
        self.warmup = int(warmup)
        self.mode = mode
        if int(coarse.shape[0]) != N_GROUP_EXPERTS or int(fine.shape[0]) != \
                K1 * K2_PER_GROUP:
            raise RuntimeError("Cluster geometry mismatch")
        self.register_buffer("coarse_centers", coarse.detach().clone())
        self.register_buffer("fine_centers", fine.detach().clone())
        self.register_buffer("fine_parent", parent.detach().clone())
        # Each expert owns an independent rank4 P per slot (same as clean MoE).
        self.proj = nn.Parameter(torch.zeros(n_slots, N_EXPERTS, hidden,
                                             RANK_PER_EXPERT))
        # Group experts: per-group condition MLP over the hierarchical
        # cluster vector v_g(u), modulating the shared QFormer pooling.
        self.group_cond = nn.ModuleList([
            nn.Sequential(
                nn.Linear(user_dim, COND_HIDDEN),
                nn.GELU(),
                nn.Linear(COND_HIDDEN, qformer_width),
            )
            for _ in range(N_GROUP_EXPERTS)
        ])
        self.group_heads = nn.ModuleList([
            nn.Linear(qformer_width, hidden * RANK_PER_EXPERT)
            for _ in range(N_GROUP_EXPERTS)
        ])
        # Personal expert: raw interaction features [u, i, u*i] (3*user_dim).
        self.personal_cond = nn.Sequential(
            nn.Linear(3 * user_dim, COND_HIDDEN),
            nn.GELU(),
            nn.Linear(COND_HIDDEN, qformer_width),
        )
        self.personal_head = nn.Linear(qformer_width, hidden * RANK_PER_EXPERT)
        # Adaptive init: match the typical coarse-cosine logit scale so the
        # personal expert starts at ~40% share instead of being swamped.
        self.personal_logit = nn.Parameter(torch.tensor(float(personal_logit_init)))
        self._route = None
        self._steps = 0
        self._calls = 0

    # ---- routing context -------------------------------------------------
    def begin(self, rec, samples, is_training):
        if self._route is not None:
            raise RuntimeError("Nested CoRA MoE forward")
        user_table = rec.user_embedding.weight
        item_table = rec.item_embedding.weight
        uids = samples["UserID"].long().view(-1).to(user_table.device)
        iids = samples["TargetItemID"].long().view(-1).to(item_table.device)
        u = F.normalize(user_table.index_select(0, uids).detach().float(),
                        p=2, dim=-1, eps=1e-8)
        i = F.normalize(item_table.index_select(0, iids).detach().float(),
                        p=2, dim=-1, eps=1e-8)
        u_raw = user_table.index_select(0, uids).detach().float()
        i_raw = item_table.index_select(0, iids).detach().float()
        raw_feat = F.normalize(torch.cat((u_raw, i_raw, u_raw * i_raw), dim=-1),
                               p=2, dim=-1, eps=1e-8)
        # Hierarchical cluster vectors v_g(u): fine attention inside each
        # coarse group, then coarse/fine mixing.
        cos2 = u @ self.fine_centers.T                       # [B, K1*K2]
        group_vecs = []
        for g in range(N_GROUP_EXPERTS):
            member = (self.fine_parent == g).float()          # [K1*K2]
            logits = cos2 / FINE_TEMP + (member - 1.0) * 1e4  # mask others
            s2 = F.softmax(logits, dim=-1)
            v_fine = s2 @ self.fine_centers                   # [B, D]
            v_g = COARSE_BETA * self.coarse_centers[g].unsqueeze(0) + \
                (1.0 - COARSE_BETA) * v_fine
            group_vecs.append(F.normalize(v_g, p=2, dim=-1, eps=1e-8))
        self._group_vecs = torch.stack(group_vecs, dim=1)     # [B, G, D]
        self._raw_feat = raw_feat

        stage_warmup = bool(is_training and self._steps < self.warmup)
        if stage_warmup:
            prob = u.new_full((u.shape[0], N_EXPERTS), 1.0 / N_EXPERTS)
        else:
            cos1 = u @ self.coarse_centers.T                  # [B, K1]
            logits = torch.cat((cos1 / ROUTE_TEMP,
                                self.personal_logit.expand(u.shape[0], 1)),
                               dim=-1)
            prob = F.softmax(logits.float(), dim=-1)
        route = 1.0 + self.strength * (N_EXPERTS * prob - 1.0)
        self._route = route
        self._calls = 0
        balance = 1e-3 * (prob.mean(dim=0) - 0.25).square().sum()
        if is_training:
            self._steps += 1
        self._last_prob_mean = prob.detach().mean(dim=0).cpu()
        return balance

    def end(self):
        self._route = None
        self._group_vecs = None
        self._raw_feat = None

    def dynamic_experts(self, pooled):
        """Four rank4 expert weights from heterogeneous conditions.

        pooled: [B, qformer_width] shared QFormer mean over query tokens.
        """
        outs = []
        for g in range(N_GROUP_EXPERTS):
            cond = F.layer_norm(pooled + self.group_cond[g](self._group_vecs[:, g]),
                                (self.qformer_width,))
            a = F.normalize(self.group_heads[g](cond), p=2, dim=-1, eps=1e-12)
            outs.append(a.reshape(pooled.shape[0], self.hidden,
                                  RANK_PER_EXPERT) / math.sqrt(N_EXPERTS))
        cond_p = F.layer_norm(pooled + self.personal_cond(self._raw_feat),
                              (self.qformer_width,))
        a = F.normalize(self.personal_head(cond_p), p=2, dim=-1, eps=1e-12)
        outs.append(a.reshape(pooled.shape[0], self.hidden,
                              RANK_PER_EXPERT) / math.sqrt(N_EXPERTS))
        return torch.stack(outs, dim=1)                       # [B, E, H, r]

    def project(self, slot, x, dynamic, dropout, scaling, base_output):
        if self._route is None:
            raise RuntimeError("MoE routing context missing")
        if dynamic.ndim != 4 or dynamic.shape[1] != N_EXPERTS \
                or dynamic.shape[-1] != RANK_PER_EXPERT:
            raise RuntimeError("Generator must provide four rank4 experts")
        if x.ndim != 3 or dynamic.shape[0] != x.shape[0] \
                or dynamic.shape[2] != x.shape[2]:
            raise RuntimeError("CoRA weight / token shape mismatch")
        p = self.proj[slot]
        xx = dropout(x).to(dtype=p.dtype)
        a = dynamic.to(dtype=p.dtype)
        z = torch.einsum("bth,behr->bter", xx, a)
        mask = self._route[:, None, :, None].to(dtype=p.dtype)
        delta = torch.einsum("bter,eor->bto", z * mask, p)
        self._calls += 1
        return base_output + (delta * scaling).to(base_output.dtype)


def _patch_model(args):
    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx
    from minigpt4.models.ppeft.plora import Linear as PLoraLinear

    prior_init = MiniGPT4Rec_vx.__init__
    prior_train = MiniGPT4Rec_vx.forward_v2
    prior_eval = MiniGPT4Rec_vx.generate_for_samples_v2
    prior_linear = PLoraLinear.forward

    def init_hier(self, *a, **kw):
        prior_init(self, *a, **kw)
        gen = self.llama_model_lora.base_model.weight_generator
        if int(gen.r) != RANK_PER_EXPERT:
            raise RuntimeError("HCP config MUST initialize r=4")
        old_final = gen.final
        if not isinstance(old_final, nn.Linear):
            raise RuntimeError("Expected original CoRA single final head")
        hidden = int(gen.hidden_size)
        if hidden != 4096:
            raise RuntimeError("Expected Vicuna-7B hidden=4096")
        qformer_width = int(old_final.in_features)
        if gen.query_tokens.shape[-1] != qformer_width:
            raise RuntimeError("QFormer/head input dimension mismatch")
        rec = self.rec_encoder
        d = int(rec.user_embedding.weight.shape[-1])
        if int(rec.item_embedding.weight.shape[-1]) != d:
            raise RuntimeError("User and item MF embedding dimensions differ")
        coarse, fine, parent = _hierarchical_centers(
            rec.user_embedding.weight.detach().cpu())
        table_cpu = rec.user_embedding.weight.detach().cpu()
        cos_all = F.normalize(table_cpu[1:], p=2, dim=-1, eps=1e-8) @ coarse.T
        z_mean = float((cos_all / ROUTE_TEMP).exp().sum(dim=-1).mean())
        # Exact 40% average share: exp(pl) = (2/3) * E[sum_g exp(cos_g/T)].
        pl_init = math.log(2.0 / 3.0) + math.log(max(z_mean, 1e-12))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED + 304)
            moe = HierClusterPersonalMoE(
                SLOTS, hidden, qformer_width, d, "hier_personal",
                args.route_strength, args.warmup_microsteps,
                coarse, fine, parent, personal_logit_init=pl_init)
        first_module_target = None
        for name, mod in self.llama_model_lora.named_modules():
            m = RANK_RE.search(name)
            if m and isinstance(mod, PLoraLinear):
                if int(mod.r[mod.active_adapter]) != RANK_PER_EXPERT:
                    raise RuntimeError("Each projection must initialize rank4")
                if first_module_target is None:
                    first_module_target = mod.lora_P[mod.active_adapter].weight
        if first_module_target is None:
            raise RuntimeError("No Q/K/V/O rank4 target found")
        moe = moe.to(device=first_module_target.device,
                     dtype=first_module_target.dtype)
        self.cora_hier_cluster_moe = moe

        def four_rank4_hier_forward(g, ui_embedding):
            if ui_embedding.ndim != 2 or \
                    ui_embedding.shape[-1] != g.embedding_size * 2:
                raise RuntimeError("Generator needs full concatenated MF user+item")
            holder = g._rec_model_ref() if getattr(g, "_rec_model_ref", None) is not None else None
            moe_ref = getattr(holder, "cora_hier_cluster_moe", None)
            if moe_ref is None:
                raise RuntimeError("HCP MoE context missing on generator")
            encoder = ui_embedding.unsqueeze(1)
            attn = torch.ones(encoder.shape[:2], dtype=torch.long,
                              device=encoder.device)
            queries = g.query_tokens.expand(encoder.shape[0], -1, -1)
            states = g.qformer(
                query_embeds=queries,
                encoder_hidden_states=encoder,
                encoder_attention_mask=attn,
                use_cache=True,
                return_dict=True,
            ).last_hidden_state
            pooled = states.mean(dim=1)
            return moe_ref.dynamic_experts(pooled.float())

        gen.forward = types.MethodType(four_rank4_hier_forward, gen)
        gen._rec_model_ref = weakref.ref(self)

        modules = {}
        for name, mod in self.llama_model_lora.named_modules():
            m = RANK_RE.search(name)
            if m and isinstance(mod, PLoraLinear):
                slot = int(m.group(1)) * len(PROJECTIONS) + \
                    PROJECTIONS.index(m.group(2))
                if slot in modules:
                    raise RuntimeError("Duplicate Q/K/V/O slot")
                modules[slot] = mod
        if len(modules) != SLOTS:
            raise RuntimeError(f"Expected 128 Q/K/V/O targets, found {len(modules)}")
        for slot, mod in modules.items():
            mod.lora_P[mod.active_adapter].weight.requires_grad_(False)
            mod._hier_slot = slot
            mod._hier_ref = weakref.ref(self.cora_hier_cluster_moe)
        train_mf = sum(p.numel() for p in rec.parameters() if p.requires_grad)
        train_gen = sum(p.numel() for p in gen.qformer.parameters()
                        if p.requires_grad)
        if not train_mf or not train_gen:
            raise RuntimeError("MF and CoRA QFormer must both be trainable")
        n_expert_params = sum(p.numel() for p in self.cora_hier_cluster_moe.parameters()
                              if p.requires_grad)
        print(f"[HCP INIT] MF_trainable={train_mf} qformer_trainable={train_gen} "
              f"expert_P={self.cora_hier_cluster_moe.proj.numel()} "
              f"expert_cond_head_params={n_expert_params - self.cora_hier_cluster_moe.proj.numel()} "
              f"K1={K1} K2={K2_PER_GROUP} beta={COARSE_BETA} "
              f"lambda={args.route_strength} personal_init={PERSONAL_INIT_LOGIT:.3f}",
              flush=True)

    def routed_linear(self, x, w=None):
        slot = getattr(self, "_hier_slot", None)
        if slot is None:
            return prior_linear(self, x, w)
        if w is None:
            raise RuntimeError("Missing four-rank4 CoRA weights")
        weight = self.weight.T if self.fan_in_fan_out else self.weight
        result = F.linear(x, weight, self.bias)
        if self.disable_adapters or self.merged:
            raise RuntimeError("HCP MoE cannot disable or merge its adapter")
        moe = self._hier_ref()
        if moe is None:
            raise RuntimeError("Missing HCP MoE module")
        return moe.project(slot, x, w, self.lora_dropout[self.active_adapter],
                           self.scaling[self.active_adapter], result)

    def wrapped(fn, train_method):
        def f(self, samples, *a, **kw):
            moe = self.cora_hier_cluster_moe
            balance = moe.begin(self.rec_encoder, samples,
                                train_method and self.training)
            try:
                out = fn(self, samples, *a, **kw)
                if moe._calls < SLOTS:
                    raise RuntimeError(
                        f"Only {moe._calls}/{SLOTS} rank4 projections used")
                if train_method and self.training:
                    if not isinstance(out, dict) or "loss" not in out:
                        raise RuntimeError("Expected CoRA BCE loss dictionary")
                    out["loss"] = out["loss"] + balance
                    if moe._steps == 1 or moe._steps % 100 == 0:
                        print(f"[HCP TRAIN] step={moe._steps} "
                              f"lambda={moe.strength} "
                              f"balance={float(balance.detach()):.8g} "
                              f"prob_mean={moe._last_prob_mean.tolist()}",
                              flush=True)
                return out
            finally:
                moe.end()
        return f

    MiniGPT4Rec_vx.__init__ = init_hier
    PLoraLinear.forward = routed_linear
    MiniGPT4Rec_vx.forward_v2 = wrapped(prior_train, True)
    MiniGPT4Rec_vx.generate_for_samples_v2 = wrapped(prior_eval, False)


def _paths(protocol):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", protocol):
        raise ValueError("Unsafe protocol name")
    cfg = HERE / "local_configs" / protocol / "HIER_MOE.yaml"
    log = HERE / "logs" / protocol / "HIER_MOE.log"
    out = HERE / "output" / protocol / "HIER_MOE"
    return cfg, log, out


def plan(args):
    source = Path(args.source_yaml).resolve()
    if not source.is_file():
        raise FileNotFoundError("Missing clean CoRA base config: " + str(source))
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    run, model = cfg["run"], cfg["model"]
    if int(run["seed"]) != SEED:
        raise RuntimeError("Base config must have seed42")
    if not model["lora_config"].get("use_lora", False):
        raise RuntimeError("Expected CoRA LoRA")
    cfg_path, log, out = _paths(args.protocol)
    if cfg_path.exists() or log.exists() or out.exists():
        raise FileExistsError("Existing artifacts; choose a NEW protocol")
    model["lora_config"]["r"] = RANK_PER_EXPERT
    model["lora_config"]["alpha"] = 2 * RANK_PER_EXPERT
    model["lora_config"]["target_modules"] = list(PROJECTIONS)
    model["freeze_rec"] = False
    model["freeze_lora"] = False
    model["ckpt"] = None
    run["init_lr"] = float(args.lr)
    if float(run.get("min_lr", 0)) > args.lr:
        run["min_lr"] = args.lr / 10
    run["batch_size_train"] = 2
    run["accum_grad_iters"] = 4
    run["iters_per_epoch"] = STEPS
    run["evaluate"] = False
    run["valid_splits"] = ["valid"]
    run["test_splits"] = []
    run["resume_ckpt_path"] = None
    run["output_dir"] = str(out)
    run["max_epoch"] = max(int(run.get("max_epoch", 1000)), 1000)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (log.parent / "hier_cluster_manifest.json").write_text(
        json.dumps({
            "protocol": args.protocol,
            "seed": SEED,
            "experts": {"group": N_GROUP_EXPERTS, "personal": 1,
                        "rank_each": RANK_PER_EXPERT},
            "clustering": {"level1": K1, "level2_per_group": K2_PER_GROUP,
                           "beta": COARSE_BETA, "fine_temp": FINE_TEMP,
                           "source": "initial pretrained MF user table (frozen buffers)"},
            "routing": {"type": "cosine over coarse centers + personal logit",
                        "route_temp": ROUTE_TEMP,
                        "personal_init_logit": PERSONAL_INIT_LOGIT,
                        "mask": "1+lambda*(4p-1)", "lambda": args.route_strength,
                        "warmup_microsteps": args.warmup_microsteps},
            "loss": "original CoRA Yes-token BCE + 1e-3 router balance",
            "lr": args.lr, "batch_size": 2, "gradient_accumulation": 4,
            "stop": "5 consecutive Valid AUC evaluations without improvement",
            "valid_only": True, "gpu0_only": True,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[HCP PLAN PASS] cfg={cfg_path} log={log} output={out} "
          "epochs=uncapped patience=5", flush=True)


def _remove_training_ceiling():
    """Same validated runner-uncap patch as the clean framework."""
    from minigpt4.runners.runner_base import RunnerBase
    train_src = textwrap.dedent(inspect.getsource(RunnerBase.train))
    lines = train_src.splitlines(keepends=True)
    old_loop = "for cur_epoch in range(self.start_epoch, self.max_epoch):"
    old_night = '_night_epochs=int(__import__("os").environ["S2_NIGHT_EPOCHS"])'
    loop_idxs = [i for i, line in enumerate(lines) if line.strip() == old_loop]
    night_idxs = [i for i, line in enumerate(lines) if line.strip() == old_night]
    if len(loop_idxs) != 1 or len(night_idxs) > 1:
        raise RuntimeError("Unexpected RunnerBase.train structure")
    loop_i = loop_idxs[0]
    indentation = lines[loop_i][:len(lines[loop_i]) - len(lines[loop_i].lstrip())]
    lines[loop_i] = (indentation +
                     'for cur_epoch in __import__("itertools").count(self.start_epoch):\n')
    if night_idxs:
        night_i = night_idxs[0]
        expected = [
            old_night,
            "assert _night_epochs in (1,2)",
            "if cur_epoch+1 >= _night_epochs:",
            'logging.info("[NIGHT8-STOP] trained_epochs=%d max_epoch=%d "',
            '"Valid and checkpoint completed",cur_epoch+1,self.max_epoch)',
            "break",
        ]
        actual = [v.strip() for v in lines[night_i:night_i + 6]]
        if actual != expected:
            raise RuntimeError("Unknown forced-stop block; refusing to modify")
        del lines[night_i:night_i + 6]
    train_src = "".join(lines)
    if "S2_NIGHT_EPOCHS" in train_src or old_loop in train_src:
        raise RuntimeError("Remaining hardcoded ceiling")
    patched_local = {}
    exec(compile(train_src,
                 inspect.getsourcefile(RunnerBase.train) or "<hcp-runner>",
                 "exec"), RunnerBase.train.__globals__, patched_local)
    RunnerBase.train = patched_local["train"]
    print("[HCP UNCAPPED PASS] epoch loop unbounded; patience early stop active",
          flush=True)


def _audit_optimizer():
    from minigpt4.runners.runner_base import RunnerBase
    old = RunnerBase.optimizer.fget

    def checked(self):
        optim = old(self)
        if not getattr(self, "_hcp_optim_checked", False):
            model = self.unwrap_dist_model(self.model)
            actual = {id(p) for g in optim.param_groups for p in g["params"]}
            wanted = {id(p) for p in model.parameters() if p.requires_grad}
            if actual != wanted:
                raise RuntimeError(
                    f"Optimizer mismatch: missing={len(wanted-actual)} "
                    f"extra={len(actual-wanted)}")
            moe = model.cora_hier_cluster_moe
            mf = {id(p) for p in model.rec_encoder.parameters() if p.requires_grad}
            expert = {id(p) for p in moe.parameters() if p.requires_grad}
            if not expert or not expert.issubset(actual):
                raise RuntimeError("HCP experts missing from optimizer")
            if not mf.issubset(actual):
                raise RuntimeError("MF missing from optimizer")
            if not any(g.get("joint8_mf") for g in optim.param_groups):
                raise RuntimeError("MF LR 0.1x group missing")
            self._hcp_optim_checked = True
            print("[HCP OPT PASS] MF + QFormer + heterogeneous heads + 128x4 "
                  "rank4 P + router registered", flush=True)
        return optim

    RunnerBase.optimizer = property(checked)


def worker(args):
    cfg_path, _, _ = _paths(args.protocol)
    if not cfg_path.is_file():
        raise FileNotFoundError("Run --plan first")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if args.epochs != 0 or args.patience != 5:
        raise RuntimeError("Uncapped HCP protocol requires --epochs 0 --patience 5")
    _remove_training_ceiling()
    joint.patch_runner(screen_epochs=0, mf_lr_ratio=0.1, patience=5)
    joint.install_compact_atomic_best_checkpoint()
    _patch_model(args)
    _audit_optimizer()
    print(f"[HCP WORKER] protocol={args.protocol} "
          f"K1={K1} K2={K2_PER_GROUP} beta={COARSE_BETA} "
          f"lambda={args.route_strength} warmup={args.warmup_microsteps} "
          f"experts=3-group+1-personal rank_budget=16 valid_only=True",
          flush=True)
    sys.argv = ["train_collm_mf_din.py", "--cfg-path", str(cfg_path)]
    runpy.run_path(str(ROOT / "train_collm_mf_din.py"), run_name="__main__")


def gpu_check(min_free):
    check = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    if check.returncode:
        raise RuntimeError(check.stderr)
    for line in check.stdout.splitlines():
        vals = [v.strip() for v in line.split(",")]
        if vals[0] == "0":
            free = int(vals[1])
            if free < min_free:
                raise RuntimeError("Insufficient free GPU0 memory")
            return
    raise RuntimeError("Physical GPU0 unavailable")


def cpu_test():
    torch.manual_seed(SEED)
    # --- hierarchical clustering sanity ---
    table = torch.randn(400, 16)
    coarse, fine, parent = _hierarchical_centers(table)
    assert tuple(coarse.shape) == (K1, 16)
    assert tuple(fine.shape) == (K1 * K2_PER_GROUP, 16)
    assert int(parent.numel()) == K1 * K2_PER_GROUP
    sizes = torch.bincount(parent, minlength=K1)
    assert bool((sizes == K2_PER_GROUP).all()), sizes
    assert torch.allclose(coarse.norm(dim=-1), torch.ones(K1), atol=1e-5)

    # --- MoE forward / routing / gradients ---
    hidden, d, batch, seq = 32, 16, 5, 7
    qwidth = 24
    rec = type("MF", (), {
        "user_embedding": nn.Embedding(400, d),
        "item_embedding": nn.Embedding(400, d)})()
    with torch.no_grad():
        rec.user_embedding.weight.copy_(table)
    ids = {"UserID": torch.tensor([1, 2, 3, 4, 5]),
           "TargetItemID": torch.tensor([6, 7, 8, 9, 10])}
    moe = HierClusterPersonalMoE(4, hidden, qwidth, d, "hier_personal",
                                 0.2, warmup=0, coarse=coarse, fine=fine,
                                 parent=parent,
                                 personal_logit_init=(
                                     math.log(2.0 / 3.0)
                                     + math.log(float(
                                         (F.normalize(table[1:], p=2, dim=-1)
                                          @ coarse.T / ROUTE_TEMP)
                                         .exp().sum(dim=-1).mean()))))
    with torch.no_grad():
        moe.proj.normal_(0, .01)
    pooled = torch.randn(batch, qwidth)
    x = torch.randn(batch, seq, hidden, requires_grad=True)

    # Warmup off (warmup=0): routing must depend on the personal logit and
    # the adaptive init must give the personal expert ~40% share.
    balance = moe.begin(rec, ids, True)
    dynamic = moe.dynamic_experts(pooled)
    assert tuple(dynamic.shape) == (batch, N_EXPERTS, hidden, RANK_PER_EXPERT)
    prob0 = moe._last_prob_mean
    assert 0.30 <= float(prob0[-1]) <= 0.50, prob0
    y0 = moe.project(0, x, dynamic, nn.Identity(), 2.0, torch.zeros_like(x))
    (y0.square().mean() + balance).backward(retain_graph=True)
    trainable = [p for p in moe.parameters() if p.requires_grad]
    assert trainable and all(p.grad is not None for p in trainable), \
        [n for n, p in moe.named_parameters() if p.grad is None]
    assert moe.proj.grad.abs().sum() > 0
    for g in range(N_GROUP_EXPERTS):
        assert moe.group_cond[g][0].weight.grad is not None
        assert moe.group_heads[g].weight.grad is not None
    assert moe.personal_cond[0].weight.grad is not None
    assert moe.personal_head.weight.grad is not None
    assert moe.personal_logit.grad is not None
    moe.end()

    # Warmup on: uniform routing exactly.
    moe2 = HierClusterPersonalMoE(4, hidden, qwidth, d, "hier_personal",
                                  1.0, warmup=5, coarse=coarse, fine=fine,
                                  parent=parent)
    moe2.begin(rec, ids, True)
    assert torch.allclose(moe2._route,
                          torch.ones_like(moe2._route), atol=1e-6)
    moe2.end()

    # Mask formula check with lambda=1.0 after warmup.
    moe3 = HierClusterPersonalMoE(4, hidden, qwidth, d, "hier_personal",
                                  1.0, warmup=0, coarse=coarse, fine=fine,
                                  parent=parent)
    moe3.begin(rec, ids, False)
    expected = 1.0 + 1.0 * (4.0 * F.softmax(
        torch.cat((F.normalize(rec.user_embedding.weight[ids["UserID"]].detach(), dim=-1)
                   @ coarse.T / ROUTE_TEMP,
                   moe3.personal_logit.expand(5, 1)), dim=-1), dim=-1) - 1.0)
    torch.testing.assert_close(moe3._route, expected, atol=1e-5, rtol=1e-5)
    moe3.end()

    # Group vectors must respond to the user identity (personalization).
    moe.begin(rec, ids, False)
    v1 = moe._group_vecs.clone()
    ids2 = {"UserID": torch.tensor([100, 101, 102, 103, 104]),
            "TargetItemID": torch.tensor([6, 7, 8, 9, 10])}
    moe.end(); moe.begin(rec, ids2, False)
    assert float((v1 - moe._group_vecs).abs().sum()) > 1e-6
    moe.end()

    _remove_training_ceiling()
    print("[HCP CPU PASS] hierarchical kmeans, heterogeneous conditions, "
          "routing formula, warmup uniform, gradient connectivity", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpu-test", action="store_true")
    p.add_argument("--plan", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    p.add_argument("--source-yaml", default=str(joint.BASE))
    p.add_argument("--route-strength", type=float, default=0.2)
    p.add_argument("--warmup-microsteps", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-free-mib", type=int, default=22000)
    args = p.parse_args()
    if args.cpu_test:
        cpu_test()
        return
    if not (0 < args.lr <= 0.001 and args.epochs == 0 and args.patience == 5
            and args.warmup_microsteps >= 0
            and 0 <= args.route_strength <= 1):
        p.error("Require --epochs 0, --patience 5, valid LR/warmup/route")
    if args.plan:
        plan(args)
        return
    if args._worker:
        worker(args)
        return
    if not args.run:
        p.error("Specify --cpu-test, --plan or --run")
    cfg_path, log, out = _paths(args.protocol)
    if not cfg_path.is_file():
        plan(args)
    log.parent.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    if log.exists() or out.exists():
        raise FileExistsError("Refusing to overwrite existing results")
    gpu_check(args.min_free_mib)
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--_worker",
           "--protocol", args.protocol,
           "--route-strength", str(args.route_strength),
           "--warmup-microsteps", str(args.warmup_microsteps),
           "--lr", str(args.lr), "--epochs", str(args.epochs),
           "--patience", str(args.patience)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    print("[HCP RUN] " + " ".join(cmd), flush=True)
    with log.open("x", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                errors="replace", bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True)
            fh.write(line)
        code = proc.wait()
    if code:
        raise SystemExit(f"[HCP STOP] code={code} log={log}")
    print(f"[HCP DONE] output={out} VALID only; TEST not used", flush=True)


if __name__ == "__main__":
    main()
