#!/usr/bin/env python3
"""Five controlled split-rank4 CoRA expert experiments (VALID only).

The official CoRA rank-16 dynamic W_ui and each layer's rank-16 P adapter are
partitioned into FOUR non-overlapping rank-4 experts, along the rank axis:
  W_ui = concat(W_0, W_1, W_2, W_3), P = [P_0 P_1 P_2 P_3].
No extra LoRA P/generator copies are constructed and no existing model files
are edited. Routing masks the generated rank channels, before the original
P multiplication, so the original 32-layer Q/K/V/O path is preserved.

E1: deterministic user-hash Top1 (capacity/sparsity negative control)
E2: train-only user-prototype Top1; cold user occupies the least-similar slot
    via a non-destructive user-generated projection (NO parameter overwrite)
E3: trainable user Top1 with a straight-through softmax router
E4: trainable user Top2, matched total rank budget
E5: depth coarse-to-fine (L1-L16 all 4, L17-L24 Top2, L25-L32 Top1)

Reference: the already-run RR1_L32_BASE uses the exact same D4+C6 backbone.
All five trials start fresh, train serially on physical GPU0, seed42, VALID
only, with independent configs/logs/trainable-only best checkpoints.

Examples:
  python experiments/no31_cora_original/cora_rank4_experts5.py --cpu-test
  python experiments/no31_cora_original/cora_rank4_experts5.py --plan
  python -u experiments/no31_cora_original/cora_rank4_experts5.py --run
  python -u experiments/no31_cora_original/cora_rank4_experts5.py --run --only E3_USER_TOP1,E5_DEPTH_C2F

Run a separate smoke protocol, never overwrite finished training:
  python -u experiments/no31_cora_original/cora_rank4_experts5.py \
      --run --screen-epochs 2 --protocol CORA_RANK4_EXPERTS5_SMOKE_20261103
"""

import argparse
import csv
import json
import math
import os
import re
import runpy
import subprocess
import sys
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


def _project_root():
    here = Path(__file__).resolve().parent
    for parent in (here, *here.parents):
        if (
            (parent / "train_collm_mf_din.py").is_file()
            and (parent / "minigpt4" / "models" / "minigpt4rec_vx.py").is_file()
            and (parent / "minigpt4" / "tasks" / "base_task.py").is_file()
        ):
            return parent
    raise RuntimeError("Expected CoRA_official source tree around experiments/no31_cora_original")


PROJECT_ROOT = _project_root()
EXP_DIR = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import cora_latevote4 as late  # noqa: E402


DEFAULT_PROTOCOL = "CORA_RANK4_EXPERTS5_20261103"
PROTOCOL = DEFAULT_PROTOCOL
SEED = 42
CORA_LR = 3e-4
MF_LR_RATIO = 0.1
PATIENCE = 10
STEPS_PER_EPOCH = 400
BATCH_SIZE = 2
ACCUMULATION = 4
RANK = 16
N_EXPERTS = 4
RANK_PER_EXPERT = 4
ROUTER_HIDDEN = 64
BALANCE_WEIGHT = 1e-3
MIN_FREE_GPU0_MIB = 22000
EXPERT_NAMES = ("E1_HASH_TOP1", "E2_SIM_TOP1", "E3_USER_TOP1",
                "E4_USER_TOP2", "E5_DEPTH_C2F")
TRIALS = [
    dict(name="E1_HASH_TOP1", expert_mode="hash",
         mode="neighbor_mix", personal_share=0.25),
    dict(name="E2_SIM_TOP1", expert_mode="prototype",
         mode="neighbor_mix", personal_share=0.25),
    dict(name="E3_USER_TOP1", expert_mode="learn_top1",
         mode="neighbor_mix", personal_share=0.25),
    dict(name="E4_USER_TOP2", expert_mode="learn_top2",
         mode="neighbor_mix", personal_share=0.25),
    dict(name="E5_DEPTH_C2F", expert_mode="depth_c2f",
         mode="neighbor_mix", personal_share=0.25),
]
TRIAL_BY_NAME = {trial["name"]: trial for trial in TRIALS}
ORDER = [trial["name"] for trial in TRIALS]
LEARNED = {"learn_top1", "learn_top2", "depth_c2f"}


def configure(protocol):
    """Point tested C6 + D4 runtime to isolated experiment artifacts."""
    global PROTOCOL
    if not re.fullmatch(r"[A-Za-z0-9_-]+", protocol):
        raise ValueError("Protocol must contain only letters, numbers, '_' and '-'")
    PROTOCOL = protocol
    late.ROOT = EXP_DIR
    late.PROTOCOL = PROTOCOL
    late.SEED = SEED
    late.CORA_LR = CORA_LR
    late.MF_LR_RATIO = MF_LR_RATIO
    late.PATIENCE = PATIENCE
    late.TRIALS = TRIALS
    late.TRIAL_BY_NAME = TRIAL_BY_NAME
    late.ORDER = ORDER
    late.base.ROOT = EXP_DIR
    late.base.core.ROOT = PROJECT_ROOT
    late.base.combo.ROOT = EXP_DIR
    late.configure()


def _ids_to_mask(ids, top_k):
    """Construct rank masks with variance-matched active expert capacity."""
    if ids.ndim != 2 or ids.shape[1] != top_k:
        raise ValueError("ids must have shape [B, top_k]")
    mask = torch.zeros(ids.shape[0], N_EXPERTS, device=ids.device,
                       dtype=torch.float32).scatter_(1, ids.long(), 1.0)
    # If 4 independent rank-4 experts have equal contribution variance,
    # sqrt(4/k) matches the dense rank-16 residual variance.
    mask = mask * math.sqrt(N_EXPERTS / top_k)
    return mask.repeat_interleave(RANK_PER_EXPERT, dim=-1)


def _ste_mask(logits, top_k):
    """Hard TopK forward; softmax straight-through gradient for user router."""
    probs = torch.softmax(logits.float(), dim=-1)
    indices = probs.topk(top_k, dim=-1).indices
    hard = torch.zeros_like(probs).scatter_(1, indices, 1.0)
    # Same expert-count mass in hard and differentiable surrogate.
    surrogate = probs * top_k
    straight_through = hard + surrogate - surrogate.detach()
    channel_mask = (straight_through * math.sqrt(N_EXPERTS / top_k))
    return channel_mask.repeat_interleave(RANK_PER_EXPERT, dim=-1), probs, indices


class Rank4Experts(nn.Module):
    """Router only. Original CoRA P and weight-generator remain rank16."""

    def __init__(self, mode, user_dim, prototype_labels=None, prototypes=None):
        super().__init__()
        self.mode = mode
        self.user_dim = int(user_dim)
        self._grad_seen = set()
        self._grad_nonfinite = set()
        if mode in LEARNED:
            n_heads = 2 if mode == "depth_c2f" else 1
            self.router = nn.Sequential(
                nn.LayerNorm(user_dim),
                nn.Linear(user_dim, ROUTER_HIDDEN),
                nn.GELU(),
                nn.Linear(ROUTER_HIDDEN, N_EXPERTS * n_heads),
            )
            nn.init.normal_(self.router[-1].weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.router[-1].bias)
            for name, param in self.named_parameters():
                param.register_hook(self._gradient_observer(name))
        if mode == "prototype":
            if prototype_labels is None or prototypes is None:
                raise RuntimeError("Prototype routing requires train-only user prototypes")
            self.register_buffer("train_cluster_labels", prototype_labels.long(),
                                 persistent=False)
            self.register_buffer("prototype_centers", prototypes.float(),
                                 persistent=False)

    def _gradient_observer(self, name):
        def observe(gradient):
            self._grad_seen.add(name)
            if not bool(torch.isfinite(gradient).all()):
                self._grad_nonfinite.add(name)
            # Do not preempt AMP GradScaler's overflow handling.
            return gradient
        return observe

    def forward(self, user_embeddings, user_ids):
        if user_embeddings.ndim != 2 or user_embeddings.shape[1] != self.user_dim:
            raise RuntimeError(
                f"Expected user MF tensor [B,{self.user_dim}], "
                f"got {tuple(user_embeddings.shape)}"
            )
        batch = user_embeddings.shape[0]
        ids = user_ids.long().reshape(-1).to(user_embeddings.device)
        if ids.numel() != batch:
            raise RuntimeError("UserID count disagrees with user MF embedding batch")
        mode = self.mode
        aux = user_embeddings.new_zeros((), dtype=torch.float32)
        cold_count = 0

        if mode == "hash":
            choices = torch.remainder(ids, N_EXPERTS).unsqueeze(1)
            channel = _ids_to_mask(choices, 1)
            masks = [channel] * 4
            summary = [choices.reshape(-1)]
        elif mode == "prototype":
            labels = self.train_cluster_labels
            if bool((ids < 0).any()) or bool((ids >= labels.numel()).any()):
                raise RuntimeError("UserID outside MF user table/prototype labels")
            choices = labels.index_select(0, ids)
            cold = choices < 0
            cold_count = int(cold.sum().item())
            if cold_count:
                cold_embeddings = F.normalize(
                    user_embeddings.detach().float()[cold], dim=-1, eps=1e-8
                )
                similarity = cold_embeddings @ self.prototype_centers.float().T
                # A new user takes a least-related expert SLOT only for this
                # example. No expert P weights are overwritten or evicted.
                choices = choices.clone()
                choices[cold] = similarity.argmin(dim=-1)
            selected = choices.unsqueeze(1)
            channel = _ids_to_mask(selected, 1)
            masks = [channel] * 4
            summary = [selected.reshape(-1)]
        else:
            # Only raw user MF, not target item or ground-truth label.
            users = F.normalize(
                user_embeddings.detach().float(), dim=-1, eps=1e-8
            )
            logits = self.router(users)
            if mode == "learn_top1":
                channel, probs, choice = _ste_mask(logits, 1)
                masks = [channel] * 4
                summary = [choice.reshape(-1)]
                aux = BALANCE_WEIGHT * (probs.mean(dim=0) - .25).square().sum()
            elif mode == "learn_top2":
                channel, probs, choice = _ste_mask(logits, 2)
                masks = [channel] * 4
                summary = [choice.reshape(-1)]
                aux = BALANCE_WEIGHT * (probs.mean(dim=0) - .25).square().sum()
            elif mode == "depth_c2f":
                logits = logits.reshape(batch, 2, N_EXPERTS)
                mid, pm, im = _ste_mask(logits[:, 0, :], 2)
                deep, pd, idp = _ste_mask(logits[:, 1, :], 1)
                dense = torch.ones_like(mid)
                # 32 decoder layers grouped in four blocks of 8:
                # L1-L16 dense (coarse), L17-L24 Top2, L25-L32 Top1.
                masks = [dense, dense, mid, deep]
                summary = [torch.arange(N_EXPERTS, device=ids.device).repeat(batch),
                           im.reshape(-1), idp.reshape(-1)]
                aux = BALANCE_WEIGHT * (
                    (pm.mean(dim=0) - .25).square().sum()
                    + (pd.mean(dim=0) - .25).square().sum()
                )
            else:
                raise ValueError(f"Unsupported expert mode: {mode}")

        # Log only detached metadata so no residual GPU graph is retained.
        count = [torch.bincount(x.detach().reshape(-1), minlength=N_EXPERTS)
                 .cpu().tolist() for x in summary]
        stats = {"expert_counts": count, "cold_users_in_batch": cold_count}
        return masks, aux, stats


def _train_prototypes(rec):
    """Compute four exclusive expert assignments from train_ood2 users only."""
    seen_users, _, source = late.base._load_train_entity_ids()
    table = rec.user_embedding.weight
    groups = late.base._kmeans_cosine(
        table, N_EXPERTS, seed=SEED + 19, seen_ids=seen_users
    )
    labels = groups["labels"].clone()
    centers = []
    for members in groups["members"]:
        if len(members) == 0:
            raise RuntimeError("Empty expert cluster in train-only 4-means")
        average = table.detach().float().cpu().index_select(
            0, members.long()
        ).mean(dim=0)
        centers.append(F.normalize(average, dim=-1, eps=1e-8))
    centers = torch.stack(centers)
    print(
        f"[R4E PROTOTYPES] train_source={source} "
        f"seen_users={len(seen_users)} sizes={[len(m) for m in groups['members']]} "
        f"unknown_rows={(labels < 0).sum().item()}; no valid/test leakage",
        flush=True,
    )
    return labels, centers


def install_model(trial):
    """Install router after D4 init but before RunnerBase builds optimizer."""
    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx
    from minigpt4.models.ppeft.weight_generator import PerceptualWeightsGenerator

    if getattr(MiniGPT4Rec_vx, "_r4e_patched", False):
        raise RuntimeError("Rank4 model patch was already installed")
    old_init = MiniGPT4Rec_vx.__init__
    old_generator_forward = PerceptualWeightsGenerator.forward

    def init_experts(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        generator = self.llama_model_lora.base_model.weight_generator
        if int(generator.r) != RANK:
            raise RuntimeError(f"Expected original CoRA generator rank={RANK}")
        rec = self.rec_encoder
        if not hasattr(rec, "user_embedding"):
            raise RuntimeError("Rank4 experts require MF user embedding")
        user_dim = int(rec.user_embedding.weight.shape[1])
        kwargs_ = {}
        if trial["expert_mode"] == "prototype":
            labels, centers = _train_prototypes(rec)
            kwargs_ = dict(prototype_labels=labels, prototypes=centers)
        if hasattr(generator, "rank4_experts"):
            raise RuntimeError("Rank4 experts already attached")
        generator.rank4_experts = Rank4Experts(
            trial["expert_mode"], user_dim, **kwargs_
        )
        adapter_params = [
            (name, p) for name, p in self.llama_model_lora.named_parameters()
            if "lora_P" in name and p.requires_grad
        ]
        print(
            f"[R4E INIT] {trial['name']} mode={trial['expert_mode']} "
            f"total_rank={RANK} experts={N_EXPERTS} rank_per_expert={RANK_PER_EXPERT} "
            f"adapter_tensors={len(adapter_params)} "
            f"new_router_params={sum(p.numel() for p in generator.rank4_experts.parameters())} "
            "no adapter P copies",
            flush=True,
        )

    def masked_generator(self, ui_embedding):
        result = old_generator_forward(self, ui_embedding)
        ctx = getattr(self, "_rankgran_ctx", None)
        if ctx is None:
            # Initialization or an external direct generator call; no user ids.
            return result
        if not hasattr(self, "rank4_experts"):
            raise RuntimeError("Rank4 expert module missing at generator forward")
        rec = ctx["rec_encoder"]
        ids = ctx["user_ids"].long().to(rec.user_embedding.weight.device)
        user_embeddings = rec.user_embedding.weight.index_select(0, ids.view(-1))
        masks, aux, summary = self.rank4_experts(user_embeddings, ids)
        self._r4e_aux = aux
        self._r4e_stats = summary
        self._r4e_calls = getattr(self, "_r4e_calls", 0) + 1

        cache = {}
        def mask_weight(w, group_idx):
            if not torch.is_tensor(w) or w.ndim != 3 or w.shape[-1] != RANK:
                raise RuntimeError(
                    f"Unexpected generated CoRA weight, expected [B,H,{RANK}]: "
                    f"{type(w)}, {tuple(w.shape) if torch.is_tensor(w) else ''}"
                )
            key = (id(w), group_idx)
            if key not in cache:
                # Mask the SAME rank axis used by the existing P rank-16
                # projection; P's four 4-column blocks are the four experts.
                mask = masks[group_idx].unsqueeze(1).to(dtype=w.dtype)
                if mask.shape[0] != w.shape[0]:
                    raise RuntimeError("Batch dimension mismatch in generated CoRA masks")
                cache[key] = w * mask
            return cache[key]

        if isinstance(result, (list, tuple)):
            if len(result) != 32:
                raise RuntimeError(f"Expected 32 layer weights, got {len(result)}")
            return [mask_weight(w, idx // 8) for idx, w in enumerate(result)]
        # Only used in direct uniform-CoRA mode (not this D4 study).
        return mask_weight(result, 3)

    MiniGPT4Rec_vx.__init__ = init_experts
    PerceptualWeightsGenerator.forward = masked_generator
    MiniGPT4Rec_vx._r4e_patched = True


def install_loss_audit(trial):
    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx
    from minigpt4.runners.runner_base import RunnerBase

    original_forward = MiniGPT4Rec_vx.forward_v2

    def forward(self, samples, *args, **kwargs):
        gen = self.llama_model_lora.base_model.weight_generator
        start_calls = getattr(gen, "_r4e_calls", 0)
        result = original_forward(self, samples, *args, **kwargs)
        if getattr(gen, "_r4e_calls", 0) <= start_calls:
            raise RuntimeError(
                "Rank4 expert generator was not invoked during the CoRA training forward"
            )
        if "_r4e_aux" not in gen:
            raise RuntimeError("Missing Rank4 route auxiliary loss")
        aux = gen._r4e_aux
        result["loss"] = result["loss"] + aux
        stats = gen._r4e_stats
        delattr(gen, "_r4e_aux")
        delattr(gen, "_r4e_stats")

        step = getattr(self, "_r4e_train_steps", 0)
        expert = gen.rank4_experts
        # Aggregate utilization over a window; B=2 per-step expert counts
        # are too noisy to distinguish routing specialization/collapse.
        counts = stats["expert_counts"]
        usage = getattr(self, "_r4e_usage_window", None)
        if usage is None:
            usage = [[0] * N_EXPERTS for _ in counts]
        if len(usage) != len(counts):
            raise RuntimeError("Expert group utilization layout changed mid-training")
        for row, current in zip(usage, counts):
            for k, amount in enumerate(current):
                row[k] += int(amount)
        self._r4e_usage_window = usage
        self._r4e_cold_window = (
            getattr(self, "_r4e_cold_window", 0) + stats["cold_users_in_batch"]
        )
        if step == 1 and not getattr(self, "_r4e_grad_checked", False):
            expected = {n for n, p in expert.named_parameters() if p.requires_grad}
            missing = expected - expert._grad_seen
            if missing:
                raise RuntimeError(
                    f"Learnable router gradients missing from backward: {sorted(missing)}"
                )
            print(
                f"[R4E GRAD PASS] trial={trial['name']} "
                f"params={len(expected)} all_connected=True "
                f"nonfinite_seen={sorted(expert._grad_nonfinite)}",
                flush=True,
            )
            self._r4e_grad_checked = True

        self._r4e_train_steps = step + 1
        if step == 0 or (step + 1) % 100 == 0:
            print(
                f"[R4E TRAIN] trial={trial['name']} step={step+1} "
                f"loss={float(result['loss'].detach()):.6f} "
                f"balance_aux={float(aux.detach()):.6g} "
                f"expert_counts_window={self._r4e_usage_window} "
                f"cold_users_window={self._r4e_cold_window}",
                flush=True,
            )
            self._r4e_usage_window = [[0] * N_EXPERTS for _ in counts]
            self._r4e_cold_window = 0
        return result

    MiniGPT4Rec_vx.forward_v2 = forward

    previous_optimizer = RunnerBase.optimizer
    if not isinstance(previous_optimizer, property):
        raise RuntimeError("Expected RunnerBase.optimizer to be a property")
    previous_getter = previous_optimizer.fget

    def audited_optimizer(self):
        optimizer = previous_getter(self)
        if getattr(self, "_r4e_optimizer_verified", False):
            return optimizer
        model = self.unwrap_dist_model(self.model)
        expert = model.llama_model_lora.base_model.weight_generator.rank4_experts
        needed = {id(p) for p in expert.parameters() if p.requires_grad}
        found = {id(p) for g in optimizer.param_groups for p in g["params"]}
        if needed - found:
            raise RuntimeError("Learnable Rank4 router params missing from optimizer")
        if not any(group.get("joint8_mf") for group in optimizer.param_groups):
            raise RuntimeError("Joint8 MF learning-rate split missing")
        self._r4e_optimizer_verified = True
        print(
            f"[R4E OPT PASS] trial={trial['name']} "
            f"new_router_params={len(needed)} in_optimizer=True "
            f"MF_LR_ratio={MF_LR_RATIO}",
            flush=True,
        )
        return optimizer

    RunnerBase.optimizer = property(audited_optimizer)


def _gpu0_preflight(stage):
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"GPU preflight failed: {proc.stderr.strip()}")
    free = None
    for line in proc.stdout.splitlines():
        fields = [s.strip() for s in line.split(",")]
        if len(fields) >= 2 and fields[0] == "0":
            free = int(fields[1])
            break
    if free is None:
        raise RuntimeError("Physical GPU0 absent from nvidia-smi")
    print(
        f"[R4E GPU] {stage} GPU0_free={free}MiB required>={MIN_FREE_GPU0_MIB}MiB",
        flush=True,
    )
    if free < MIN_FREE_GPU0_MIB:
        raise RuntimeError(
            f"GPU0 only {free} MiB free (requires >= {MIN_FREE_GPU0_MIB} MiB); "
            "no model loading. Inspect nvidia-smi; do not kill unrelated jobs."
        )


def _check_config(path):
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    run = cfg["run"]
    checks = {
        "seed": int(run["seed"]) == SEED,
        "trainableMF": cfg["model"]["freeze_rec"] is False,
        "batch": int(run["batch_size_train"]) == BATCH_SIZE,
        "accum": int(run["accum_grad_iters"]) == ACCUMULATION,
        "steps": int(run["iters_per_epoch"]) == STEPS_PER_EPOCH,
        "lr": abs(float(run["init_lr"]) - CORA_LR) < 1e-12,
        "valid": run.get("valid_splits") == ["valid"] and not run.get("test_splits"),
        "fresh": not run.get("resume_ckpt_path") and not cfg["model"].get("ckpt"),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"Config mismatch: {path}: " +
            str([k for k, ok in checks.items() if not ok])
        )
    return cfg


def plan(protocol):
    configure(protocol)
    late.base.core.check_sources()
    late.base._model_api_preflight()
    late.base.core.prepare()
    paths = []
    for trial in TRIALS:
        cfg_path = late.base.core.CONFIG_DIR / (trial["name"] + ".yaml")
        cfg = _check_config(cfg_path)
        paths.append(str(cfg_path))
    manifest = {
        "protocol": PROTOCOL,
        "source": "official CoRA + existing C6/D4 runner",
        "experiment": "4 rank4 experts partitioning the SAME 16 P rank channels and dynamic W_ui",
        "non_destructive_cold_user": (
            "train-only four-prototype experts; a cold user selects the "
            "least-similar expert slot for the user's own dynamic CoRA W_ui. "
            "The shared P expert weights are NOT overwritten."
        ),
        "reference": "RR1_L32_BASE Valid AUC=.732387, UAUC=.701999 (prior independent run)",
        "seed": SEED,
        "gpu": "0",
        "valid_only": True,
        "test_splits": [],
        "input": "D4_TOPK_RES025 layer group neighbors top40/top20/top10/top5",
        "loss": "C6_KD_BOTH: LLM BCE + 0.005 MF CTR + 0.010 gated MF KD; learned routers add mean-probability expert balance",
        "rank": RANK,
        "experts": N_EXPERTS,
        "rank_per_expert": RANK_PER_EXPERT,
        "mask_scaling": "sqrt(4/active_experts) to match dense variance",
        "train": {
            "CoRA_LR": CORA_LR,
            "MF_LR": CORA_LR * MF_LR_RATIO,
            "batch": BATCH_SIZE,
            "accumulation": ACCUMULATION,
            "iters_per_epoch": STEPS_PER_EPOCH,
            "patience": PATIENCE,
        },
        "trials": TRIALS,
        "config_paths": paths,
    }
    target = late.base.core.LOG_DIR / "rank4_experts_manifest.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"[R4E PLAN] configs={late.base.core.CONFIG_DIR}")
    print(f"[R4E PLAN] logs={late.base.core.LOG_DIR}")
    print(f"[R4E PLAN] outputs={late.base.core.OUT_DIR}")
    for trial in TRIALS:
        print(f"[R4E TRIAL] {trial['name']} mode={trial['expert_mode']}")
    print(f"[R4E PLAN PASS] {len(TRIALS)} independent configs; Valid only; no GPU training")


def worker(name, protocol, patience, screen_epochs):
    configure(protocol)
    _gpu0_preflight("worker-before-model")
    if name not in TRIAL_BY_NAME:
        raise ValueError(f"Unknown experiment: {name}")
    trial = TRIAL_BY_NAME[name]
    path = late.base.core.CONFIG_DIR / (name + ".yaml")
    if not path.is_file():
        raise FileNotFoundError(f"Run --plan first: {path}")
    cfg = _check_config(path)
    late.base.core.patch_runner(
        screen_epochs=screen_epochs,
        mf_lr_ratio=MF_LR_RATIO,
        patience=patience,
    )
    late.base.core.install_compact_atomic_best_checkpoint()
    late.base.combo.install_combo_model(late.base.combo.NAMES["C6_KD_BOTH"])
    late.base._patch_runtime_classes(trial)
    install_model(trial)
    install_loss_audit(trial)
    print(
        f"[R4E WORKER] {name} output={cfg['run']['output_dir']} "
        "C6+D4 seed42 GPU0 Valid-only; Test disabled",
        flush=True,
    )
    sys.argv = ["train_collm_mf_din.py", "--cfg-path", str(path)]
    runpy.run_path(str(PROJECT_ROOT / "train_collm_mf_din.py"), run_name="__main__")


def run_selected(names, protocol, patience, screen_epochs):
    configure(protocol)
    if not late.base.core.CONFIG_DIR.is_dir():
        plan(protocol)
    for name in names:
        if name not in TRIAL_BY_NAME:
            raise ValueError(f"Unknown experiment: {name}")
        log = late.base.core.LOG_DIR / f"{name}.log"
        if log.exists():
            raise RuntimeError(
                f"Refusing to overwrite {log}; use a new --protocol for reruns."
            )
    for name in names:
        _gpu0_preflight(f"before-{name}")
        logfile = late.base.core.LOG_DIR / f"{name}.log"
        command = [
            sys.executable, "-u", str(SCRIPT),
            "--_worker", name,
            "--protocol", protocol,
            "--patience", str(patience),
            "--screen-epochs", str(screen_epochs),
        ]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
        print(f"[R4E START] {name}; serial GPU0; log={logfile}", flush=True)
        with logfile.open("x", encoding="utf-8") as fp:
            process = subprocess.Popen(
                command, cwd=PROJECT_ROOT, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
            for line in process.stdout:
                print(line, end="", flush=True)
                fp.write(line)
            code = process.wait()
        if code != 0:
            raise SystemExit(
                f"[R4E STOP] {name} exit={code}; inspect {logfile}. "
                "No later experiments were started."
            )
    print("[R4E COMPLETE] five serial Valid-only experiments finished.")


AUC_PAIR = re.compile(
    r"\*\*\*auc:\s*([0-9.]+)\s+\*\*\*uauc:\s*\(?([0-9.]+)",
    re.I,
)
EPOCH_RE = re.compile(r"Start training epoch\s+(\d+)")
FAILED_RE = re.compile(r"Traceback \(most recent call last\)|^\s*RuntimeError:", re.M)


def summary(protocol):
    configure(protocol)
    root = late.base.core.OUT_DIR
    log_root = late.base.core.LOG_DIR
    rows = []
    for trial in TRIALS:
        name = trial["name"]
        logfile = log_root / (name + ".log")
        records = []
        if (root / name).exists():
            for result_path in sorted((root / name).rglob("valid_log.txt")):
                try:
                    for line_no, line in enumerate(result_path.read_text(
                            encoding="utf-8", errors="replace").splitlines(), 1):
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        auc = entry.get("valid_agg_metrics")
                        uauc = entry.get("valid_uauc")
                        if isinstance(uauc, (tuple, list)):
                            uauc = uauc[0] if uauc else None
                        if isinstance(uauc, dict):
                            uauc = uauc.get("score")
                        if auc is not None:
                            records.append((
                                float(auc),
                                float(uauc) if uauc is not None else None,
                                result_path.parent,
                                f"{result_path}:{line_no}",
                                entry.get("valid_best_epoch"),
                            ))
                except OSError:
                    pass
        best = max(records, key=lambda x: x[0]) if records else None
        ckpt = (
            best is not None
            and (best[2] / "checkpoint_best.pth").is_file()
            and zipfile.is_zipfile(best[2] / "checkpoint_best.pth")
        )
        status = "NO LOG"
        if logfile.exists():
            text = logfile.read_text(encoding="utf-8", errors="replace")
            if FAILED_RE.search(text):
                status = "FAILED"
            elif "Training time" in text or "[JOINT8 EARLY STOP]" in text:
                status = "COMPLETE"
            else:
                status = "INCOMPLETE"
        rows.append({
            "trial": name,
            "valid_auc": best[0] if best else None,
            "valid_uauc_at_best_auc": best[1] if best else None,
            "best_epoch": best[4] if best else None,
            "delta_auc_vs_RR1": best[0] - .732387 if best else None,
            "delta_uauc_vs_RR1": best[1] - .701999 if best and best[1] is not None else None,
            "status": status,
            "ckpt": "FOUND" if ckpt else "MISSING",
            "source": best[3] if best else "",
        })
    print("\nRANK4 EXPERTS (VALID ONLY; best AUC and paired UAUC)")
    print(f"{'trial':22} {'AUC':>10} {'UAUC@best':>12} {'dAUC':>10} {'dUAUC':>10} {'epoch':>6} {'status':>12} {'ckpt':>10}")
    for row in rows:
        auc = row["valid_auc"]
        ua = row["valid_uauc_at_best_auc"]
        print(
            f"{row['trial']:22} "
            f"{auc:.6f} " if auc is not None else f"{row['trial']:22} {'—':>10} ",
            end=""
        )
        print(
            f"{ua:.6f}" if ua is not None else "—",
            f"{row['delta_auc_vs_RR1']:+.6f}" if row["delta_auc_vs_RR1"] is not None else "—",
            f"{row['delta_uauc_vs_RR1']:+.6f}" if row["delta_uauc_vs_RR1"] is not None else "—",
            str(row["best_epoch"]) if row["best_epoch"] is not None else "—",
            row["status"], row["ckpt"],
        )
    report_dir = EXP_DIR / "out" / PROTOCOL
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "rank4_expert_valid_summary.csv").open(
            "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("[R4E SUMMARY] CSV:", report_dir / "rank4_expert_valid_summary.csv")
    print("[R4E SUMMARY] Reference RR1: Valid AUC=.732387 UAUC=.701999 (prior separate run)")


def cpu_test():
    torch.manual_seed(SEED)
    batch, hidden, udim = 8, 32, 16
    w = torch.randn(batch, hidden, RANK, requires_grad=True)
    P = torch.randn(24, RANK, requires_grad=True)
    x = torch.randn(batch, hidden, requires_grad=True)

    # Rank16 == concatenation of exactly four rank4 experts (no copied P).
    wparts = w.split(RANK_PER_EXPERT, dim=-1)
    Pparts = P.split(RANK_PER_EXPERT, dim=-1)
    dense = F.linear(torch.bmm(x.unsqueeze(1), w).squeeze(1), P)
    experts = sum(
        F.linear(torch.bmm(x.unsqueeze(1), wi).squeeze(1), pi)
        for wi, pi in zip(wparts, Pparts)
    )
    torch.testing.assert_close(dense, experts, atol=1e-4, rtol=1e-5)
    assert len(wparts) == N_EXPERTS and len(Pparts) == N_EXPERTS

    ui = torch.randn(batch, udim)
    ids = torch.arange(batch)
    fake_labels = torch.tensor([0, 1, 2, 3, -1, 0, 1, -1])
    centers = F.normalize(torch.randn(N_EXPERTS, udim), dim=-1)
    for trial in TRIALS:
        mode = trial["expert_mode"]
        gate = Rank4Experts(
            mode, udim,
            prototype_labels=fake_labels if mode == "prototype" else None,
            prototypes=centers if mode == "prototype" else None,
        )
        masks, aux, stats = gate(ui, ids)
        assert len(masks) == 4
        assert all(m.shape == (batch, RANK) for m in masks)
        assert all(torch.isfinite(m).all() for m in masks)
        if mode == "depth_c2f":
            active = [4, 4, 2, 1]
        else:
            active = [2 if mode == "learn_top2" else 1] * 4
        for mask, k in zip(masks, active):
            groups = mask.reshape(batch, 4, RANK_PER_EXPERT)
            torch.testing.assert_close(
                groups, groups[:, :, :1].expand_as(groups)
            )
            assert bool((groups.abs() > .01).sum(dim=1).eq(k).all())
        assert torch.isfinite(aux)
        output = F.linear(torch.bmm(x.unsqueeze(1), w * masks[3].unsqueeze(1))
                          .squeeze(1), P)
        loss = output.square().mean() + aux
        loss.backward(retain_graph=True)
        if mode in LEARNED:
            params = list(gate.parameters())
            assert params and all(p.grad is not None for p in params)
            assert all(bool(torch.isfinite(p.grad).all()) for p in params)
        if mode == "prototype":
            assert stats["cold_users_in_batch"] == 2

    print("[R4E CPU PASS] rank16=4xrank4 identity; 5 routes; "
          "Top1/Top2/depth masks; prototype cold-slot assignment; "
          "straight-through learned-router gradients", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-test", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--only", default="")
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--screen-epochs", type=int, default=0)
    parser.add_argument("--_worker", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    if args.cpu_test:
        cpu_test()
        return
    if args._worker:
        worker(args._worker, args.protocol, args.patience, args.screen_epochs)
        return
    if args.patience < 1 or args.screen_epochs < 0:
        parser.error("patience must be >=1, screen-epochs >=0")
    if args.plan:
        plan(args.protocol)
        return
    if args.summary:
        summary(args.protocol)
        return
    if args.run:
        selected = (
            [name.strip() for name in args.only.split(",") if name.strip()]
            if args.only else list(ORDER)
        )
        if len(selected) != len(set(selected)) or any(x not in TRIAL_BY_NAME for x in selected):
            parser.error("Unknown or duplicate --only experiment name")
        configure(args.protocol)
        if not late.base.core.CONFIG_DIR.is_dir():
            plan(args.protocol)
        run_selected(selected, args.protocol, args.patience, args.screen_epochs)
        return
    parser.error("Specify --cpu-test, --plan, --run, or --summary")


if __name__ == "__main__":
    main()
