#!/usr/bin/env python3
"""HP24 warm-start routing improvements: W1/W2/W3, single-model, VALID only.

Each trial starts from the exact trainable state of HP24_B07_T05. The frozen
Vicuna weights are reused from the configured base model; no second LLM,
teacher/student, or optimizer-state resume is used.

W1: candidate-aware personal logit from [user, item, user*item].
W2: user-adaptive coarse/fine blend beta in [0.50, 0.90].
W3: zero-initialized personal-logit residual for four 8-layer groups.

A two-epoch router-only warmup aligns with the 800-step LR warmup. At epoch 2,
the original HP24 trainable parameters are unfrozen and added to the optimizer.
The best checkpoint always contains the complete HP24 state plus new gates.

Run from CoRA_official root:
  python experiments/no31_cora_original/cora_hp24_warmstart.py --check
  python experiments/no31_cora_original/cora_hp24_warmstart.py --cpu-test
  python experiments/no31_cora_original/cora_hp24_warmstart.py --plan
  CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/cora_hp24_warmstart.py --run
  python experiments/no31_cora_original/cora_hp24_warmstart.py --summary
  CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/cora_hp24_warmstart.py --test --only W1_PAIR_AWARE

The test command is separate and must only be used after valid-only selection.
"""
import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import nn

HERE = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()
ROOT_DIR = HERE.parent.parent
for _d in (str(ROOT_DIR), str(HERE), str(ROOT_DIR.parent)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import cora_hp_model6 as hp

BASE = hp.moe.base
RUNTIME = BASE.late.base
CORE = RUNTIME.core
COMBO = RUNTIME.combo
PROJECT_ROOT = BASE.PROJECT_ROOT

DEFAULT_PROTOCOL = "CORA_HP24_WARMSTART_20261120"
SOURCE_PROTOCOL = "CORA_HIER_FULLSTACK_20261113"
SOURCE_TRIAL = "HP24_B07_T05"
DEFAULT_TEST_PROTOCOL = "CORA_HP24_WARMSTART_TEST_20261120"

SEED = 42
CORA_LR = 1e-4
MF_LR_RATIO = 0.1
MIN_LR = 2e-5
WARMUP_LR = 5e-6
WARMUP_STEPS = 800
ROUTER_WARMUP_EPOCHS = 2
PATIENCE = 10
STEPS_PER_EPOCH = 400
BATCH_SIZE = 2
ACCUMULATION = 4
RANK = 16
EXPERTS = 4
RANK_PER = 4
K1 = 3
K2_PER_GROUP = 8

TRIALS = [
    dict(name="W1_PAIR_AWARE", expert_mode="hier_personal",
         model_variant="baseline", warm_variant="pair",
         route="hier_personal", route_lambda=0.2, route_temp=0.50,
         init_temp=0.50, personal_target=0.40, route_fine=True,
         beta=0.70, mode="neighbor_mix", personal_share=0.25),
    dict(name="W2_ADAPTIVE_BETA", expert_mode="hier_personal",
         model_variant="baseline", warm_variant="adaptive_beta",
         route="hier_personal", route_lambda=0.2, route_temp=0.50,
         init_temp=0.50, personal_target=0.40, route_fine=True,
         beta=0.70, mode="neighbor_mix", personal_share=0.25),
    dict(name="W3_LAYER_PERSONAL", expert_mode="hier_personal",
         model_variant="baseline", warm_variant="layer_personal_residual",
         route="hier_personal", route_lambda=0.2, route_temp=0.50,
         init_temp=0.50, personal_target=0.40, route_fine=True,
         beta=0.70, mode="neighbor_mix", personal_share=0.25),
]
ORDER = [t["name"] for t in TRIALS]
TRIAL_BY_NAME = {t["name"]: t for t in TRIALS}
LOG_LINE = re.compile(r"\*\*\*auc:\s*([0-9.]+)\s+\*\*\*uauc:\s*\(?([0-9.]+)", re.I)
BEST_LINE = re.compile(r"\[JOINT8 BEST\]\s*epoch=(\d+)\s+valid_auc=([-+\d.eE]+)")


def _set_learning_rate_constants():
    # Keep all inherited config validators and LR manifests consistent.
    BASE.CORA_LR = CORA_LR
    BASE.late.CORA_LR = CORA_LR
    RUNTIME.CORA_LR = CORA_LR
    CORE.CORA_LR = CORA_LR
    BASE.MF_LR_RATIO = MF_LR_RATIO
    BASE.late.MF_LR_RATIO = MF_LR_RATIO
    RUNTIME.MF_LR_RATIO = MF_LR_RATIO
    CORE.MF_LR_RATIO = MF_LR_RATIO


def configure(protocol):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", protocol):
        raise ValueError("Protocol must contain only letters, digits, '_' and '-'")
    _set_learning_rate_constants()
    hp.DEFAULT_PROTOCOL = protocol
    hp.TRIALS = TRIALS
    hp.TRIAL_BY_NAME = TRIAL_BY_NAME
    hp.ORDER = ORDER
    hp.configure(protocol)
    BASE.TRIALS = TRIALS
    BASE.TRIAL_BY_NAME = TRIAL_BY_NAME
    BASE.ORDER = ORDER
    BASE.SCRIPT = SCRIPT
    BASE.install_model = install_model
    BASE.install_loss_audit = hp.fac.install_audit


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as reader:
        for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def find_hp24_checkpoint(explicit=None):
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        folder = HERE / "output" / SOURCE_PROTOCOL / SOURCE_TRIAL
        candidates = sorted(folder.rglob("checkpoint_best.pth")) if folder.is_dir() else []
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one HP24 checkpoint under {folder}; found {len(candidates)}. "
                "Pass --checkpoint with the exact checkpoint_best.pth path."
            )
        path = candidates[0].resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"HP24 checkpoint missing or empty: {path}")
    if SOURCE_TRIAL not in str(path):
        raise RuntimeError(
            f"Refusing non-HP24 checkpoint path: {path}; expected source trial {SOURCE_TRIAL}"
        )
    payload = _torch_load(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise RuntimeError("HP24 checkpoint must contain a model state dictionary")
    if payload.get("next8_checkpoint_format") != "trainable_model_only_v1":
        raise RuntimeError("HP24 checkpoint is not the verified trainable_model_only_v1 format")
    if payload.get("optimizer_included") is not False:
        raise RuntimeError("Expected HP24 model-only checkpoint without optimizer state")
    cfg = payload.get("config", {})
    run_cfg = cfg.get("run", {}) if isinstance(cfg, dict) else {}
    if int(run_cfg.get("seed", -1)) != SEED:
        raise RuntimeError(f"HP24 checkpoint seed mismatch: {run_cfg.get('seed')}")
    if SOURCE_TRIAL not in str(run_cfg.get("output_dir", "")):
        raise RuntimeError("Checkpoint metadata does not identify HP24_B07_T05 output")
    print(
        f"[HP24 SOURCE PASS] path={path} epoch={payload.get('epoch')} "
        f"tensors={len(payload['model'])} sha256={_sha256(path)}",
        flush=True,
    )
    return path, payload


def _new_router_from_hp24(old_router, variant):
    if old_router.mode != "hier_personal":
        raise RuntimeError(f"HP24 router mode mismatch: {old_router.mode}")
    personal = float(old_router.personal_logit.detach().float().cpu().item())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + {"pair": 2101, "adaptive_beta": 2102,
                                  "layer_personal_residual": 2103}[variant])
        router = WarmStartHierPersonalMoE(
            "hier_personal", old_router.dim,
            coarse=old_router.coarse_centers.detach().clone(),
            fine=old_router.fine_centers.detach().clone(),
            parent=old_router.fine_parent.detach().clone(),
            personal_logit_init=personal,
            route_temp=old_router.route_temp,
            route_fine=old_router.route_fine,
            beta=old_router.beta,
            hybrid_residual=old_router.hybrid_residual,
            warm_variant=variant,
        )
    result = router.load_state_dict(old_router.state_dict(), strict=False)
    expected_missing = set(router.warm_parameter_local_names())
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise RuntimeError(
            f"HP24 router transplant mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    return router


class WarmStartHierPersonalMoE(hp.HierPersonalMoE):
    """HP24-compatible router with one zero-initialized additive capability."""

    def __init__(self, mode, dim, labels=None, centers=None, coarse=None,
                 fine=None, parent=None, personal_logit_init=hp.PERSONAL_INIT_LOGIT,
                 route_temp=None, route_fine=False, beta=hp.COARSE_BETA,
                 hybrid_residual=False, warm_variant="pair"):
        super().__init__(
            mode, dim, labels=labels, centers=centers, coarse=coarse,
            fine=fine, parent=parent, personal_logit_init=personal_logit_init,
            route_temp=route_temp, route_fine=route_fine, beta=beta,
            hybrid_residual=hybrid_residual, route_variant="shared",
            model_variant="baseline",
        )
        if warm_variant not in ("pair", "adaptive_beta", "layer_personal_residual"):
            raise ValueError(f"Unknown warm-start route: {warm_variant}")
        self.warm_variant = warm_variant
        if warm_variant == "pair":
            self.pair_gate = nn.Sequential(
                nn.Linear(3 * self.dim, 64), nn.GELU(), nn.Linear(64, 1)
            )
            nn.init.zeros_(self.pair_gate[-1].weight)
            nn.init.zeros_(self.pair_gate[-1].bias)
        elif warm_variant == "adaptive_beta":
            self.beta_gate = nn.Sequential(
                nn.Linear(self.dim, 64), nn.GELU(), nn.Linear(64, 1)
            )
            nn.init.zeros_(self.beta_gate[-1].weight)
            nn.init.zeros_(self.beta_gate[-1].bias)
        else:
            # Keep the original scalar personal_logit key/shape for exact
            # HP24 checkpoint compatibility; add four zero residual offsets.
            self.group_personal_delta = nn.Parameter(torch.zeros(4))
        for local_name in self.warm_parameter_local_names():
            parameter = dict(self.named_parameters())[local_name]
            parameter.register_hook(self._grad_hook(local_name))

    def warm_parameter_local_names(self):
        if self.warm_variant == "pair":
            return [
                "pair_gate.0.weight", "pair_gate.0.bias",
                "pair_gate.2.weight", "pair_gate.2.bias",
            ]
        if self.warm_variant == "adaptive_beta":
            return [
                "beta_gate.0.weight", "beta_gate.0.bias",
                "beta_gate.2.weight", "beta_gate.2.bias",
            ]
        return ["group_personal_delta"]

    def warm_parameters(self):
        names = set(self.warm_parameter_local_names())
        return [p for n, p in self.named_parameters() if n in names]

    def forward(self, rec, user_ids, item_ids, samples):
        if self.mode != "hier_personal":
            return super().forward(rec, user_ids, item_ids, samples)
        user_table = rec.user_embedding.weight
        item_table = rec.item_embedding.weight
        uids = user_ids.long().to(user_table.device).reshape(-1)
        iids = item_ids.long().to(item_table.device).reshape(-1)
        if bool((uids < 0).any()) or bool((uids >= user_table.shape[0]).any()):
            raise RuntimeError("User ID outside MF user table")
        if bool((iids < 0).any()) or bool((iids >= item_table.shape[0]).any()):
            raise RuntimeError("Item ID outside MF item table")
        if uids.numel() != iids.numel():
            raise RuntimeError("User/item paired IDs have mismatched batch")
        user = F.normalize(
            user_table.index_select(0, uids).detach().float(),
            p=2, dim=-1, eps=1e-8
        )
        batch = user.shape[0]

        beta_user = None
        if self.warm_variant == "adaptive_beta":
            beta_user = 0.5 + 0.4 * torch.sigmoid(self.beta_gate(user))

        if self.route_fine:
            cos2 = user @ self.fine_centers.T
            vectors = []
            for group in range(int(self.coarse_centers.shape[0])):
                member = (self.fine_parent == group).float()
                fine_prob = F.softmax(
                    cos2 / hp.FINE_TEMP + (member - 1.0) * 1e4, dim=-1
                )
                fine_vector = fine_prob @ self.fine_centers
                beta_value = self.beta if beta_user is None else beta_user
                blended = (
                    beta_value * self.coarse_centers[group].unsqueeze(0)
                    + (1.0 - beta_value) * fine_vector
                )
                vectors.append(F.normalize(blended, p=2, dim=-1, eps=1e-8))
            route_vectors = torch.stack(vectors, dim=1)
            coarse_logits = (user.unsqueeze(1) * route_vectors).sum(dim=-1)
        else:
            coarse_logits = user @ self.coarse_centers.T

        if self.warm_variant == "pair":
            item = F.normalize(
                item_table.index_select(0, iids).detach().float(),
                p=2, dim=-1, eps=1e-8
            )
            pair = torch.cat((user, item, user * item), dim=-1)
            personal_logit = self.personal_logit.expand(batch, 1) + (
                0.1 * torch.tanh(self.pair_gate(pair))
            )
            logits = torch.cat(
                (coarse_logits / self.route_temp, personal_logit), dim=-1
            )
            probs = torch.softmax(logits.float(), dim=-1)
            probs4 = probs.unsqueeze(1).expand(-1, 4, -1)
        elif self.warm_variant == "adaptive_beta":
            personal_logit = self.personal_logit.expand(batch, 1)
            logits = torch.cat(
                (coarse_logits / self.route_temp, personal_logit), dim=-1
            )
            probs = torch.softmax(logits.float(), dim=-1)
            probs4 = probs.unsqueeze(1).expand(-1, 4, -1)
        else:
            group_personal = (
                self.personal_logit.reshape(1, 1)
                + self.group_personal_delta.reshape(1, 4)
            ).expand(batch, -1)
            group_logits = torch.cat(
                (
                    torch.stack([coarse_logits / self.route_temp] * 4, dim=1),
                    group_personal.unsqueeze(-1),
                ),
                dim=-1,
            )
            probs4 = torch.softmax(group_logits.float(), dim=-1)

        masks = 1.0 + hp.moe.RESIDUAL_LAMBDA * (EXPERTS * probs4 - 1.0)
        if not bool(torch.isfinite(masks).all()):
            raise FloatingPointError("Nonfinite HP24 warm-start expert masks")
        if self.warm_variant == "layer_personal_residual":
            route_average = probs4.mean(dim=0)
        else:
            route_average = probs.mean(dim=0)
        balance = hp.moe.BALANCE_WEIGHT * (
            route_average - 1.0 / EXPERTS
        ).square().sum()
        rank_masks = [
            masks[:, group, :].repeat_interleave(RANK_PER, dim=-1)
            for group in range(4)
        ]
        stats = {
            "expert_counts": [[batch] * EXPERTS for _ in range(4)],
            "cold_users_in_batch": 0,
            "empty_histories": 0,
        }
        return rank_masks, balance, stats, probs4.detach()


def _install_save_patch():
    from minigpt4.runners.runner_base import RunnerBase
    if getattr(RunnerBase, "_hp24_warmstart_save_patch", False):
        return
    original_save = RunnerBase._save_checkpoint
    try:
        original_parameters = inspect.signature(original_save).parameters
    except (TypeError, ValueError):
        original_parameters = {}

    def save_complete_warmstart(self, cur_epoch, is_best=False, *args, **kwargs):
        model = self.unwrap_dist_model(self.model)
        required = set(getattr(model, "_hp24_warmstart_save_names", ()))
        toggled = []
        if required:
            for name, parameter in model.named_parameters():
                if name in required and not parameter.requires_grad:
                    toggled.append(parameter)
                    parameter.requires_grad_(True)
        try:
            if "trainable_only" in original_parameters:
                return original_save(
                    self, cur_epoch, is_best=is_best,
                    trainable_only=kwargs.get("trainable_only", True)
                )
            return original_save(self, cur_epoch, is_best=is_best)
        finally:
            for parameter in toggled:
                parameter.requires_grad_(False)

    RunnerBase._save_checkpoint = save_complete_warmstart
    RunnerBase._hp24_warmstart_save_patch = True
    print(
        "[HP24 WARMSTART CKPT] save patch preserves frozen HP24 source tensors "
        "during router-only warmup",
        flush=True,
    )


def _parameter_part(name):
    if name.startswith("rec_encoder."):
        return "rec"
    if name.startswith("llama_proj."):
        return "proj"
    if name.startswith("gcn.") or name.startswith("gnn."):
        return "gnn"
    if name.startswith("r3_adapter."):
        return "cora"
    if ".cora_lora_P." in name:
        return "cora_p"
    if "lora_" in name or ".lora_" in name:
        return "lora"
    return "other"


def _install_router_warmup(warmup_epochs):
    from minigpt4.runners.runner_base import RunnerBase
    if warmup_epochs < 1:
        return
    if getattr(RunnerBase, "_hp24_warmstart_stage_patch", False):
        raise RuntimeError("HP24 warm-start stage patch already installed in this worker")
    original_train_epoch = RunnerBase.train_epoch

    def staged_train_epoch(self, epoch):
        model = self.unwrap_dist_model(self.model)
        old_names = list(getattr(model, "_hp24_warmstart_old_names", ()))
        if not old_names:
            raise RuntimeError("Missing HP24 source parameter manifest on model")
        named = dict(model.named_parameters())
        missing = [name for name in old_names if name not in named]
        if missing:
            raise RuntimeError(f"HP24 source parameters disappeared: {missing[:10]}")

        # Materialize the optimizer while MF still requires gradients. The
        # inherited Joint8 optimizer splits MF into a joint8_mf group; the
        # legacy optimizer audit explicitly requires that group to exist.
        optimizer = self.optimizer
        mf_groups = [group for group in optimizer.param_groups
                     if group.get("joint8_mf")]
        if not mf_groups:
            raise RuntimeError(
                "Joint8 MF optimizer group missing before router-only warmup"
            )

        if not getattr(self, "_hp24_warmstart_router_only_started", False):
            for name in old_names:
                if name.startswith("rec_encoder."):
                    named[name].requires_grad_(False)
            self._hp24_warmstart_router_only_started = True
            print(
                f"[HP24 WARMSTART STAGE1] epoch={epoch} "
                f"new_router_only=True frozen_source_tensors="
                f"{sum(not named[n].requires_grad for n in old_names)} "
                "MF optimizer group retained but MF parameters frozen",
                flush=True,
            )

        if (
            epoch >= warmup_epochs
            and not getattr(self, "_hp24_warmstart_joint_enabled", False)
        ):
            weight_decay = float(self.config.run_cfg.weight_decay)
            run_cfg = self.config.run_cfg
            scales = {
                "rec": float(run_cfg.get("lr_scale_rec", 1.0)),
                "proj": float(run_cfg.get("lr_scale_proj", 1.0)),
                "gnn": float(run_cfg.get("lr_scale_gnn", 1.0)),
                "lora": float(run_cfg.get("lr_scale_lora", 1.0)),
                "other": float(run_cfg.get("lr_scale_other", 1.0)),
                "cora": float(run_cfg.get("lr_scale_cora", 0.25)),
                "cora_p": float(run_cfg.get("lr_scale_cora_p", 1.0)),
            }
            for group in mf_groups:
                group["lr_scale"] = 1.0
            found_ids = {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            groups = {}
            for name in old_names:
                parameter = named[name]
                parameter.requires_grad_(True)
                if id(parameter) in found_ids:
                    continue
                part = _parameter_part(name)
                no_decay = (
                    parameter.ndim < 2 or "bias" in name
                    or "ln" in name or "bn" in name
                )
                key = (part, no_decay)
                if key not in groups:
                    groups[key] = {
                        "params": [],
                        "weight_decay": 0.0 if no_decay else weight_decay,
                        "lr_scale": scales.get(part, 1.0),
                        "group_name": "warmstart_" + part
                                      + ("_no_wd" if no_decay else "_wd"),
                    }
                groups[key]["params"].append(parameter)

            for group in groups.values():
                group["lr"] = float(optimizer.defaults.get("lr", CORA_LR))
                optimizer.add_param_group(group)

            found_ids = {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            absent = [
                name for name in old_names
                if id(named[name]) not in found_ids
            ]
            if absent:
                raise RuntimeError(
                    "HP24 stage-2 optimizer is missing source parameters: "
                    + repr(absent[:10])
                )
            self._hp24_warmstart_joint_enabled = True
            print(
                f"[HP24 WARMSTART STAGE2] epoch={epoch} "
                f"unfrozen_source_tensors={len(old_names)} "
                f"new_optimizer_groups={len(groups)} "
                f"MF_lr_ratio={MF_LR_RATIO}; joint fine-tuning enabled",
                flush=True,
            )
        return original_train_epoch(self, epoch)

    RunnerBase.train_epoch = staged_train_epoch
    RunnerBase._hp24_warmstart_stage_patch = True


def install_model(trial, hp24_checkpoint, warmup_epochs=ROUTER_WARMUP_EPOCHS,
                  training=True):
    """Install the proven HP24 runtime, replace only the router, then load HP24."""
    hp.install_model(trial)
    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx

    original_init = MiniGPT4Rec_vx.__init__

    def init_with_hp24_warmstart(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if getattr(self, "_hp24_warmstart_loaded", False):
            raise RuntimeError("HP24 warm-start loader invoked twice")
        generator = self.llama_model_lora.base_model.weight_generator
        if not hasattr(generator, "rank4_experts"):
            raise RuntimeError("HP24 hierarchical router was not installed")
        old_router = generator.rank4_experts
        warm_router = _new_router_from_hp24(old_router, trial["warm_variant"])
        generator.rank4_experts = warm_router

        payload = _torch_load(hp24_checkpoint)
        trained = payload["model"]
        current = self.state_dict()
        current_trainable = {
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        new_ids = {id(parameter) for parameter in warm_router.warm_parameters()}
        new_names = {
            name for name, parameter in self.named_parameters()
            if id(parameter) in new_ids
        }
        if len(new_names) != len(new_ids):
            raise RuntimeError("Could not uniquely identify new HP24 routing parameters")
        expected_source = current_trainable - new_names
        source_names = set(trained)
        missing = sorted(expected_source - source_names)
        extra = sorted(source_names - expected_source)
        shape_mismatch = sorted(
            name for name, value in trained.items()
            if name not in current or tuple(value.shape) != tuple(current[name].shape)
        )
        if missing or extra or shape_mismatch:
            raise RuntimeError(
                "HP24 checkpoint/model compatibility audit failed: "
                f"missing_source={missing[:12]} extra_source={extra[:12]} "
                f"shape_mismatch={shape_mismatch[:12]}"
            )
        incompat = self.load_state_dict(trained, strict=False)
        if incompat.unexpected_keys:
            raise RuntimeError(f"Unexpected HP24 checkpoint keys: {incompat.unexpected_keys}")
        for name, value in trained.items():
            if not torch.equal(self.state_dict()[name].detach().cpu(), value.detach().cpu()):
                raise RuntimeError(f"HP24 tensor did not load exactly: {name}")

        self._hp24_warmstart_loaded = True
        self._hp24_source_checkpoint = str(hp24_checkpoint)
        self._hp24_source_sha256 = _sha256(hp24_checkpoint)
        self._hp24_source_epoch = int(payload.get("epoch", -1))
        self._hp24_warmstart_new_names = sorted(new_names)
        self._hp24_warmstart_old_names = sorted(expected_source)
        self._hp24_warmstart_save_names = sorted(expected_source | new_names)

        if training:
            for name, parameter in self.named_parameters():
                # Keep MF requires_grad=True until RunnerBase.optimizer is
                # materialized so Joint8 creates its required joint8_mf group.
                # The stage hook freezes MF immediately after optimizer audit.
                if name in expected_source and not name.startswith("rec_encoder."):
                    parameter.requires_grad_(False)
            if warmup_epochs < 1:
                raise ValueError("Router-only warmup must be at least one epoch")
            self._hp24_warmstart_router_only_epochs = int(warmup_epochs)
            print(
                f"[HP24 WARMSTART LOAD PASS] trial={trial['name']} "
                f"source_epoch={self._hp24_source_epoch} "
                f"loaded_source_tensors={len(trained)} "
                f"new_router_tensors={len(new_names)} "
                f"router_only_epochs={warmup_epochs} "
                f"sha256={self._hp24_source_sha256}",
                flush=True,
            )
        else:
            print(
                f"[HP24 WARMSTART TEST LOAD] trial={trial['name']} "
                f"source_epoch={self._hp24_source_epoch} "
                f"loaded_source_tensors={len(trained)} "
                f"new_router_tensors={len(new_names)}",
                flush=True,
            )

    MiniGPT4Rec_vx.__init__ = init_with_hp24_warmstart
    if training:
        _install_save_patch()
        _install_router_warmup(warmup_epochs)


def _validate_trial_config(path, protocol):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    run = cfg["run"]
    checks = {
        "seed42": int(run["seed"]) == SEED,
        "MF_trainable": cfg["model"].get("freeze_rec") is False,
        "LoRA_trainable": cfg["model"].get("freeze_lora") is False,
        "rank16": int(cfg["model"]["lora_config"]["r"]) == RANK,
        "batch2": int(run["batch_size_train"]) == BATCH_SIZE,
        "accum4": int(run["accum_grad_iters"]) == ACCUMULATION,
        "steps400": int(run["iters_per_epoch"]) == STEPS_PER_EPOCH,
        "valid_only": run.get("valid_splits") == ["valid"] and not run.get("test_splits"),
        "no_generic_resume": not run.get("resume_ckpt_path"),
        "no_model_ckpt": not cfg["model"].get("ckpt"),
        "D4_C6_inputs": run.get("mode") == "v2",
        "output_protocol": protocol in str(run.get("output_dir", "")),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"Warm-start config mismatch {path}: "
            + repr([name for name, ok in checks.items() if not ok])
        )
    return cfg


def plan(protocol, checkpoint):
    configure(protocol)
    checkpoint, payload = find_hp24_checkpoint(checkpoint)
    BASE.plan(protocol)
    config_dir = CORE.CONFIG_DIR
    for trial in TRIALS:
        path = config_dir / (trial["name"] + ".yaml")
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        run = cfg["run"]
        run["init_lr"] = CORA_LR
        run["min_lr"] = MIN_LR
        run["warmup_lr"] = WARMUP_LR
        run["warmup_steps"] = WARMUP_STEPS
        run["seed"] = SEED
        run["valid_splits"] = ["valid"]
        run["test_splits"] = []
        run["evaluate"] = False
        run["resume_ckpt_path"] = None
        run["output_dir"] = str(HERE / "output" / protocol / trial["name"])
        cfg["model"]["ckpt"] = None
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        _validate_trial_config(path, protocol)

    log_dir = HERE / "logs" / protocol
    log_dir.mkdir(parents=True, exist_ok=True)
    out_dir = HERE / "out" / protocol
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": protocol,
        "source_protocol": SOURCE_PROTOCOL,
        "source_trial": SOURCE_TRIAL,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "source_checkpoint_epoch": int(payload.get("epoch", -1)),
        "seed": SEED,
        "gpu": "0",
        "single_model": True,
        "teacher_student": False,
        "warm_start": "load all trainable HP24_B07_T05 tensors; frozen Vicuna reused from configured base model",
        "new_modules": {
            "W1_PAIR_AWARE": "0.1*tanh(MLP([u,i,u*i])) additive personal-expert logit",
            "W2_ADAPTIVE_BETA": "beta_u=0.5+0.4*sigmoid(MLP(u)); zero-init gives beta=0.70",
            "W3_LAYER_PERSONAL": "four zero-initialized personal-logit residual offsets; original scalar HP24 key/shape preserved",
        },
        "initial_function": "all three new variants reproduce HP24 routing exactly before training",
        "training": {
            "router_only_warmup_epochs": ROUTER_WARMUP_EPOCHS,
            "warmup_steps": WARMUP_STEPS,
            "cora_lr": CORA_LR,
            "mf_lr": CORA_LR * MF_LR_RATIO,
            "min_lr": MIN_LR,
            "warmup_lr": WARMUP_LR,
            "batch_size": BATCH_SIZE,
            "accum_grad_iters": ACCUMULATION,
            "iters_per_epoch": STEPS_PER_EPOCH,
            "patience": PATIENCE,
            "valid_only": True,
            "test_during_training": False,
        },
        "unchanged": {
            "rank": 16, "rank_slices": "4xRank4",
            "hierarchy": {"K1": K1, "K2_per_group": K2_PER_GROUP,
                          "route_temperature": 0.50, "beta": 0.70,
                          "personal_target": 0.40, "route_lambda": 0.20},
            "D4": "TOPK_RES025 grouped inputs; neighbor_mix; personal_share=.25",
            "C6": "CTR=.005; MF-KD=.010",
        },
        "trials": TRIALS,
        "configs": [str(config_dir / (t["name"] + ".yaml")) for t in TRIALS],
    }
    (log_dir / "hp24_warmstart_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[HP24 WARMSTART PLAN PASS] configs={config_dir}", flush=True)
    print(f"[HP24 WARMSTART LOGS] {log_dir}", flush=True)
    print(f"[HP24 WARMSTART OUTPUT] {HERE / 'output' / protocol}", flush=True)
    print(
        f"[HP24 WARMSTART LR] CoRA={CORA_LR:g} MF={CORA_LR*MF_LR_RATIO:g}; "
        f"router-only epochs={ROUTER_WARMUP_EPOCHS}; Valid only",
        flush=True,
    )


def cpu_test():
    torch.manual_seed(SEED)
    dim, batch = 32, 8
    rec = type("MF", (), {
        "user_embedding": nn.Embedding(320, dim),
        "item_embedding": nn.Embedding(420, dim),
    })()
    with torch.no_grad():
        rec.user_embedding.weight.normal_()
        rec.item_embedding.weight.normal_()
    coarse, fine, parent = hp.hierarchical_centers(rec.user_embedding.weight)
    user_ids = torch.arange(1, batch + 1)
    item_ids = torch.arange(2, batch + 2)
    base_router = hp.HierPersonalMoE(
        "hier_personal", dim, coarse=coarse, fine=fine, parent=parent,
        personal_logit_init=math.log(2.0), route_temp=0.50,
        route_fine=True, beta=0.70, route_variant="shared",
        model_variant="baseline",
    )
    base_masks, _, _, base_probs = base_router(rec, user_ids, item_ids, {})
    for variant in ("pair", "adaptive_beta", "layer_personal_residual"):
        router = _new_router_from_hp24(base_router, variant)
        masks, aux, _, probs = router(rec, user_ids, item_ids, {})
        expected = base_probs
        torch.testing.assert_close(probs, expected, atol=1e-6, rtol=1e-6)
        for group in range(4):
            torch.testing.assert_close(masks[group], base_masks[group], atol=1e-6, rtol=1e-6)
        assert tuple(probs.shape) == (batch, 4, 4)
        assert all(tuple(mask.shape) == (batch, RANK) for mask in masks)
        signal = torch.randn(batch, RANK)
        loss = (signal * masks[0]).sum() + aux
        loss.backward()
        new_params = router.warm_parameters()
        assert new_params and all(p.grad is not None for p in new_params), variant
        assert all(bool(torch.isfinite(p.grad).all()) for p in new_params), variant
        print(
            f"[HP24 WARMSTART CPU] {variant} identity=True "
            f"new_param_tensors={len(new_params)} gradients=True PASS",
            flush=True,
        )
    print(
        "[HP24 WARMSTART CPU PASS] HP24 identity, W1 pair routing, "
        "W2 adaptive beta, W3 layer residuals and gradients",
        flush=True,
    )


def summary(protocol):
    log_dir = HERE / "logs" / protocol
    out_dir = HERE / "out" / protocol
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ORDER:
        log = log_dir / f"{name}.log"
        if not log.is_file():
            rows.append({
                "trial": name, "valid_auc": "", "valid_uauc_at_best_auc": "",
                "best_epoch": "", "evaluations": 0, "checkpoint": "MISSING",
                "status": "NO_LOG",
            })
            continue
        text = log.read_text(encoding="utf-8", errors="replace")
        pairs = [(float(a), float(u)) for a, u in LOG_LINE.findall(text)]
        best_idx = max(range(len(pairs)), key=lambda i: pairs[i][0]) if pairs else None
        best_marks = [(int(e), float(a)) for e, a in BEST_LINE.findall(text)]
        best_epoch = max(best_marks, key=lambda x: x[1])[0] if best_marks else ""
        folder = HERE / "output" / protocol / name
        checkpoints = sorted(folder.rglob("checkpoint_best.pth")) if folder.is_dir() else []
        if "[JOINT8 EARLY STOP]" in text or "Training time" in text:
            status = "COMPLETE"
        elif "Traceback (most recent call last)" in text:
            status = "FAILED"
        else:
            status = "RUNNING/CHECK"
        rows.append({
            "trial": name,
            "valid_auc": f"{pairs[best_idx][0]:.9f}" if best_idx is not None else "",
            "valid_uauc_at_best_auc": f"{pairs[best_idx][1]:.9f}" if best_idx is not None else "",
            "best_epoch": best_epoch,
            "evaluations": len(pairs),
            "checkpoint": "FOUND" if len(checkpoints) == 1 else
                          ("AMBIGUOUS" if len(checkpoints) > 1 else "MISSING"),
            "status": status,
        })
    target = out_dir / "hp24_warmstart_valid_summary.csv"
    with target.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[HP24 WARMSTART SUMMARY] {target}")
    for row in rows:
        print(
            f"{row['trial']:20} AUC={row['valid_auc'] or '—':>11} "
            f"UAUC={row['valid_uauc_at_best_auc'] or '—':>11} "
            f"ep={row['best_epoch'] or '—':>4} "
            f"ckpt={row['checkpoint']:9} {row['status']}"
        )


def worker(name, protocol, hp24_checkpoint, patience, screen_epochs,
           warmup_epochs):
    configure(protocol)
    if name not in TRIAL_BY_NAME:
        raise ValueError("Unknown warm-start trial: " + name)
    BASE._gpu0_preflight("HP24-warmstart-" + name)
    trial = TRIAL_BY_NAME[name]
    cfg_path = CORE.CONFIG_DIR / (name + ".yaml")
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Run --plan first: {cfg_path}")
    cfg = _validate_trial_config(cfg_path, protocol)

    CORE.patch_runner(
        screen_epochs=screen_epochs,
        mf_lr_ratio=MF_LR_RATIO,
        patience=patience,
    )
    CORE.install_compact_atomic_best_checkpoint()
    COMBO.install_combo_model(COMBO.NAMES["C6_KD_BOTH"])
    RUNTIME._patch_runtime_classes(trial)
    install_model(trial, hp24_checkpoint, warmup_epochs, training=True)
    hp.fac.install_audit(trial)

    print(
        f"[HP24 WARMSTART WORKER] {name} source={hp24_checkpoint} "
        f"output={cfg['run']['output_dir']} seed42 GPU0 "
        "D4/C6 unchanged; Test disabled",
        flush=True,
    )
    sys.argv = ["train_collm_mf_din.py", "--cfg-path", str(cfg_path)]
    runpy.run_path(str(PROJECT_ROOT / "train_collm_mf_din.py"), run_name="__main__")


def run_selected(names, protocol, checkpoint, patience, screen_epochs,
                 warmup_epochs):
    configure(protocol)
    checkpoint, _ = find_hp24_checkpoint(checkpoint)
    if not CORE.CONFIG_DIR.is_dir():
        plan(protocol, checkpoint)
    CORE.LOG_DIR.mkdir(parents=True, exist_ok=True)
    CORE.OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name not in TRIAL_BY_NAME:
            raise ValueError("Unknown trial: " + name)
        log = CORE.LOG_DIR / (name + ".log")
        if log.exists():
            raise RuntimeError(
                f"Refusing to overwrite existing log: {log}; use a new protocol."
            )
        folder = CORE.OUT_DIR / name
        if folder.exists() and any(folder.rglob("checkpoint_best.pth")):
            raise RuntimeError(
                f"Existing best checkpoint protects prior run: {folder}; "
                "use a new protocol for a clean rerun."
            )
    for name in names:
        BASE._gpu0_preflight("before-" + name)
        logfile = CORE.LOG_DIR / (name + ".log")
        command = [
            sys.executable, "-u", str(SCRIPT),
            "--_worker", name,
            "--protocol", protocol,
            "--checkpoint", str(checkpoint),
            "--patience", str(patience),
            "--screen-epochs", str(screen_epochs),
            "--router-warmup-epochs", str(warmup_epochs),
        ]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
        print(f"[HP24 WARMSTART START] {name}; serial GPU0; log={logfile}", flush=True)
        with logfile.open("x", encoding="utf-8") as handle:
            proc = subprocess.Popen(
                command, cwd=PROJECT_ROOT, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
            for line in proc.stdout:
                print(line, end="", flush=True)
                handle.write(line)
            code = proc.wait()
        if code != 0:
            raise SystemExit(
                f"[HP24 WARMSTART STOP] {name} exit={code}; inspect {logfile}. "
                "Later trials were not started."
            )
    print("[HP24 WARMSTART COMPLETE] selected trials finished; Valid only.", flush=True)


def _find_trial_checkpoint(protocol, name, explicit=None):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file() or path.name != "checkpoint_best.pth":
            raise FileNotFoundError(f"Invalid warm-start best checkpoint: {path}")
        return path
    folder = HERE / "output" / protocol / name
    candidates = sorted(folder.rglob("checkpoint_best.pth")) if folder.is_dir() else []
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one best checkpoint for {name} under {folder}; "
            f"found {len(candidates)}"
        )
    return candidates[0].resolve()


def _best_valid_record(protocol, name):
    log = HERE / "logs" / protocol / (name + ".log")
    if not log.is_file():
        raise FileNotFoundError(f"Training log missing: {log}")
    matches = [(int(epoch), float(auc)) for epoch, auc in BEST_LINE.findall(
        log.read_text(encoding="utf-8", errors="replace")
    )]
    if not matches:
        raise RuntimeError(f"No [JOINT8 BEST] valid marker found in {log}")
    return max(matches, key=lambda row: row[1])


def _make_test_config(protocol, name, output_protocol):
    source = CORE.CONFIG_DIR / (name + ".yaml")
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    cfg["run"]["evaluate"] = False
    cfg["run"]["hp_test_only"] = True
    cfg["run"]["valid_splits"] = []
    cfg["run"]["test_splits"] = ["test"]
    cfg["run"]["resume_ckpt_path"] = None
    cfg["run"]["output_dir"] = str(HERE / "output" / output_protocol / name)
    target_dir = HERE / "local_configs" / output_protocol
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / (name + "_TEST.yaml")
    rendered = yaml.safe_dump(cfg, sort_keys=False)
    if target.exists() and target.read_text(encoding="utf-8") != rendered:
        raise FileExistsError(f"Refusing to overwrite different test config: {target}")
    if not target.exists():
        target.write_text(rendered, encoding="utf-8")
    return target


def _test_worker(name, protocol, hp24_checkpoint, warm_checkpoint,
                 output_protocol, valid_epoch, valid_auc):
    configure(protocol)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0 for HP24 warm-start test")
    BASE._gpu0_preflight("HP24-warmstart-test-" + name)
    trial = TRIAL_BY_NAME[name]
    test_yaml = _make_test_config(protocol, name, output_protocol)
    report_dir = HERE / "out" / output_protocol
    report_dir.mkdir(parents=True, exist_ok=True)
    report_json = report_dir / (name + "_test_metrics.json")
    report_csv = report_dir / (name + "_test_metrics.csv")
    if report_json.exists() or report_csv.exists():
        raise FileExistsError(
            f"Test report already exists for {name}; use a new --output-protocol"
        )

    CORE.patch_runner(screen_epochs=0, mf_lr_ratio=MF_LR_RATIO, patience=PATIENCE)
    CORE.install_compact_atomic_best_checkpoint()
    COMBO.install_combo_model(COMBO.NAMES["C6_KD_BOTH"])
    RUNTIME._patch_runtime_classes(trial)
    install_model(trial, hp24_checkpoint, warmup_epochs=0, training=False)

    from minigpt4.runners.runner_base import RunnerBase

    def test_only(self):
        if self.config.run_cfg.get("hp_test_only") is not True:
            raise RuntimeError("HP24 warm-start test-only sentinel missing")
        if list(self.test_splits) != ["test"] or list(self.valid_splits):
            raise RuntimeError("Expected test-only split; no validation during test")
        self.set_model_mode(self.config.run_cfg.mode)
        if "test" not in self.dataloaders:
            raise RuntimeError(f"Missing test dataloader: {list(self.dataloaders)}")
        model = self.unwrap_dist_model(self.model)
        payload = _torch_load(warm_checkpoint)
        if payload.get("next8_checkpoint_format") != "trainable_model_only_v1":
            raise RuntimeError("Warm-start checkpoint format mismatch")
        if payload.get("optimizer_included") is not False:
            raise RuntimeError("Expected model-only warm-start checkpoint")
        epoch = int(payload.get("epoch", -1))
        if epoch != int(valid_epoch):
            raise RuntimeError(
                f"Checkpoint epoch {epoch} != selected Valid-best epoch {valid_epoch}"
            )
        trained = payload.get("model")
        if not isinstance(trained, dict):
            raise RuntimeError("Warm-start checkpoint missing model state")
        current = model.state_dict()
        expected = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        missing = sorted(expected - set(trained))
        extra = sorted(set(trained) - set(current))
        shapes = sorted(
            name for name, value in trained.items()
            if name not in current or tuple(value.shape) != tuple(current[name].shape)
        )
        if missing or extra or shapes:
            raise RuntimeError(
                f"Warm-start test checkpoint mismatch: missing={missing[:12]} "
                f"extra={extra[:12]} shapes={shapes[:12]}"
            )
        incompat = model.load_state_dict(trained, strict=False)
        if incompat.unexpected_keys:
            raise RuntimeError(f"Unexpected test checkpoint keys: {incompat.unexpected_keys}")
        print(
            f"[HP24 WARMSTART TEST CKPT PASS] {name} epoch={epoch} "
            f"tensors={len(trained)} sha256={_sha256(warm_checkpoint)}",
            flush=True,
        )
        model.eval()
        with torch.no_grad():
            result = self.eval_epoch(
                split_name="test", cur_epoch=epoch, skip_reload=True
            )
        auc = float(result["agg_metrics"])
        uauc_value = result["uauc"]
        if isinstance(uauc_value, (tuple, list)):
            uauc_value = uauc_value[0]
        uauc = float(uauc_value)
        if not (0.0 <= auc <= 1.0 and 0.0 <= uauc <= 1.0):
            raise RuntimeError(f"Invalid test metrics: AUC={auc} UAUC={uauc}")
        report = {
            "trial": name,
            "test_auc": auc,
            "test_uauc": uauc,
            "valid_auc_selected": float(valid_auc),
            "checkpoint_best_epoch": epoch,
            "hp24_source_checkpoint": str(hp24_checkpoint),
            "warmstart_checkpoint": str(warm_checkpoint),
            "warmstart_checkpoint_sha256": _sha256(warm_checkpoint),
            "source_protocol": protocol,
            "test_protocol": output_protocol,
            "seed": SEED,
            "test_only": True,
        }
        with report_json.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        with report_csv.open("x", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report))
            writer.writeheader()
            writer.writerow(report)
        print(
            f"[HP24 WARMSTART TEST RESULT] {name} "
            f"AUC={auc:.9f} UAUC={uauc:.9f} epoch={epoch}",
            flush=True,
        )
        return report

    RunnerBase.train = test_only
    sys.argv = ["train_collm_mf_din.py", "--cfg-path", str(test_yaml)]
    runpy.run_path(str(PROJECT_ROOT / "train_collm_mf_din.py"), run_name="__main__")


def test_selected(names, protocol, hp24_checkpoint, output_protocol,
                  explicit_warm_checkpoints=None):
    configure(protocol)
    hp24_checkpoint, _ = find_hp24_checkpoint(hp24_checkpoint)
    test_log_dir = HERE / "logs" / output_protocol
    test_log_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name not in TRIAL_BY_NAME:
            raise ValueError("Unknown test trial: " + name)
        warm = _find_trial_checkpoint(
            protocol, name,
            (explicit_warm_checkpoints or {}).get(name)
        )
        valid_epoch, valid_auc = _best_valid_record(protocol, name)
        log = test_log_dir / (name + "_test.log")
        if log.exists():
            raise RuntimeError(f"Refusing to overwrite test log: {log}")
        command = [
            sys.executable, "-u", str(SCRIPT),
            "--_test-worker", name,
            "--protocol", protocol,
            "--checkpoint", str(hp24_checkpoint),
            "--warmstart-checkpoint", str(warm),
            "--output-protocol", output_protocol,
            "--valid-epoch", str(valid_epoch),
            "--valid-auc", str(valid_auc),
        ]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
        print(f"[HP24 WARMSTART TEST START] {name}; one test evaluation; log={log}", flush=True)
        with log.open("x", encoding="utf-8") as handle:
            proc = subprocess.Popen(
                command, cwd=PROJECT_ROOT, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
            for line in proc.stdout:
                print(line, end="", flush=True)
                handle.write(line)
            code = proc.wait()
        if code != 0:
            raise SystemExit(f"[HP24 WARMSTART TEST STOP] {name} exit={code}; see {log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--cpu-test", action="store_true")
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--summary", action="store_true")
    mode.add_argument("--test", action="store_true")
    parser.add_argument("--only", default="")
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--warmstart-checkpoint", default="")
    parser.add_argument("--output-protocol", default=DEFAULT_TEST_PROTOCOL)
    parser.add_argument("--router-warmup-epochs", type=int, default=ROUTER_WARMUP_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--screen-epochs", type=int, default=0)
    parser.add_argument("--valid-epoch", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--valid-auc", type=float, default=-1.0, help=argparse.SUPPRESS)
    parser.add_argument("--_worker", default="", help=argparse.SUPPRESS)
    parser.add_argument("--_test-worker", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)

    if args.cpu_test:
        cpu_test()
        return
    if args.check:
        checkpoint, payload = find_hp24_checkpoint(args.checkpoint or None)
        print(
            f"[HP24 WARMSTART CHECK PASS] source_epoch={payload.get('epoch')} "
            f"source={checkpoint}",
            flush=True,
        )
        return
    if args.patience < 1 or args.screen_epochs < 0 or args.router_warmup_epochs < 1:
        parser.error("patience>=1, screen-epochs>=0, router-warmup-epochs>=1 required")

    if args._worker:
        checkpoint, _ = find_hp24_checkpoint(args.checkpoint or None)
        worker(
            args._worker, args.protocol, checkpoint,
            args.patience, args.screen_epochs, args.router_warmup_epochs,
        )
        return
    if args._test_worker:
        checkpoint, _ = find_hp24_checkpoint(args.checkpoint or None)
        if not args.warmstart_checkpoint:
            parser.error("--warmstart-checkpoint required for test worker")
        _test_worker(
            args._test_worker, args.protocol, checkpoint,
            Path(args.warmstart_checkpoint).expanduser().resolve(),
            args.output_protocol, args.valid_epoch, args.valid_auc,
        )
        return
    if args.summary:
        configure(args.protocol)
        summary(args.protocol)
        return

    chosen = [x.strip() for x in args.only.split(",") if x.strip()] if args.only else list(ORDER)
    if len(chosen) != len(set(chosen)) or any(name not in TRIAL_BY_NAME for name in chosen):
        parser.error("--only contains an unknown or duplicated trial name")

    checkpoint, _ = find_hp24_checkpoint(args.checkpoint or None)
    if args.plan:
        plan(args.protocol, checkpoint)
        return
    if args.run:
        run_selected(
            chosen, args.protocol, checkpoint,
            args.patience, args.screen_epochs, args.router_warmup_epochs,
        )
        return
    if args.test:
        if not args.only:
            parser.error("--test requires explicit --only; select only the Valid-chosen trial(s)")
        test_selected(
            chosen, args.protocol, checkpoint, args.output_protocol
        )
        return
    parser.error("Specify one supported mode")


if __name__ == "__main__":
    main()
