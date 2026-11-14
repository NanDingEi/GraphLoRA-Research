#!/usr/bin/env python3
"""Six controlled, end-to-end CoRA late-layer residual-routing experiments.

All six trials use the same official CoRA + D4_TOPK_RES025 input path and
C6_KD_BOTH auxiliary losses. Only the late-logit residual fusion differs.
Each trial runs in a fresh subprocess, serially on physical GPU0, with its own
config, log and best checkpoint. VALID only; TEST is disabled.

Usage from the CoRA_official source root:
  python experiments/no31_cora_original/cora_residual_routing6.py --cpu-test
  python experiments/no31_cora_original/cora_residual_routing6.py --plan
  python experiments/no31_cora_original/cora_residual_routing6.py --run
  python experiments/no31_cora_original/cora_residual_routing6.py --run --only RR5_USER_GATE_R31,RR6_USER_ROUTER_MULTI
"""
import argparse
import json
import math
import os
import runpy
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


def _find_project_root():
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (
            (candidate / "train_collm_mf_din.py").is_file()
            and (candidate / "minigpt4" / "models" / "minigpt4rec_vx.py").is_file()
            and (candidate / "minigpt4" / "tasks" / "base_task.py").is_file()
        ):
            return candidate
    raise RuntimeError(
        "Cannot find CoRA_official root. Place this script under "
        "experiments/no31_cora_original/ in the CoRA source checkout."
    )


PROJECT_ROOT = _find_project_root()
EXP_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = Path(__file__).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import cora_latevote4 as late  # noqa: E402


PROTOCOL = "CORA_RESIDUAL_ROUTING6_20261103"
SEED = 42
CORA_LR = 3e-4
MF_LR_RATIO = 0.1
PATIENCE = 10
STEPS_PER_EPOCH = 400
BATCH_SIZE = 2
ACCUM_GRAD_ITERS = 4
LATE_LAYERS = (28, 29, 30, 31)  # zero-based: L29, L30, L31, L32
GATE_HIDDEN = 64
GATE_INIT_BIAS = -1.5
GATE_PRIOR = 1.0 / (1.0 + math.exp(-GATE_INIT_BIAS))
GATE_MEAN_PRIOR_WEIGHT = 1e-3
MIN_FREE_GPU0_MIB = 22000

# The "mode" and "personal_share" fields preserve the already-tested
# D4_TOPK_RES025 / neighbor_mix input path in cora_struct10.
TRIALS = [
    dict(name="RR1_L32_BASE", kind="baseline",
         mode="neighbor_mix", personal_share=0.25),
    dict(name="RR2_GLOBAL_R31", kind="single_global",
         mode="neighbor_mix", personal_share=0.25),
    dict(name="RR3_FIXED_MULTI", kind="fixed_multi",
         mode="neighbor_mix", personal_share=0.25),
    dict(name="RR4_GLOBAL_ROUTER", kind="global_router",
         mode="neighbor_mix", personal_share=0.25),
    dict(name="RR5_USER_GATE_R31", kind="user_gate",
         mode="neighbor_mix", personal_share=0.25),
    dict(name="RR6_USER_ROUTER_MULTI", kind="user_router",
         mode="neighbor_mix", personal_share=0.25),
]
TRIAL_BY_NAME = {t["name"]: t for t in TRIALS}
ORDER = [t["name"] for t in TRIALS]
USER_KINDS = {"user_gate", "user_router"}
ROUTED_KINDS = {"single_global", "fixed_multi", "global_router",
                "user_gate", "user_router"}


def configure():
    """Bind the proven CoRA/Joint8/D4 infrastructure to this new protocol."""
    late.ROOT = EXP_DIR
    late.PROTOCOL = PROTOCOL
    late.SEED = SEED
    late.CORA_LR = CORA_LR
    late.MF_LR_RATIO = MF_LR_RATIO
    late.PATIENCE = PATIENCE
    late.LATE_LAYERS = LATE_LAYERS
    late.TRIALS = TRIALS
    late.TRIAL_BY_NAME = TRIAL_BY_NAME
    late.ORDER = ORDER

    late.base.ROOT = EXP_DIR
    late.base.core.ROOT = PROJECT_ROOT
    late.base.combo.ROOT = EXP_DIR
    late.configure()


class ResidualFusion(nn.Module):
    """Minimal residual-fusion module; no trainable affine residualizer.

    Residuals are r_l = z_l - z_32 for l in {29,30,31}.
      single_global: z32 + alpha * r31
      fixed_multi:   z32 + alpha * mean(r29,r30,r31)
      global_router: z32 + alpha * sum softmax(q)_l * r_l
      user_gate:     z32 + sigmoid(g(u)) * r31
      user_router:   z32 + sigmoid(g(u)) * sum softmax(q(u))_l * r_l

    User conditioning uses only the user half of the raw MF UI embedding.
    It is detached from the router loss, so the gate cannot directly distort
    the MF user table; MF still receives its original C6 and CoRA gradients.
    """

    def __init__(self, kind, user_dim):
        super().__init__()
        self.kind = str(kind)
        self.user_dim = int(user_dim)
        self._r6_grad_seen = set()
        self._r6_grad_nonfinite = set()

        if self.kind in {"single_global", "fixed_multi", "global_router"}:
            self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        if self.kind == "global_router":
            self.global_layer_logits = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        if self.kind in USER_KINDS:
            self.user_gate = nn.Sequential(
                nn.LayerNorm(self.user_dim),
                nn.Linear(self.user_dim, GATE_HIDDEN),
                nn.GELU(),
                nn.Linear(GATE_HIDDEN, 1),
            )
            nn.init.zeros_(self.user_gate[-1].weight)
            nn.init.constant_(self.user_gate[-1].bias, GATE_INIT_BIAS)
        if self.kind == "user_router":
            self.user_layer_router = nn.Sequential(
                nn.LayerNorm(self.user_dim),
                nn.Linear(self.user_dim, GATE_HIDDEN),
                nn.GELU(),
                nn.Linear(GATE_HIDDEN, 3),
            )
            nn.init.zeros_(self.user_layer_router[-1].weight)
            nn.init.zeros_(self.user_layer_router[-1].bias)

        # Observe scaled AMP gradients but never raise from a gradient hook:
        # GradScaler must be allowed to detect/skip overflow itself.
        for name, parameter in self.named_parameters():
            parameter.register_hook(self._make_grad_hook(name))

    def _make_grad_hook(self, name):
        def observe(gradient):
            self._r6_grad_seen.add(name)
            if not bool(torch.isfinite(gradient).all()):
                self._r6_grad_nonfinite.add(name)
            return gradient
        return observe

    def forward(self, ui_embedding, layer_logits, final_logits):
        base = final_logits.float().reshape(-1)
        if layer_logits.ndim != 2 or tuple(layer_logits.shape) != (4, base.numel()):
            raise RuntimeError(
                f"Expected [L29,L30,L31,L32] logits [4,{base.numel()}], "
                f"got {tuple(layer_logits.shape)}"
            )
        residuals = layer_logits[:3].float().transpose(0, 1) - base.unsqueeze(1)
        batch = base.numel()
        zero_reg = base.new_zeros(())

        if self.kind == "single_global":
            weights = base.new_tensor([0.0, 0.0, 1.0]).view(1, 3).expand(batch, -1)
            strength = self.alpha.expand(batch)
            reg = zero_reg
        elif self.kind == "fixed_multi":
            weights = base.new_full((batch, 3), 1.0 / 3.0)
            strength = self.alpha.expand(batch)
            reg = zero_reg
        elif self.kind == "global_router":
            weights = torch.softmax(self.global_layer_logits.float(), dim=-1)
            weights = weights.view(1, 3).expand(batch, -1)
            strength = self.alpha.expand(batch)
            reg = zero_reg
        elif self.kind in USER_KINDS:
            if ui_embedding is None:
                raise RuntimeError(f"{self.kind} requires the live CoRA UI embedding")
            if ui_embedding.ndim != 2 or ui_embedding.shape != (batch, 2 * self.user_dim):
                raise RuntimeError(
                    f"Expected raw MF UI embedding [{batch},{2*self.user_dim}], "
                    f"got {tuple(ui_embedding.shape)}"
                )
            user = F.normalize(
                ui_embedding[:, :self.user_dim].detach().float(),
                p=2, dim=-1, eps=1e-8,
            )
            strength = torch.sigmoid(self.user_gate(user)).reshape(-1)
            if self.kind == "user_gate":
                weights = base.new_tensor([0.0, 0.0, 1.0]).view(1, 3).expand(batch, -1)
            else:
                weights = torch.softmax(self.user_layer_router(user).float(), dim=-1)
            # Important: constrain only the batch mean, NOT each user's gate.
            # This preserves the possibility of user-specific gate variation.
            reg = GATE_MEAN_PRIOR_WEIGHT * (strength.mean() - GATE_PRIOR).square()
        else:
            raise ValueError(f"Unknown residual fusion kind: {self.kind}")

        mixed_residual = (weights * residuals).sum(dim=-1)
        fused = base + strength * mixed_residual
        return fused, strength, weights, residuals, reg


def _capture_live(model, samples, forward_fn, need_user, *args, **kwargs):
    ui_capture = {}
    handle = None
    if need_user:
        generator = model.llama_model_lora.base_model.weight_generator

        def capture_ui(_module, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                raise RuntimeError("CoRA weight_generator did not receive tensor UI input")
            ui_capture["ui"] = inputs[0]

        handle = generator.register_forward_pre_hook(capture_ui)
    try:
        result, final_logits, layer_logits, max_error = late._capture_late(
            model, samples, forward_fn, *args, **kwargs
        )
    finally:
        if handle is not None:
            handle.remove()

    ui_embedding = ui_capture.get("ui")
    if need_user:
        if ui_embedding is None:
            raise RuntimeError("Failed to capture raw MF user/item embedding")
        batch = int(samples["label"].shape[0])
        if ui_embedding.ndim != 2 or ui_embedding.shape[0] != batch:
            raise RuntimeError(
                "Expected the pre-grouping raw UI tensor from the CoRA generator; "
                f"batch={batch}, captured={tuple(ui_embedding.shape)}"
            )
    if tuple(layer_logits.shape) != (4, int(samples["label"].shape[0])):
        raise RuntimeError(f"Unexpected late-logit tensor shape: {tuple(layer_logits.shape)}")
    return result, final_logits, layer_logits, ui_embedding, max_error


def install_trial_module(trial):
    """Attach router parameters after the rankgran model-init patch but pre-optimizer."""
    if trial["kind"] == "baseline":
        print("[R6 MODEL] baseline: no residual module or additional parameters", flush=True)
        return

    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx

    if getattr(MiniGPT4Rec_vx, "_r6_init_patched", False):
        raise RuntimeError("R6 model-init patch already installed in this worker")
    original_init = MiniGPT4Rec_vx.__init__

    def init_with_residual_module(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if hasattr(self, "cora_residual_fusion"):
            raise RuntimeError("Residual fusion module already exists")
        rec = self.rec_encoder
        if not hasattr(rec, "user_embedding"):
            raise RuntimeError("R6 requires the official MF user_embedding table")
        user_dim = int(rec.user_embedding.weight.shape[1])
        self.cora_residual_fusion = ResidualFusion(trial["kind"], user_dim)
        count = sum(p.numel() for p in self.cora_residual_fusion.parameters())
        print(
            f"[R6 MODULE] {trial['name']} kind={trial['kind']} "
            f"user_dim={user_dim} trainable_router_params={count}",
            flush=True,
        )

    MiniGPT4Rec_vx.__init__ = init_with_residual_module
    MiniGPT4Rec_vx._r6_init_patched = True


def install_optimizer_audit(trial):
    """Verify that every new residual parameter is in the real Runner optimizer."""
    from minigpt4.runners.runner_base import RunnerBase

    current = RunnerBase.optimizer
    if not isinstance(current, property) or current.fget is None:
        raise RuntimeError("RunnerBase.optimizer is not the expected property")
    original_getter = current.fget

    def audited_optimizer(self):
        optimizer = original_getter(self)
        if getattr(self, "_r6_optimizer_audited", False):
            return optimizer
        self._r6_optimizer_audited = True
        model = self.unwrap_dist_model(self.model)
        mf_groups = [g for g in optimizer.param_groups if g.get("joint8_mf")]
        if not mf_groups:
            raise RuntimeError("MF LR split is missing; expected the Joint8 0.1x MF group")

        fusion = getattr(model, "cora_residual_fusion", None)
        if fusion is not None:
            expected = {id(p) for p in fusion.parameters() if p.requires_grad}
            present = {
                id(p)
                for group in optimizer.param_groups
                for p in group["params"]
            }
            missing = expected - present
            if missing:
                raise RuntimeError(
                    f"Residual parameters missing from optimizer: {len(missing)}"
                )
            print(
                f"[R6 OPTIMIZER PASS] {trial['name']} residual_params={len(expected)} "
                f"all_registered=True MF_LR_ratio={MF_LR_RATIO}",
                flush=True,
            )
        else:
            print(
                f"[R6 OPTIMIZER PASS] {trial['name']} baseline; "
                f"MF_LR_ratio={MF_LR_RATIO}",
                flush=True,
            )
        return optimizer

    RunnerBase.optimizer = property(audited_optimizer)


def install_trial_forward(trial):
    """Replace only the main BCE; preserve the original C6 CTR/KD additions."""
    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx

    if getattr(MiniGPT4Rec_vx, "_r6_forward_patched", False):
        raise RuntimeError("R6 forward patch already installed in this worker")
    original_train = MiniGPT4Rec_vx.forward_v2
    original_eval = MiniGPT4Rec_vx.generate_for_samples_v2
    kind = trial["kind"]

    def compute(self, samples, fn, *args, **kwargs):
        need_user = kind in USER_KINDS
        result, final_logits, layer_logits, ui_embedding, max_error = _capture_live(
            self, samples, fn, need_user, *args, **kwargs
        )
        fused, strength, weights, residuals, regularization = (
            self.cora_residual_fusion(ui_embedding, layer_logits, final_logits)
        )
        return (
            result, final_logits, fused, strength, weights,
            residuals, regularization, max_error,
        )

    def train_forward(self, samples, *args, **kwargs):
        if kind == "baseline":
            return original_train(self, samples, *args, **kwargs)

        (
            result, final_logits, fused, strength, weights,
            residuals, regularization, max_error,
        ) = compute(self, samples, original_train, *args, **kwargs)
        if not isinstance(result, dict) or "loss" not in result:
            raise RuntimeError("C6 training forward did not return a loss dictionary")
        labels = samples["label"].float().to(final_logits.device)
        base_main = F.binary_cross_entropy_with_logits(final_logits.float(), labels)
        fused_main = F.binary_cross_entropy_with_logits(fused.float(), labels)
        # C6 result = original L32 BCE + CTR + gated KD. Keep CTR/KD unchanged.
        c6_extras = result["loss"] - base_main
        total = c6_extras + fused_main + regularization
        if not bool(torch.isfinite(total).all()):
            raise FloatingPointError(f"Nonfinite residual-routing loss: {trial['name']}")
        result["loss"] = total
        result["logits"] = fused

        fusion = self.cora_residual_fusion
        step = getattr(self, "_r6_train_steps", 0)
        if step == 1 and not getattr(self, "_r6_grad_audited", False):
            expected = {name for name, p in fusion.named_parameters() if p.requires_grad}
            missing = expected - fusion._r6_grad_seen
            if missing:
                raise RuntimeError(
                    f"Residual branch is disconnected from backward: {sorted(missing)}"
                )
            nonfinite = sorted(fusion._r6_grad_nonfinite)
            print(
                f"[R6 GRADIENT PASS] {trial['name']} parameters={len(expected)} "
                f"all_connected=True first_backward_finite={not bool(nonfinite)} "
                f"nonfinite={nonfinite}; AMP scaler remains responsible for overflow handling",
                flush=True,
            )
            self._r6_grad_audited = True
        self._r6_train_steps = step + 1

        if step == 0 or (step + 1) % 100 == 0:
            route_mean = weights.detach().float().mean(dim=0).cpu().tolist()
            route_entropy = -(
                weights.detach().float().clamp_min(1e-8)
                * weights.detach().float().clamp_min(1e-8).log()
            ).sum(dim=-1).mean().item()
            print(
                f"[R6 LOSS] {trial['name']} step={step+1} "
                f"base_bce={float(base_main.detach()):.6f} "
                f"fused_bce={float(fused_main.detach()):.6f} "
                f"strength_mean={float(strength.detach().mean()):.5f} "
                f"strength_std={float(strength.detach().std(unbiased=False)):.5f} "
                f"route_mean={[round(x,5) for x in route_mean]} "
                f"route_entropy={route_entropy:.5f} "
                f"residual_abs_mean={float(residuals.detach().abs().mean()):.6g} "
                f"regularization={float(regularization.detach()):.6g} "
                f"l32_reconstruction_error={max_error:.6g}",
                flush=True,
            )
        return result

    def eval_forward(self, samples, *args, **kwargs):
        if kind == "baseline":
            return original_eval(self, samples, *args, **kwargs)
        (
            result, final_logits, fused, strength, weights,
            residuals, regularization, max_error,
        ) = compute(self, samples, original_eval, *args, **kwargs)
        labels = samples["label"].float().to(final_logits.device)
        if isinstance(result, tuple) and len(result) == 2:
            return result[0], fused
        if not isinstance(result, dict):
            raise RuntimeError("Unexpected CoRA validation output type")
        result["logits"] = fused
        result["loss"] = F.binary_cross_entropy_with_logits(fused.float(), labels)
        return result

    MiniGPT4Rec_vx.forward_v2 = train_forward
    MiniGPT4Rec_vx.generate_for_samples_v2 = eval_forward
    MiniGPT4Rec_vx._r6_forward_patched = True
    print(
        f"[R6 FORWARD] {trial['name']} kind={kind}; "
        "L29/L30/L31 residuals anchored at L32; C6 CTR/KD retained",
        flush=True,
    )


def gpu0_preflight(stage):
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    if query.returncode != 0:
        raise RuntimeError(
            f"[R6 GPU PREFLIGHT] nvidia-smi failed at {stage}: {query.stderr.strip()}"
        )
    free = None
    for line in query.stdout.splitlines():
        fields = [x.strip() for x in line.split(",")]
        if len(fields) >= 2 and fields[0] == "0":
            free = int(fields[1])
            break
    if free is None:
        raise RuntimeError(f"[R6 GPU PREFLIGHT] physical GPU0 not found: {query.stdout}")
    print(
        f"[R6 GPU PREFLIGHT] {stage} GPU0_free={free} MiB "
        f"required>={MIN_FREE_GPU0_MIB} MiB",
        flush=True,
    )
    if free < MIN_FREE_GPU0_MIB:
        detail = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
        raise RuntimeError(
            f"[R6 GPU PREFLIGHT] refusing to load model: only {free} MiB free; "
            f"need {MIN_FREE_GPU0_MIB} MiB. No training started.\n{detail}"
        )


def _validate_config(path):
    cfg = yaml.safe_load(path.read_text())
    run = cfg["run"]
    checks = {
        "seed42": int(run["seed"]) == SEED,
        "MF_trainable": cfg["model"]["freeze_rec"] is False,
        "batch2": int(run["batch_size_train"]) == BATCH_SIZE,
        "accum4": int(run["accum_grad_iters"]) == ACCUM_GRAD_ITERS,
        "iters400": int(run["iters_per_epoch"]) == STEPS_PER_EPOCH,
        "CoRA_LR": abs(float(run["init_lr"]) - CORA_LR) < 1e-12,
        "valid_only": run.get("valid_splits") == ["valid"] and not run.get("test_splits"),
        "fresh_init": not run.get("resume_ckpt_path") and not cfg["model"].get("ckpt"),
    }
    failed = [key for key, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"Config control mismatch in {path}: {failed}")
    return cfg


def prepare():
    configure()
    late.base.core.check_sources()
    late.base._model_api_preflight()
    late.base.core.prepare()

    for trial in TRIALS:
        cfg_path = late.base.core.CONFIG_DIR / (trial["name"] + ".yaml")
        _validate_config(cfg_path)

    manifest = {
        "protocol": PROTOCOL,
        "seed": SEED,
        "gpu": "physical GPU0 only; serial foreground workers",
        "source": "official VanillaCreamer/CoRA model + existing local Joint8/Combo8/RankGran7/Struct10 patches",
        "input": "D4_TOPK_RES025; neighbor_mix; personal_share=0.25; TopK=40/20/10/5 by layer group",
        "loss": "C6_KD_BOTH; CTR=0.005; gated KD=0.01; only the L32 main BCE is replaced for RR2-RR6",
        "late_layers_one_based": [29, 30, 31, 32],
        "residual_definition": "r_l = z_l - z_32; no trainable affine residualizer",
        "training": {
            "cora_lr": CORA_LR,
            "mf_lr": CORA_LR * MF_LR_RATIO,
            "mf_lr_ratio": MF_LR_RATIO,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": ACCUM_GRAD_ITERS,
            "iterations_per_epoch": STEPS_PER_EPOCH,
            "early_stopping_patience": PATIENCE,
            "seed": SEED,
            "valid_only": True,
            "test_evaluation": False,
        },
        "gate_regularization": {
            "applies_to": ["RR5_USER_GATE_R31", "RR6_USER_ROUTER_MULTI"],
            "formula": "1e-3 * (batch_mean(g_u) - sigmoid(-1.5))^2",
            "note": "mean-only constraint; no per-user gate shrinkage",
        },
        "trials": [
            {"id": "RR1_L32_BASE", "structure": "s=z32"},
            {"id": "RR2_GLOBAL_R31", "structure": "s=z32+alpha*(z31-z32)"},
            {"id": "RR3_FIXED_MULTI", "structure": "s=z32+alpha*mean(r29,r30,r31)"},
            {"id": "RR4_GLOBAL_ROUTER", "structure": "s=z32+alpha*sum(softmax(q)_l*r_l)"},
            {"id": "RR5_USER_GATE_R31", "structure": "s=z32+sigmoid(g(u))*r31"},
            {"id": "RR6_USER_ROUTER_MULTI", "structure": "s=z32+sigmoid(g(u))*sum(softmax(q(u))_l*r_l)"},
        ],
        "checkpoint": "trainable-parameter-only atomic best checkpoint; each trial has its own output directory",
    }
    log_dir = late.base.core.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "residual_routing6_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[R6 PLAN] protocol={PROTOCOL}", flush=True)
    print(f"[R6 PLAN] configs={late.base.core.CONFIG_DIR}", flush=True)
    print(f"[R6 PLAN] logs={late.base.core.LOG_DIR}", flush=True)
    print(f"[R6 PLAN] output={late.base.core.OUT_DIR}", flush=True)
    for trial in TRIALS:
        print(f"[R6 TRIAL] {trial['name']} kind={trial['kind']} "
              "input=D4_TOPK_RES025 CTR=.005 KD=.01 seed=42", flush=True)
    print("[R6 PLAN ONLY] configs validated; no model/GPU training started", flush=True)


def worker(name, patience, screen_epochs):
    configure()
    gpu0_preflight("worker-before-model-load")
    if name not in TRIAL_BY_NAME:
        raise ValueError(f"Unknown trial: {name}")
    trial = TRIAL_BY_NAME[name]
    cfg_path = late.base.core.CONFIG_DIR / (name + ".yaml")
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Run --plan first: {cfg_path}")
    cfg = _validate_config(cfg_path)

    late.base.core.patch_runner(
        screen_epochs=screen_epochs,
        mf_lr_ratio=MF_LR_RATIO,
        patience=patience,
    )
    late.base.core.install_compact_atomic_best_checkpoint()
    late.base.combo.install_combo_model(late.base.combo.NAMES["C6_KD_BOTH"])
    late.base._patch_runtime_classes(trial)
    install_trial_module(trial)
    install_optimizer_audit(trial)
    install_trial_forward(trial)

    print(
        f"[R6 WORKER READY] {name}; output={cfg['run']['output_dir']}; "
        "fresh model/MF init; valid-only; test disabled",
        flush=True,
    )
    sys.argv = ["train_collm_mf_din.py", "--cfg-path", str(cfg_path)]
    runpy.run_path(str(PROJECT_ROOT / "train_collm_mf_din.py"), run_name="__main__")


def _refresh_summary():
    summary = EXP_DIR / "cora_all_summary.py"
    if summary.is_file():
        subprocess.run(
            [sys.executable, str(summary), "--sort", "auc"],
            cwd=PROJECT_ROOT, check=False,
        )


def run_selected(names, patience, screen_epochs):
    configure()
    if not late.base.core.CONFIG_DIR.is_dir():
        prepare()
    for name in names:
        if name not in TRIAL_BY_NAME:
            raise ValueError(f"Unknown trial: {name}")
        log_path = late.base.core.LOG_DIR / f"{name}.log"
        if log_path.exists():
            raise RuntimeError(
                f"Refusing to overwrite existing log: {log_path}. "
                "Use --only to select unstarted trials; archive failed logs explicitly."
            )

    for name in names:
        gpu0_preflight(f"launcher-before-{name}")
        log_path = late.base.core.LOG_DIR / f"{name}.log"
        command = [
            sys.executable, "-u", str(SCRIPT_PATH),
            "--_worker", name,
            "--patience", str(patience),
            "--screen-epochs", str(screen_epochs),
        ]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
        print(f"[R6 START] {name}; serial foreground GPU0; log={log_path}", flush=True)
        with log_path.open("x", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
            code = process.wait()
        _refresh_summary()
        if code != 0:
            raise SystemExit(
                f"[R6 STOP ON FAILURE] {name} exit={code}; inspect {log_path}. "
                "Later experiments were not started."
            )
    print("[R6 COMPLETE] selected VALID-only trials completed; no TEST evaluation.", flush=True)


def _cpu_test():
    torch.manual_seed(SEED)
    batch, user_dim = 5, 16
    ui = torch.randn(batch, user_dim * 2)
    layer_logits = torch.randn(4, batch, requires_grad=True)
    final = layer_logits[3]
    labels = torch.randint(0, 2, (batch,)).float()
    for trial in TRIALS:
        kind = trial["kind"]
        if kind == "baseline":
            fused = final
            loss = F.binary_cross_entropy_with_logits(fused, labels)
            loss.backward(retain_graph=True)
            assert torch.allclose(fused, final)
            continue
        module = ResidualFusion(kind, user_dim)
        fused, strength, weights, residuals, reg = module(ui, layer_logits, final)
        loss = F.binary_cross_entropy_with_logits(fused, labels) + reg
        loss.backward(retain_graph=True)
        assert fused.shape == (batch,)
        assert strength.shape == (batch,)
        assert weights.shape == (batch, 3)
        assert residuals.shape == (batch, 3)
        assert torch.isfinite(loss)
        assert torch.isfinite(layer_logits.grad).all()
        assert all(p.grad is not None for p in module.parameters())
        assert all(torch.isfinite(p.grad).all() for p in module.parameters())
        if kind == "fixed_multi":
            torch.testing.assert_close(
                weights, torch.full_like(weights, 1.0 / 3.0), atol=1e-6, rtol=0
            )
        if kind in USER_KINDS:
            assert abs(float(strength.detach().mean()) - GATE_PRIOR) < 1e-5
            assert float(reg.detach()) < 1e-10
        if kind == "global_router":
            torch.testing.assert_close(
                weights, torch.full_like(weights, 1.0 / 3.0), atol=1e-6, rtol=0
            )
    assert [t["kind"] for t in TRIALS] == [
        "baseline", "single_global", "fixed_multi",
        "global_router", "user_gate", "user_router",
    ]
    print(
        "[R6 CPU PASS] six variants; residual formulas; user-only gates; "
        "uniform initialization; finite loss/gradients; mean-only gate prior",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-test", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--only", default="")
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--screen-epochs", type=int, default=0)
    parser.add_argument("--_worker", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    if args.cpu_test:
        _cpu_test()
        return
    if args._worker:
        worker(args._worker, args.patience, args.screen_epochs)
        return
    if args.patience < 1 or args.screen_epochs < 0:
        parser.error("--patience must be >=1 and --screen-epochs must be >=0")
    if args.plan:
        prepare()
        return
    if args.run:
        names = (
            [x.strip() for x in args.only.split(",") if x.strip()]
            if args.only else ORDER[:]
        )
        if len(names) != len(set(names)) or any(x not in TRIAL_BY_NAME for x in names):
            parser.error("--only contains unknown or duplicate trial IDs")
        configure()
        if not late.base.core.CONFIG_DIR.is_dir():
            prepare()
        run_selected(names, args.patience, args.screen_epochs)
        return
    parser.error("Specify --cpu-test, --plan, or --run")


if __name__ == "__main__":
    main()
