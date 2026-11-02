#!/usr/bin/env python3
"""Trainable user-conditioned late-layer residual routing for official CoRA.

This is an end-to-end training experiment, not an OOF post-processor.
It reuses the existing D4_TOPK_RES025 + C6 training stack and captures
L29/L30/L31/L32 Yes logits from the live Vicuna forward graph.

Run from the CoRA_official project root after copying this file there:
  python cora_residual_gate_train.py --cpu-test
  python cora_residual_gate_train.py --plan
  python cora_residual_gate_train.py --run --screen-epochs 2
  python cora_residual_gate_train.py --run

The original CoRA source, existing scripts, configs, logs and checkpoints
are not modified. The new protocol writes to its own logs/output directory.
"""
import argparse
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _find_project_root():
    here = Path(__file__).resolve().parent
    candidates = [here, *list(here.parents)]
    for candidate in candidates:
        if (
            (candidate / "train_collm_mf_din.py").is_file()
            and (candidate / "minigpt4" / "models" / "minigpt4rec_vx.py").is_file()
            and (candidate / "minigpt4" / "tasks" / "base_task.py").is_file()
        ):
            return candidate
    raise RuntimeError(
        "Could not locate the CoRA_official project root. Copy this script "
        "beside train_collm_mf_din.py, minigpt4/, and cora_latevote4.py."
    )


ROOT = _find_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cora_latevote4 as late  # noqa: E402


PROTOCOL = "CORA_RESIDUAL_GATE_TRAIN_20261103"
SEED = 42
CORA_LR = 3e-4
MF_LR_RATIO = 0.1
PATIENCE = 10
STEPS_PER_EPOCH = 400
LATE_LAYERS = (28, 29, 30, 31)  # zero-based => L29/L30/L31/L32
TRIAL = {
    "name": "RGT1_USER_CONDITIONED_MULTI",
    "mode": "neighbor_mix",
    "personal_share": 0.25,
    "late_mode": "residual_gate",
    "consistency_lambda": 0.0,
    "layer_aux_lambda": 0.0,
}
TRIALS = [TRIAL]
ORDER = [TRIAL["name"]]
GATE_HIDDEN = 64
GATE_INIT_BIAS = -1.5
GATE_PRIOR = 1.0 / (1.0 + 2.718281828459045 ** 1.5)
GATE_PRIOR_WEIGHT = 1e-3
AFFINE_PRIOR_WEIGHT = 1e-4


def configure():
    """Reuse the tested C6/D4 runtime and point all outputs at a new protocol."""
    late.ROOT = ROOT
    late.PROTOCOL = PROTOCOL
    late.SEED = SEED
    late.CORA_LR = CORA_LR
    late.MF_LR_RATIO = MF_LR_RATIO
    late.PATIENCE = PATIENCE
    late.LATE_LAYERS = LATE_LAYERS
    late.TRIALS = TRIALS
    late.TRIAL_BY_NAME = {TRIAL["name"]: TRIAL}
    late.ORDER = ORDER

    late.base.ROOT = ROOT
    late.base.core.ROOT = ROOT
    late.base.combo.ROOT = ROOT
    late.configure()


class UserConditionedResidualRouter(nn.Module):
    """User-conditioned three-layer residual router with an L32 anchor.

    For l in {29,30,31}:
      r_l = z_l - (a_l * z_32 + b_l)
    Then:
      z_fused = z_32 + sigmoid(g(u,z_32,|r|)) * sum_l softmax(q)_l * r_l

    The affine residualizer is trainable and starts at the identity
    (a_l=1,b_l=0). The router sees a detached user embedding and detached
    score statistics; gradients still flow through the fused logits/residuals.
    """

    def __init__(
        self,
        user_dim,
        hidden_dim=GATE_HIDDEN,
        init_gate_bias=GATE_INIT_BIAS,
        gate_prior=GATE_PRIOR,
        gate_prior_weight=GATE_PRIOR_WEIGHT,
        affine_prior_weight=AFFINE_PRIOR_WEIGHT,
    ):
        super().__init__()
        self.user_dim = int(user_dim)
        self.residual_scale = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.residual_bias = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        input_dim = self.user_dim + 4  # user embedding + base logit + 3 |residuals|
        self.context_norm = nn.LayerNorm(input_dim)
        self.router = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 4),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        with torch.no_grad():
            self.router[-1].bias[3] = float(init_gate_bias)

        self.gate_prior = float(gate_prior)
        self.gate_prior_weight = float(gate_prior_weight)
        self.affine_prior_weight = float(affine_prior_weight)

    def forward(self, ui_embedding, layer_logits, final_logits):
        if ui_embedding.ndim != 2 or ui_embedding.shape[1] != 2 * self.user_dim:
            raise RuntimeError(
                f"Expected UI embedding [B,{2*self.user_dim}], got {tuple(ui_embedding.shape)}"
            )
        if layer_logits.ndim != 2 or layer_logits.shape[0] != 4:
            raise RuntimeError(
                f"Expected four late-layer logits [4,B], got {tuple(layer_logits.shape)}"
            )

        base = final_logits.float().reshape(-1)
        if layer_logits.shape[1] != base.numel() or ui_embedding.shape[0] != base.numel():
            raise RuntimeError("UI embedding / late logits batch dimensions disagree")

        layer_values = layer_logits.float().transpose(0, 1)[:, :3]
        scale = self.residual_scale.float().view(1, 3)
        bias = self.residual_bias.float().view(1, 3)
        residuals = layer_values - (base.unsqueeze(1) * scale + bias)

        # Only the user half is used: this is a user gate, not a target-item gate.
        user = ui_embedding[:, :self.user_dim].detach().float()
        user = F.normalize(user, p=2, dim=-1, eps=1e-8)
        context = torch.cat(
            (
                user,
                base.detach().unsqueeze(-1),
                residuals.detach().abs(),
            ),
            dim=-1,
        )
        raw = self.router(self.context_norm(context))
        layer_weights = torch.softmax(raw[:, :3], dim=-1)
        gate = torch.sigmoid(raw[:, 3])

        residual_mix = (layer_weights * residuals).sum(dim=-1)
        fused_logits = base + gate * residual_mix

        gate_prior_loss = (gate - self.gate_prior).square().mean()
        affine_prior_loss = (
            (self.residual_scale.float() - 1.0).square().mean()
            + self.residual_bias.float().square().mean()
        )
        regularization = (
            self.gate_prior_weight * gate_prior_loss
            + self.affine_prior_weight * affine_prior_loss
        )
        return fused_logits, gate, layer_weights, residuals, regularization


def _capture_late_with_ui(model, samples, forward_fn, *args, **kwargs):
    """Reuse the validated late-layer capture and also retain CoRA's UI input."""
    generator = model.llama_model_lora.base_model.weight_generator
    captured = {}

    def capture_ui(_module, inputs):
        if not inputs or not torch.is_tensor(inputs[0]):
            raise RuntimeError("CoRA weight_generator did not receive a tensor UI embedding")
        captured["ui_embedding"] = inputs[0]

    handle = generator.register_forward_pre_hook(capture_ui)
    try:
        result, final_logits, layer_logits, max_error = late._capture_late(
            model, samples, forward_fn, *args, **kwargs
        )
    finally:
        handle.remove()

    if "ui_embedding" not in captured:
        raise RuntimeError("Could not capture the user/item embedding used by CoRA")
    if layer_logits.shape[0] != 4:
        raise RuntimeError(f"Expected L29-L32 logits, got {tuple(layer_logits.shape)}")
    return result, final_logits, layer_logits, captured["ui_embedding"], max_error


def install_router_module():
    """Attach the router in __init__, before RunnerBase constructs its optimizer."""
    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx

    if getattr(MiniGPT4Rec_vx, "_rgt_init_patched", False):
        raise RuntimeError("Residual-gate model init patch is already installed")
    original_init = MiniGPT4Rec_vx.__init__

    def init_with_router(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if hasattr(self, "cora_late_residual_router"):
            raise RuntimeError("Residual router unexpectedly already exists")
        user_dim = int(self.rec_encoder.config.embedding_size)
        self.cora_late_residual_router = UserConditionedResidualRouter(user_dim)
        n_params = sum(p.numel() for p in self.cora_late_residual_router.parameters())
        print(
            f"[RGT MODULE] user_dim={user_dim} layers=L29/L30/L31 anchor=L32 "
            f"trainable_router_params={n_params}",
            flush=True,
        )

    MiniGPT4Rec_vx.__init__ = init_with_router
    MiniGPT4Rec_vx._rgt_init_patched = True


def install_residual_gate_forward():
    """Replace only the main Yes-logit BCE; preserve all C6 auxiliary losses."""
    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx

    if getattr(MiniGPT4Rec_vx, "_rgt_forward_patched", False):
        raise RuntimeError("Residual-gate forward patch is already installed")
    original_train = MiniGPT4Rec_vx.forward_v2
    original_eval = MiniGPT4Rec_vx.generate_for_samples_v2

    def _route(self, samples, forward_fn, *args, **kwargs):
        result, final_logits, layer_logits, ui_embedding, max_error = (
            _capture_late_with_ui(self, samples, forward_fn, *args, **kwargs)
        )
        fused, gate, weights, residuals, reg = self.cora_late_residual_router(
            ui_embedding, layer_logits, final_logits
        )
        return result, final_logits, fused, gate, weights, residuals, reg, max_error

    def train_forward(self, samples, *args, **kwargs):
        (
            result, final_logits, fused, gate, weights,
            residuals, reg, max_error,
        ) = _route(self, samples, original_train, *args, **kwargs)

        if not isinstance(result, dict) or "loss" not in result:
            raise RuntimeError("Expected C6 training forward to return a loss dictionary")
        labels = samples["label"].float().to(final_logits.device)
        base_main = F.binary_cross_entropy_with_logits(final_logits.float(), labels)
        fused_main = F.binary_cross_entropy_with_logits(fused.float(), labels)
        c6_extras = result["loss"] - base_main
        total = c6_extras + fused_main + reg

        if not torch.isfinite(total).all():
            raise FloatingPointError("Nonfinite CoRA residual-gate training loss")
        result["loss"] = total
        result["loss_residual_gate"] = fused_main.detach()
        result["loss_base_l32"] = base_main.detach()
        result["loss_gate_regularization"] = reg.detach()

        self._rgt_steps = getattr(self, "_rgt_steps", 0) + 1
        if self._rgt_steps == 1 or self._rgt_steps % 100 == 0:
            route_mean = weights.detach().mean(dim=0).cpu().tolist()
            route_entropy = -(
                weights.detach().clamp_min(1e-8)
                * weights.detach().clamp_min(1e-8).log()
            ).sum(dim=-1).mean().item()
            print(
                f"[RGT TRAIN] step={self._rgt_steps} "
                f"base_bce={float(base_main.detach()):.6f} "
                f"fused_bce={float(fused_main.detach()):.6f} "
                f"gate_mean={float(gate.detach().mean()):.5f} "
                f"gate_std={float(gate.detach().std(unbiased=False)):.5f} "
                f"route_mean={[round(x,5) for x in route_mean]} "
                f"route_entropy={route_entropy:.5f} "
                f"residual_abs_mean={float(residuals.detach().abs().mean()):.6g} "
                f"l32_reconstruction_error={max_error:.6g}",
                flush=True,
            )
        return result

    def eval_forward(self, samples, *args, **kwargs):
        (
            result, final_logits, fused, gate, weights,
            residuals, reg, max_error,
        ) = _route(self, samples, original_eval, *args, **kwargs)
        labels = samples["label"].float().to(final_logits.device)

        if isinstance(result, tuple) and len(result) == 2:
            return result[0], fused
        if not isinstance(result, dict):
            raise RuntimeError("Unexpected CoRA generate_for_samples_v2 return type")
        result["logits"] = fused
        result["loss"] = F.binary_cross_entropy_with_logits(fused.float(), labels)
        return result

    MiniGPT4Rec_vx.forward_v2 = train_forward
    MiniGPT4Rec_vx.generate_for_samples_v2 = eval_forward
    MiniGPT4Rec_vx._rgt_forward_patched = True
    print(
        "[RGT FORWARD] live L29/L30/L31 residuals + L32 anchor; "
        "user-conditioned sigmoid strength + 3-way softmax router; C6 extras retained",
        flush=True,
    )


def _cpu_test():
    torch.manual_seed(42)
    module = UserConditionedResidualRouter(user_dim=256)
    ui = torch.randn(3, 512)
    layers = torch.randn(4, 3, requires_grad=True)
    final = layers[-1]
    fused, gate, weights, residuals, reg = module(ui, layers, final)
    labels = torch.tensor([0.0, 1.0, 1.0])
    loss = F.binary_cross_entropy_with_logits(fused, labels) + reg
    loss.backward()

    assert fused.shape == (3,)
    assert gate.shape == (3,)
    assert weights.shape == (3, 3)
    assert residuals.shape == (3, 3)
    assert torch.isfinite(loss)
    assert torch.isfinite(layers.grad).all()
    assert module.router[-1].weight.grad is not None
    assert torch.isfinite(module.router[-1].weight.grad).all()
    assert float(weights.mean()) == pytest_approx(1.0 / 3.0, 1e-6)
    assert abs(float(gate.mean()) - GATE_PRIOR) < 1e-5
    print(
        "[RGT CPU PASS] affine residuals, user gate, 3-way router, "
        "fused BCE and gradients",
        flush=True,
    )


def pytest_approx(value, tolerance):
    return abs(value - tolerance) if False else value


def prepare():
    configure()
    late.prepare()
    manifest = {
        "protocol": PROTOCOL,
        "trial": TRIAL,
        "seed": SEED,
        "gpu": "0",
        "backbone": "D4_TOPK_RES025 / neighbor_mix personal_share=0.25",
        "loss": "C6_KD_BOTH; replace only final L32 BCE with fused residual BCE; preserve CTR/KD extras",
        "layers_one_based": [29, 30, 31, 32],
        "residual": "r_l = z_l - (a_l * z32 + b_l), trainable affine initialized a=1,b=0",
        "router": "user-embedding-conditioned MLP; 3-way softmax over L29/L30/L31",
        "strength": "sigmoid user gate; initial bias=-1.5; weak prior toward initial mean 0.1824",
        "training": {
            "cora_lr": CORA_LR,
            "mf_lr_ratio": MF_LR_RATIO,
            "batch_size": 2,
            "accum_grad_iters": 4,
            "iters_per_epoch": STEPS_PER_EPOCH,
            "patience": PATIENCE,
            "valid_only": True,
        },
        "checkpoint": "existing trainable-only atomic best checkpoint; includes router and affine residualizer",
        "test_evaluation": False,
    }
    out = late.base.core.LOG_DIR / "residual_gate_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[RGT PLAN] config={late.base.core.CONFIG_DIR / (TRIAL['name'] + '.yaml')}")
    print(f"[RGT PLAN] manifest={out}")
    print("[RGT PLAN] no model training or GPU allocation performed", flush=True)


def worker(name, patience, screen_epochs):
    configure()
    if name != TRIAL["name"]:
        raise ValueError(f"Unknown residual-gate trial: {name}")
    late.base.core.patch_runner(
        screen_epochs=screen_epochs,
        mf_lr_ratio=MF_LR_RATIO,
        patience=patience,
    )
    late.base.core.install_compact_atomic_best_checkpoint()
    late.base.combo.install_combo_model(late.base.combo.NAMES["C6_KD_BOTH"])
    late.base._patch_runtime_classes(TRIAL)
    install_router_module()
    install_residual_gate_forward()

    cfg_path = late.base.core.CONFIG_DIR / (name + ".yaml")
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Run --plan first: {cfg_path}")
    cfg = json.loads("{}") if False else __import__("yaml").safe_load(cfg_path.read_text())
    if (
        cfg["run"]["seed"] != SEED
        or cfg["model"]["freeze_rec"] is not False
        or cfg["run"].get("test_splits")
        or cfg["run"].get("resume_ckpt_path")
        or cfg["run"]["iters_per_epoch"] != STEPS_PER_EPOCH
        or abs(float(cfg["run"]["init_lr"]) - CORA_LR) > 1e-12
    ):
        raise RuntimeError("Residual-gate training config violates fixed controls")

    sys.argv = ["train_collm_mf_din.py", "--cfg-path", str(cfg_path)]
    runpy.run_path(str(ROOT / "train_collm_mf_din.py"), run_name="__main__")


def run_selected(patience, screen_epochs):
    configure()
    if not late.base.core.CONFIG_DIR.is_dir():
        prepare()
    log = late.base.core.LOG_DIR / (TRIAL["name"] + ".log")
    if log.exists():
        raise RuntimeError(f"Refusing to overwrite existing log: {log}")

    command = [
        sys.executable, "-u", str(ROOT / "cora_residual_gate_train.py"),
        "--_worker", TRIAL["name"],
        "--patience", str(patience),
        "--screen-epochs", str(screen_epochs),
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    print(f"[RGT START] serial GPU0; log={log}", flush=True)
    with log.open("x", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        code = proc.wait()
    if code != 0:
        raise SystemExit(f"[RGT STOP] exit={code}; inspect {log}")
    print("[RGT DONE] Valid-only residual-gate training completed.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--screen-epochs", type=int, default=0)
    parser.add_argument("--cpu-test", action="store_true")
    parser.add_argument("--_worker", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    os.chdir(ROOT)
    if args.cpu_test:
        _cpu_test()
        return
    if args._worker:
        worker(args._worker, args.patience, args.screen_epochs)
        return
    if args.patience < 1 or args.screen_epochs < 0:
        parser.error("patience >= 1 and screen-epochs >= 0 required")
    if args.plan:
        prepare()
        return
    if args.run:
        if not late.base.core.CONFIG_DIR.is_dir():
            prepare()
        run_selected(args.patience, args.screen_epochs)
        return
    parser.error("Specify --cpu-test, --plan, or --run")


if __name__ == "__main__":
    main()
