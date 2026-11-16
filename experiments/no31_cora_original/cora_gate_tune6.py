#!/usr/bin/env python3
"""Overnight gate-tuning trials on the proven D4+C6 residual-routing stack.

Extends cora_residual_routing6 with three controlled variations that history
says are the most promising unexplored directions:

  G1_UGATE_B05    user_gate with gate bias -0.5 (prior .38 vs RR5's .18)
  G2_UGATE_B00    user_gate with gate bias  0.0 (prior .50)
  G3_FIXED_A025   single_global with alpha FROZEN at 0.25 (co-adapted blend,
                  mirrors the OOF post-hoc winner that end-to-end alpha missed)
  G4_ROUTE10_UGATE E2's expert-level user soft4 lambda=1.0 routing STACKED with
                  RR5's logit-space user gate (bias -1.5): two orthogonal,
                  independently validated personalization mechanisms.

Base for every trial: official CoRA rank16 generator + D4_TOPK_RES025 input +
C6_KD_BOTH losses + trainable MF, seed 42, GPU0 serial, VALID only.

Run from the CoRA_official root:
  python experiments/no31_cora_original/cora_gate_tune6.py --cpu-test
  python experiments/no31_cora_original/cora_gate_tune6.py --plan
  CUDA_VISIBLE_DEVICES=0 python -u experiments/no31_cora_original/cora_gate_tune6.py --run
  python experiments/no31_cora_original/cora_gate_tune6.py --summary
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
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for _d in (str(ROOT), str(HERE)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import cora_residual_routing6 as R6

PROTOCOL = "CORA_GATE_TUNE_20261112"
SEED = 42
PATIENCE = 10
SCRIPT_PATH = Path(__file__).resolve()

TRIALS = [
    dict(name="G1_UGATE_B05", kind="user_gate", gate_bias=-0.5,
         mode="neighbor_mix", personal_share=0.25),
    dict(name="G2_UGATE_B00", kind="user_gate", gate_bias=0.0,
         mode="neighbor_mix", personal_share=0.25),
    dict(name="G3_FIXED_A025", kind="single_global", fixed_alpha=0.25,
         mode="neighbor_mix", personal_share=0.25),
    dict(name="G4_ROUTE10_UGATE", kind="user_gate", gate_bias=-1.5,
         moe_trial="E2_USER_L10",
         mode="neighbor_mix", personal_share=0.25),
    dict(name="G5_UGATE_B20", kind="user_gate", gate_bias=-2.0,
         mode="neighbor_mix", personal_share=0.25),
    dict(name="G6_UGATE_M15", kind="user_gate", gate_bias=-1.5,
         mode="neighbor_mix", personal_share=0.25),
    dict(name="G7_UGATE_KD20", kind="user_gate", gate_bias=-1.5,
         combo_kd=0.020,
         mode="neighbor_mix", personal_share=0.25),
    dict(name="G9_UGATE_KD05", kind="user_gate", gate_bias=-1.5,
         combo_kd=0.005,
         mode="neighbor_mix", personal_share=0.25),
]
TRIAL_BY_NAME = {t["name"]: t for t in TRIALS}
ORDER = [t["name"] for t in TRIALS]
LOG_LINE = re.compile(r"\*\*\*auc:\s*([0-9.]+)\s+\*\*\*uauc:\s*\(?([0-9.]+)")


def bind_globals():
    """Point the proven R6 pipeline at this protocol's trials and dirs."""
    R6.PROTOCOL = PROTOCOL
    R6.TRIALS = TRIALS
    R6.TRIAL_BY_NAME = TRIAL_BY_NAME
    R6.ORDER = ORDER


def apply_trial_overrides(trial):
    """Per-trial gate bias / frozen alpha; must run before install_trial_module."""
    bias = trial.get("gate_bias")
    if bias is not None:
        R6.GATE_INIT_BIAS = float(bias)
        R6.GATE_PRIOR = 1.0 / (1.0 + math.exp(-float(bias)))
        print(f"[G6 OVERRIDE] {trial['name']} gate_bias={bias} "
              f"prior={R6.GATE_PRIOR:.4f}", flush=True)
    alpha = trial.get("fixed_alpha")
    if alpha is not None:
        if trial["kind"] != "single_global":
            raise RuntimeError("fixed_alpha only supports single_global")
        original_cls = R6.ResidualFusion

        class FixedAlphaFusion(original_cls):
            def __init__(self, kind, user_dim):
                super().__init__(kind, user_dim)
                with torch.no_grad():
                    self.alpha.fill_(float(alpha))
                self.alpha.requires_grad_(False)

        R6.ResidualFusion = FixedAlphaFusion
        print(f"[G6 OVERRIDE] {trial['name']} alpha frozen at {alpha}", flush=True)


def install_moe_route():
    """Stack the factorial E2 expert-level user lambda=1.0 soft4 routing."""
    import cora_collab_moe6 as moe6
    import cora_route_factorial6 as fac

    trial = fac.TRIAL_BY_NAME["E2_USER_L10"]
    if trial["expert_mode"] != "user_soft4" or float(trial["route_lambda"]) != 1.0:
        raise RuntimeError("E2 route definition unexpectedly changed")
    moe6.RESIDUAL_LAMBDA = float(trial["route_lambda"])
    fac._install_grad_norm_hooks()
    moe6.install_moe_model(trial)
    print("[G6 MOE] E2 user_soft4 lambda=1.0 expert routing installed "
          "BEFORE the logit-space fusion wrapper", flush=True)


def worker(name, patience, screen_epochs):
    bind_globals()
    R6.configure()
    R6.gpu0_preflight("g6-worker-before-model-load")
    if name not in TRIAL_BY_NAME:
        raise ValueError(f"Unknown trial: {name}")
    trial = TRIAL_BY_NAME[name]
    cfg_path = R6.late.base.core.CONFIG_DIR / (name + ".yaml")
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Run --plan first: {cfg_path}")
    cfg = R6._validate_config(cfg_path)

    R6.late.base.core.patch_runner(
        screen_epochs=screen_epochs, mf_lr_ratio=R6.MF_LR_RATIO,
        patience=patience,
    )
    R6.late.base.core.install_compact_atomic_best_checkpoint()
    combo_cfg = dict(R6.late.base.combo.NAMES["C6_KD_BOTH"])
    if trial.get("combo_kd") is not None:
        combo_cfg["name"] = f"C6_KD{trial['combo_kd']:.3f}"
        combo_cfg["kd"] = float(trial["combo_kd"])
        print(f"[G6 COMBO OVERRIDE] {name} kd={combo_cfg['kd']} "
              f"ctr={combo_cfg['ctr']} gate={combo_cfg['kd_gate']}", flush=True)
    R6.late.base.combo.install_combo_model(combo_cfg)
    R6.late.base._patch_runtime_classes(trial)
    if trial.get("moe_trial"):
        install_moe_route()
    apply_trial_overrides(trial)
    R6.install_trial_module(trial)
    R6.install_optimizer_audit(trial)
    R6.install_trial_forward(trial)

    print(f"[G6 WORKER READY] {name}; output={cfg['run']['output_dir']}; "
          "valid-only; test disabled", flush=True)
    sys.argv = ["train_collm_mf_din.py", "--cfg-path", str(cfg_path)]
    runpy.run_path(str(ROOT / "train_collm_mf_din.py"), run_name="__main__")


def plan():
    bind_globals()
    R6.configure()
    R6.prepare()
    manifest = {
        "protocol": PROTOCOL, "seed": SEED,
        "base": "CoRA rank16 + D4_TOPK_RES025 + C6_KD_BOTH + trainable MF",
        "trials": [
            {"id": "G1_UGATE_B05",
             "structure": "s=z32+sigmoid(g(u;bias=-0.5))*(z31-z32)"},
            {"id": "G2_UGATE_B00",
             "structure": "s=z32+sigmoid(g(u;bias=0.0))*(z31-z32)"},
            {"id": "G3_FIXED_A025",
             "structure": "s=z32+0.25*(z31-z32); alpha frozen (co-adapted)"},
            {"id": "G4_ROUTE10_UGATE",
             "structure": "E2 user soft4 lambda=1.0 expert routing + "
                          "RR5 user gate (bias -1.5) at the logit level"},
        ],
        "references": {
            "RR5_valid_auc": 0.733768, "RR5_valid_uauc": 0.694790,
            "RR1_valid_auc": 0.732387, "RR1_valid_uauc": 0.701999,
            "E2_valid_auc": 0.731639, "E2_valid_uauc": 0.695967,
        },
        "training": {"patience": PATIENCE, "cora_lr": R6.CORA_LR,
                     "mf_lr_ratio": R6.MF_LR_RATIO, "valid_only": True},
    }
    path = R6.late.base.core.LOG_DIR / "gate_tune6_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"[G6 PLAN PASS] manifest={path}", flush=True)


def cpu_test():
    torch.manual_seed(SEED)
    batch, user_dim = 5, 16
    ui = torch.randn(batch, user_dim * 2)
    layer_logits = torch.randn(4, batch, requires_grad=True)
    final = layer_logits[3]
    saved_bias, saved_prior = R6.GATE_INIT_BIAS, R6.GATE_PRIOR
    saved_cls = R6.ResidualFusion
    try:
        for trial in TRIALS:
            name = trial["name"]
            if trial.get("moe_trial"):
                print(f"[G6 CPU] {name}: stacking checked structurally; "
                      "full route install verified in its worker preflight",
                      flush=True)
                continue
            R6.ResidualFusion = saved_cls
            R6.GATE_INIT_BIAS, R6.GATE_PRIOR = saved_bias, saved_prior
            apply_trial_overrides(trial)
            fusion = R6.ResidualFusion(trial["kind"], user_dim)
            if trial["kind"] == "user_gate":
                with torch.no_grad():
                    gate_logit = fusion.user_gate(
                        F.normalize(ui[:, :user_dim], dim=-1))
                    gate_out = float(torch.sigmoid(gate_logit).mean())
                expected = 1.0 / (1.0 + math.exp(-trial["gate_bias"]))
                if abs(gate_out - expected) > 1e-5:
                    raise RuntimeError(
                        f"{name}: init gate mean {gate_out:.6f} != prior "
                        f"{expected:.6f}")
            if trial.get("fixed_alpha") is not None:
                if float(fusion.alpha.detach()) != trial["fixed_alpha"]:
                    raise RuntimeError(f"{name}: alpha init mismatch")
                if fusion.alpha.requires_grad:
                    raise RuntimeError(f"{name}: alpha must be frozen")
            fused, strength, weights, residuals, reg = fusion(
                ui, layer_logits, final)
            labels = torch.randint(0, 2, (batch,)).float()
            loss = F.binary_cross_entropy_with_logits(fused, labels) + reg
            loss.backward()
            trainable = [p for p in fusion.parameters() if p.requires_grad]
            if trial.get("fixed_alpha") is not None:
                # Frozen alpha leaves NO trainable fusion params; the gradient
                # must instead reach the captured late-layer logits.
                assert not trainable
                assert fusion.alpha.grad is None
                assert layer_logits.grad is not None and bool(
                    layer_logits.grad.abs().sum() > 0)
            else:
                assert trainable and all(p.grad is not None for p in trainable)
            print(f"[G6 CPU] {name}: init_prior_ok backward_ok "
                  f"fused[0]={float(fused[0]):.4f}", flush=True)
    finally:
        R6.GATE_INIT_BIAS, R6.GATE_PRIOR = saved_bias, saved_prior
        R6.ResidualFusion = saved_cls
    print("[G6 CPU PASS] gate-bias overrides, frozen alpha, gradients", flush=True)


def summary():
    log_dir = HERE / "logs" / PROTOCOL
    out_dir = HERE / "out" / PROTOCOL
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "gate_tune6_valid_summary.csv"
    rows = []
    for trial in TRIALS:
        name = trial["name"]
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
        rows.append(dict(
            trial=name,
            valid_auc=f"{pairs[best_i][0]:.9f}",
            valid_uauc=f"{pairs[best_i][1]:.9f}",
            best_epoch=best_i, total_epochs=len(pairs),
            status="EARLY_STOP" if stopped else "CHECK",
        ))
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[G6 SUMMARY] {target}")
    for row in rows:
        print(row["trial"], row["valid_auc"], row["valid_uauc"],
              f"ep={row['best_epoch']}/{row['total_epochs']}", row["status"])


def run_launcher(chosen, patience, screen_epochs):
    bind_globals()
    R6.configure()
    if not R6.late.base.core.CONFIG_DIR.is_dir():
        plan()
    for name in chosen:
        log_path = R6.late.base.core.LOG_DIR / f"{name}.log"
        if log_path.exists():
            raise RuntimeError(f"Refusing to overwrite existing log: {log_path}")
    for name in chosen:
        R6.gpu0_preflight(f"g6-launcher-before-{name}")
        log_path = R6.late.base.core.LOG_DIR / f"{name}.log"
        command = [sys.executable, "-u", str(SCRIPT_PATH),
                   "--_worker", name, "--patience", str(patience),
                   "--screen-epochs", str(screen_epochs)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
        print(f"[G6 START] {name}; serial GPU0; log={log_path}", flush=True)
        with log_path.open("x", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace", bufsize=1)
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
            code = process.wait()
        summary()
        if code != 0:
            raise SystemExit(f"[G6 STOP ON FAILURE] {name} exit={code}; "
                             f"inspect {log_path}")
    print("[G6 COMPLETE] selected VALID-only trials finished", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-test", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--only", default="")
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--screen-epochs", type=int, default=0)
    parser.add_argument("--_worker", default="")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.cpu_test:
        cpu_test()
        return
    if args.patience < 1 or args.screen_epochs < 0:
        parser.error("patience>=1 and screen-epochs>=0 required")
    if args._worker:
        worker(args._worker, args.patience, args.screen_epochs)
        return
    if args.plan:
        plan()
        return
    if args.summary:
        summary()
        return
    if args.run:
        chosen = ([x.strip() for x in args.only.split(",") if x.strip()]
                  if args.only else ORDER)
        if len(chosen) != len(set(chosen)) or any(
                x not in TRIAL_BY_NAME for x in chosen):
            parser.error("--only has unknown or duplicated trial names")
        run_launcher(chosen, args.patience, args.screen_epochs)
        return
    parser.error("Specify --cpu-test, --plan, --run or --summary")


if __name__ == "__main__":
    main()
