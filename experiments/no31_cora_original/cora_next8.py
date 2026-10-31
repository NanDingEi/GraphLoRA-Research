#!/usr/bin/env python3
"""Eight independent follow-up CoRA+MF loss experiments, GPU0 only.

Use alongside the tested cora_joint8.py (commit 2117229582b12e306fbe44c70eeae16f87c3ef25)
at /home/bbx/user_lx/CoRA_official. Official CoRA files are not modified.

All eight:
  - identical pretrained MF, fresh CoRA weights, Vicuna-7B, seed=42
  - CoRA LR=3e-4, MF LR=3e-5, BS=2, accum=4
  - Valid AUC / UAUC, 10 consecutive epochs without improved AUC -> stop
  - one GPU (CUDA_VISIBLE_DEVICES=0), sequential foreground workers
  - logs + checkpoints isolated in CORA_NEXT8_20261030

See CORA_NEXT8_README.md for exact contrasts and interpretation.
"""
import argparse
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import yaml
import cora_joint8 as core

ROOT = Path(__file__).resolve().parent
PROTOCOL = "CORA_NEXT8_20261030"
RUNS = [
    ("N1_CTR_001",      False, 0.01, 0.00, 0.0, 0.0),
    ("N2_CTR_010",      False, 0.10, 0.00, 0.0, 0.0),
    ("N3_KD_001",       False, 0.00, 0.01, 0.0, 0.0),
    ("N4_KD_010",       False, 0.00, 0.10, 0.0, 0.0),
    ("N5_CTR5_KD1",     False, 0.05, 0.01, 0.0, 0.0),
    ("N6_CTR1_KD5",     False, 0.01, 0.05, 0.0, 0.0),
    ("N7_KD_RAMP10",    False, 0.05, 0.05, 0.0, 0.0),
    ("N8_MF_DECOUPLE",  False, 0.05, 0.05, 0.0, 0.0),
]
ITERS_PER_EPOCH = 400
EXPECTED_LR = 3e-4
EXPECTED_MF_RATIO = 0.1

# Keep the tested base script's code and paths, but isolate new runs.
core.RUNS = RUNS
core.PROTOCOL = PROTOCOL
core.CONFIG_DIR = ROOT / "local_configs" / PROTOCOL
core.OUT_DIR = ROOT / "output" / PROTOCOL
core.LOG_DIR = ROOT / "logs" / PROTOCOL


class EpochKDRamp:
    """Callable weight used by cora_joint8's tensor loss multiply.

    __bool__ stays True so the teacher hook is installed even in epoch 0.
    __mul__ is called once per *training microstep* by the original joint_forward.
    """

    def __init__(self, maximum=0.05, ramp_epochs=10):
        self.maximum = float(maximum)
        self.ramp_epochs = int(ramp_epochs)
        self.calls = 0

    def __bool__(self):
        return True

    def __mul__(self, loss):
        epoch = self.calls // ITERS_PER_EPOCH
        weight = self.maximum * min(1.0, epoch / self.ramp_epochs)
        if self.calls % ITERS_PER_EPOCH == 0:
            print(f"[NEXT8 KD SCHEDULE] epoch={epoch} kd_weight={weight:.6g}",
                  flush=True)
        self.calls += 1
        return weight * loss


def install_mf_main_gradient_decouple():
    """No LLM-main gradient into MF; CTR/KD still update MF.

    The CoRA weight generator *receives exactly the same value* ui_embedding
    as before; only the autograd edge to MF is detached. The generator and all
    CoRA parameters remain fully trainable.
    """
    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx

    old_joint_forward = MiniGPT4Rec_vx.forward_v2

    def separated_forward(model, samples):
        # This object is created by the official CoRA PLoraModel.
        weight_generator = model.llama_model_lora.base_model.weight_generator
        if not getattr(weight_generator, "_next8_detached_input", False):
            original_generator_forward = weight_generator.forward

            def forward_with_detached_mf(ui_embedding):
                return original_generator_forward(ui_embedding.detach())

            weight_generator.forward = forward_with_detached_mf
            weight_generator._next8_detached_input = True
            print("[NEXT8 MF DECOUPLE] MF->weight_generator autograd detached; "
                  "CTR/KD->MF gradients retained", flush=True)
        return old_joint_forward(model, samples)

    MiniGPT4Rec_vx.forward_v2 = separated_forward


def preflight():
    if Path(core.__file__).resolve() != ROOT / "cora_joint8.py":
        raise RuntimeError("Install tested cora_joint8.py alongside cora_next8.py")
    source = Path(core.__file__).read_text()
    if "AMP_INIT_SCALE = 256.0" not in source:
        raise RuntimeError("Missing tested AMP stability correction in cora_joint8.py")
    if "import torch\n                if rec.user_embedding" in source:
        raise RuntimeError("Outdated cora_joint8.py with conditional torch import; update first")
    if core.CORA_LR != EXPECTED_LR or core.MF_LR_RATIO != EXPECTED_MF_RATIO:
        raise RuntimeError("CoRA / MF learning rates differ from this experiment protocol")
    if core.ROOT != ROOT:
        raise RuntimeError("Base and wrapper scripts must share the same directory")
    core.check_sources()


def prepare():
    preflight()
    core.prepare()
    for row in RUNS:
        name = row[0]
        path = core.CONFIG_DIR / f"{name}.yaml"
        cfg = yaml.safe_load(path.read_text())
        assert cfg["model"]["freeze_rec"] is False
        assert abs(float(cfg["run"]["init_lr"]) - EXPECTED_LR) < 1e-12
        assert int(cfg["run"]["seed"]) == 42
        assert int(cfg["run"]["iters_per_epoch"]) == ITERS_PER_EPOCH
    meta = {
        "protocol": PROTOCOL,
        "purpose": "MF direct supervision strength, CTR/KD conflict, schedule, and gradient isolation",
        "reference_best_valid": {
            "R2_CTR005": {"auc": 0.725905, "uauc": 0.698475},
            "R3_KD005": {"auc": 0.726337, "uauc": 0.695700},
            "R4_CTR005_KD005": {"auc": 0.725750, "uauc": 0.689830},
        },
        "controls": "seed42; same pretrained MF, fresh CoRA; 3e-4/3e-5 LR; valid only",
        "ramp": {"run": "N7_KD_RAMP10", "kd_start": 0.0,
                 "kd_max": 0.05, "full_weight_from_epoch": 10,
                 "microsteps_per_epoch": ITERS_PER_EPOCH},
        "decouple": {"run": "N8_MF_DECOUPLE",
                     "operation": "detach MF user/item features only at CoRA weight-generator input",
                     "still_trains": "MF through CTR/KD; CoRA through LLM main BCE"},
        "variants": [
            {"id": name, "freeze_rec": frozen, "ctr": ctr, "kd_max": kd,
             "semantic": sem, "anchor": anchor}
            for name, frozen, ctr, kd, sem, anchor in RUNS
        ],
    }
    (core.LOG_DIR / "next8_manifest.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[NEXT8 READY] all eight independent configs generated. "
          "No training started.", flush=True)


def install_compact_atomic_best_checkpoint():
    """Patch only the new experiment's checkpoint saver, not official CoRA code.

    Official RunnerBase serializes both a model state_dict and optimizer state.
    This saves only the actually trainable parameter keys and simple metadata,
    and writes via a checked, atomic temporary file. It is suitable for
    best-model evaluation, but NOT full optimizer-state resumption.
    """
    import os
    import shutil
    import zipfile
    import torch
    from minigpt4.runners.runner_base import RunnerBase

    def compact_save(self, cur_epoch, is_best=False):
        model = self.unwrap_dist_model(self.model)
        trainable = {name for name, p in model.named_parameters()
                     if p.requires_grad}
        if not trainable:
            raise RuntimeError("[NEXT8 CKPT] No trainable parameters found")
        state = model.state_dict()
        missing = trainable - set(state)
        if missing:
            raise RuntimeError("[NEXT8 CKPT] trainable keys absent from state_dict: "
                               + repr(sorted(missing)[:10]))
        compact = {name: tensor.detach() for name, tensor in state.items()
                   if name in trainable}
        expected = sum(v.numel() * v.element_size() for v in compact.values())
        if expected > 2 * (1024 ** 3):
            raise RuntimeError("[NEXT8 CKPT] trainable state exceeds 2 GiB; "
                               "inspect frozen/training parameter flags before saving")

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / (
            "checkpoint_best.pth" if is_best else f"checkpoint_{cur_epoch}.pth"
        )
        temporary = target.with_name(target.name + ".writing")
        if temporary.exists():
            raise RuntimeError(
                f"[NEXT8 CKPT] leftover temporary checkpoint: {temporary}. "
                "Inspect it before renaming/removing; refusing to overwrite.")

        free_space = shutil.disk_usage(output_dir).free
        minimum = int(expected * 1.25) + 128 * (1024 ** 2)
        print(f"[NEXT8 CKPT] epoch={cur_epoch} trainable_tensors={len(compact)} "
              f"estimated={expected / (1024 ** 3):.3f}GiB "
              f"disk_free={free_space / (1024 ** 3):.3f}GiB "
              "model-only; no optimizer state", flush=True)
        if free_space < minimum:
            raise OSError(
                "[NEXT8 CKPT] insufficient free disk for safe atomic save: "
                f"free={free_space / (1024 ** 3):.3f}GiB "
                f"required_at_least={minimum / (1024 ** 3):.3f}GiB. "
                "Use df/du to inspect disk and preserve prior results.")

        payload = {
            "model": compact,
            "config": self.config.to_dict(),
            "epoch": int(cur_epoch),
            "next8_checkpoint_format": "trainable_model_only_v1",
            "optimizer_included": False,
        }
        try:
            torch.save(payload, str(temporary))
            with zipfile.ZipFile(temporary, "r") as zf:
                error_member = zf.testzip()
                if error_member:
                    raise IOError(f"Checkpoint CRC error in {error_member}")
            with open(temporary, "rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except BaseException:
            # Never replace an existing good best checkpoint with a partial
            # write. Preserve failed temporary files for investigation.
            print(f"[NEXT8 CKPT FAIL] temporary={temporary} "
                  f"existing_best_unchanged={target}", flush=True)
            raise
        print(f"[NEXT8 CKPT SAVED] path={target} "
              f"size={target.stat().st_size / (1024 ** 3):.3f}GiB", flush=True)

    RunnerBase._save_checkpoint = compact_save
    print("[NEXT8 CKPT PATCH] atomic trainable-only saving enabled; "
          "optimizer-state resume intentionally unavailable", flush=True)


def worker(name, patience, screen_epochs):
    preflight()
    spec = core.get_trial(name)
    core.patch_runner(
        screen_epochs=screen_epochs,
        mf_lr_ratio=EXPECTED_MF_RATIO,
        patience=patience,
    )
    install_compact_atomic_best_checkpoint()
    if name == "N7_KD_RAMP10":
        spec = list(spec)
        spec[3] = EpochKDRamp(maximum=0.05, ramp_epochs=10)
        spec = tuple(spec)

    core.patch_model(spec)

    if name == "N8_MF_DECOUPLE":
        install_mf_main_gradient_decouple()

    cfg = core.CONFIG_DIR / f"{name}.yaml"
    if not cfg.exists():
        raise RuntimeError("Missing config; run preflight without --run first: " + str(cfg))
    config = yaml.safe_load(cfg.read_text())
    if config["run"]["seed"] != 42 or abs(config["run"]["init_lr"] - EXPECTED_LR) > 1e-12:
        raise RuntimeError(f"Unexpected seed or LR in {cfg}")
    sys.argv = ["train_collm_mf_din.py", "--cfg-path", str(cfg)]
    runpy.run_path(str(ROOT / "train_collm_mf_din.py"), run_name="__main__")


def run_selected(names, patience, screen_epochs):
    for name in names:
        core.get_trial(name)
    for name in names:
        log_path = core.LOG_DIR / f"{name}.log"
        # Fail closed: never overwrite any previous trial log, including failures.
        if log_path.exists() and log_path.stat().st_size:
            raise RuntimeError(f"Existing trial log will NOT be overwritten: {log_path}\n"
                               "To retry a failed run, manually rename its log and verify the "
                               "old checkpoint directory is not being reused.")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        args = [sys.executable, "-u", str(ROOT / "cora_next8.py"),
                "--_worker", name, "--patience", str(patience),
                "--screen-epochs", str(screen_epochs)]
        print(f"\n[NEXT8 START] {name}: GPU0, foreground, {log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as fh:
            proc = subprocess.Popen(args, cwd=str(ROOT), env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    encoding="utf-8", errors="replace", bufsize=1)
            for line in proc.stdout:
                print(line, end="", flush=True)
                fh.write(line)
            rc = proc.wait()
        if rc:
            raise SystemExit(f"[NEXT8 STOP ON FAILURE] {name} exit={rc}; check {log_path}")
        core.summarize()
    print("\n[NEXT8 COMPLETE] all requested runs finished; VALID-only, no TEST.",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="Run all requested experiments in foreground on GPU0")
    parser.add_argument("--only", default="",
                        help="Optional comma-separated trial subset, e.g. N5_CTR5_KD1,N7_KD_RAMP10")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--screen-epochs", type=int, default=0,
                        help="Optional early cutoff, default=0 means no cutoff")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--_worker", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args._worker:
        worker(args._worker, args.patience, args.screen_epochs)
        return
    if args.summary:
        # Unified report covering R0-R7 and N1-N8 across both log protocols.
        # Import only here: training workers never touch the report writer.
        try:
            from cora_all_summary import collect, display, write_reports
        except ImportError as exc:
            raise RuntimeError(
                "Missing cora_all_summary.py: download the unified summary "
                "script alongside cora_next8.py before using --summary") from exc
        rows = collect(ROOT)
        display(rows, "auc")
        csv_path, md_path = write_reports(rows, ROOT / "out")
        print(f"CSV: {csv_path}\\nMarkdown: {md_path}")
        return
    if args.patience < 1 or args.screen_epochs < 0:
        parser.error("patience must be >= 1, screen-epochs must be >= 0")
    prepare()
    if not args.run:
        return
    names = [s.strip() for s in args.only.split(",") if s.strip()] if args.only \
        else [r[0] for r in RUNS]
    run_selected(names, args.patience, args.screen_epochs)


if __name__ == "__main__":
    main()
