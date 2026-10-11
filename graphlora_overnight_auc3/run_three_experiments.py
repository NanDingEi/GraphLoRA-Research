#!/usr/bin/env python3
"""Single-GPU, sequential three-run AUC-oriented GraphLoRA sweep.

Usage (from GraphLoRA repository root):
  python graphlora_overnight_auc3/run_three_experiments.py --base-config tran_configs/collm_pretrain_mf_ood_scratch_rlm.yaml --dry-run
  python -u graphlora_overnight_auc3/run_three_experiments.py --base-config tran_configs/collm_pretrain_mf_ood_scratch_rlm.yaml

Requires: PyYAML, the v2 projector regularization patch already installed.
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

RUNS = [
    ("S1_sem_proj_0001", True, 0.0001, "Semantic alignment + light projector preservation"),
    ("S2_sem_proj_0005", True, 0.0005, "Semantic alignment + medium projector preservation"),
    ("S3_sem_proj_0010", True, 0.0010, "Semantic alignment + stronger projector preservation"),
]


def prepare_config(base, name, semantic_enabled, proj_weight, base_dir):
    cfg = json.loads(json.dumps(base))  # detach nested data and reject non-portable values
    model = cfg["model"]
    run = cfg["run"]

    # Single-variable controls; remaining parameters come from the user's YAML.
    model["gnn_semantic_align"] = bool(semantic_enabled)
    model["gnn_semantic_weight"] = 1e-4 if semantic_enabled else 0.0
    model["train_gnn_with_rec_loss"] = True
    model["proj_reg_weight"] = float(proj_weight)
    model["proj_reg_items"] = 64
    model["gcn_layer"] = 31
    model["gnn_scale"] = 0.1
    model["lora_scale"] = 1.0
    # Preserve injection_mode=legacy if already present; do not add unknown config keys.

    run["evaluate"] = False
    run["resume_ckpt_path"] = None
    run["max_epoch"] = int(base["run"].get("max_epoch", 3))  # respect base YAML
    run["output_dir"] = str(base_dir / "runs" / name)
    return cfg


def read_valid_metrics(run_dir):
    """Read saved per-epoch VALID metrics only; no test-based model choice."""
    rows = []
    for path in sorted(run_dir.glob("*/valid_log.txt")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(o, dict):
                continue
            auc = o.get("valid_agg_metrics", o.get("valid_auc"))
            epoch = o.get("valid_epoch", "")
            raw_uauc = o.get("valid_uauc")
            if isinstance(raw_uauc, (list, tuple)) and raw_uauc:
                raw_uauc = raw_uauc[0]
            if not isinstance(auc, (int, float)):
                continue
            rows.append({"run_subdir": path.parent.name, "epoch": epoch,
                         "valid_auc": round(float(auc), 6),
                         "valid_uauc": round(float(raw_uauc), 6) if isinstance(raw_uauc, (int, float)) else ""})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run 3 sequential GraphLoRA experiments")
    parser.add_argument("--base-config", default="tran_configs/collm_pretrain_mf_ood_scratch_rlm.yaml")
    parser.add_argument("--out", default="output/overnight_auc_sweep_3")
    parser.add_argument("--dry-run", action="store_true", help="Generate and inspect configs only; do not train")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop instead of continuing after a failed run")
    args = parser.parse_args()

    repo = Path.cwd().resolve()
    trainer = repo / "train_collm_mf_din.py"
    model_path = repo / "minigpt4" / "models" / "minigpt4rec_v2.py"
    source = (repo / args.base_config).resolve()
    root = (repo / args.out).resolve()

    for p in (trainer, model_path, source):
        if not p.is_file():
            parser.error(f"Missing required file: {p}")
    model_source = model_path.read_text(encoding="utf-8")
    if "projector_regularized_forward" not in model_source:
        parser.error("Projector v2 patch not detected in minigpt4rec_v2.py. Apply it before running.")
    base = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(base, dict) or not isinstance(base.get("model"), dict) or not isinstance(base.get("run"), dict):
        parser.error("Base YAML must contain model: and run: mappings")
    if base["model"].get("ckpt") not in (None, "", "null"):
        parser.error("model.ckpt is not null: these runs should start from the same initial weights. Set it to null in the source YAML.")
    print("[INFO] All three runs use semantic alignment weight=1e-4, original Layer31 injection, same base YAML; only projector regularization varies.", flush=True)

    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "status").mkdir(parents=True, exist_ok=True)

    plans = []
    for name, semantic, weight, why in RUNS:
        cfg = prepare_config(base, name, semantic, weight, root)
        body = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
        config_file = root / "configs" / f"{name}.yaml"
        config_file.write_text(body, encoding="utf-8")
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        plans.append((name, why, config_file, digest))
        print(f"[PLAN] {name}: semantic={semantic}, semantic_weight=1e-4, proj_reg_weight={weight:g}, config={config_file}", flush=True)

    if args.dry_run:
        print("[DRY RUN] Generated three YAML configs. No training launched.", flush=True)
        return 0

    any_failures = False
    for name, reason, cfg_path, digest in plans:
        status_file = root / "status" / f"{name}.json"
        run_directory = root / "runs" / name
        if status_file.exists():
            try:
                old = json.loads(status_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                old = {}
            if old.get("state") == "success":
                if old.get("config_sha256") != digest:
                    print(f"[ERROR] {name}: config has changed since successful run. Refusing to silently skip or overwrite.", flush=True)
                    any_failures = True
                    if args.stop_on_error:
                        break
                    continue
                print(f"[SKIP] {name} completed previously", flush=True)
                continue

        started = datetime.now().isoformat(timespec="seconds")
        cmd = [sys.executable, "-u", str(trainer), "--cfg-path", str(cfg_path)]
        attempt = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = root / "logs" / f"{name}_{attempt}.log"
        print(f"\n[RUN] {name} -- {reason}\n[LOG] {log_file}\n[CMD] {' '.join(cmd)}", flush=True)
        returncode = 99
        try:
            with log_file.open("w", encoding="utf-8", buffering=1) as log:
                log.write("COMMAND: " + " ".join(cmd) + "\n")
                log.write("STARTED: " + started + "\n")
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                process = subprocess.Popen(cmd, cwd=str(repo), env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
                for line in process.stdout:
                    log.write(line)
                    print(line, end="", flush=True)
                returncode = process.wait()
        except KeyboardInterrupt:
            print("[INTERRUPTED] stopping runner", flush=True)
            if 'process' in locals() and process.poll() is None:
                process.terminate()
                process.wait()
            raise
        except Exception as ex:
            print(f"[ERROR] {name}: {ex}", flush=True)
        checkpoint_list = list(run_directory.glob("*/checkpoint_best.pth"))
        good = returncode == 0 and bool(checkpoint_list)
        status = {
            "name": name,
            "state": "success" if good else "failed",
            "returncode": returncode,
            "started": started,
            "finished": datetime.now().isoformat(timespec="seconds"),
            "config": str(cfg_path),
            "config_sha256": digest,
            "log": str(log_file),
            "best_checkpoints": [str(x) for x in checkpoint_list],
        }
        status_file.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{'DONE' if good else 'FAILED'}] {name}, returncode={returncode}, best_checkpoints={len(checkpoint_list)}", flush=True)
        if not good:
            any_failures = True
            if args.stop_on_error:
                break

    summary = []
    for name, *_ in plans:
        metrics = read_valid_metrics(root / "runs" / name)
        for row in metrics:
            summary.append({"experiment": name, **row})
    if summary:
        csv_path = root / "valid_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["experiment", "run_subdir", "epoch", "valid_auc", "valid_uauc"])
            writer.writeheader()
            writer.writerows(summary)
        print(f"[SUMMARY] {csv_path}", flush=True)
        for name, *_ in plans:
            v = [row for row in summary if row["experiment"] == name]
            if v:
                best = max(v, key=lambda row: row["valid_auc"])
                print(f"[BEST VALID] {name}: epoch={best['epoch']}, AUC={best['valid_auc']}, UAUC={best['valid_uauc']}", flush=True)
    print(f"[STATUS] {root / 'status'}", flush=True)
    return 1 if any_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
