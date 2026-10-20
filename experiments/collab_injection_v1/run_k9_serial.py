#!/usr/bin/env python3
"""Foreground GPU0 SERIAL nine-arm GraphLoRA candidate-conditioned L31 Key study.

H0 H1 HS (history), T0 T1 TS (target), HT0 HT1 HTS (history+target).
Each independently warm-starts the SAME S2 champion, trains one epoch,
runs Valid Small AUC/UAUC, merges adapter into checkpoint and verifies zero
baseline drift. KS swaps ONLY candidate features for NEW Key branch.
No background execution, no test predictions, no checkpoint cross-warm-start.

Default runs all nine. --stage permits one group of three.
--resume-dir permits restarting after an interruption without re-running
COMPLETED arms, after verifying the manifest, commit and checkpoint.
"""
import argparse
import copy
import datetime
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

# Limit device visibility BEFORE importing PyTorch or touching CUDA.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

ARMS = [
    ("H0", "history", "target_k_no_candidate"),
    ("H1", "history", "target_k_candidate"),
    ("HS", "history", "target_k_shuffle"),
    ("T0", "target", "target_k_no_candidate"),
    ("T1", "target", "target_k_candidate"),
    ("TS", "target", "target_k_shuffle"),
    ("HT0", "history_target", "target_k_no_candidate"),
    ("HT1", "history_target", "target_k_candidate"),
    ("HTS", "history_target", "target_k_shuffle"),
]
S2_AUC = 0.7296752596287361
OUTPUT_COLUMNS = ("arm", "scope", "condition", "auc", "uauc",
                  "delta_auc_vs_s2", "merged_checkpoint")


def run(cmd, logfile):
    """Run a single child process to completion and stop on the first error."""
    logfile.parent.mkdir(parents=True, exist_ok=True)
    print("[K9-CMD]", " ".join(map(str, cmd)), flush=True)
    with logfile.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            [str(x) for x in cmd], cwd=ROOT, env=os.environ.copy(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            errors="replace", bufsize=1,
        )
        try:
            for line in proc.stdout:
                fh.write(line)
                fh.flush()
                # AUC, smoke errors, trainable policy and summary remain
                # visible without printing full UAUC-per-user sequences.
                if (line.startswith(("[KEY-ANCHOR]", "[TRAINABLE-POLICY]",
                                     "[DRIFT]", "[K-UPDATE-PASS]", "[K9-",
                                     "[KS-MISMATCH]"))
                    or "Averaged stats:" in line
                    or "Traceback" in line
                    or "RuntimeError:" in line
                    or "Error:" in line):
                    print(line[:800].rstrip(), flush=True)
            status = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            print(f"[K9-INTERRUPT] Child stopped, log preserved: {logfile}", flush=True)
            raise
    if status:
        raise RuntimeError(f"Subprocess exited {status}: {logfile}")


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_ckpt(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return torch.load(str(path), map_location="cpu", mmap=True,
                          weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def sole_checkpoint(folder, name):
    matches = list(folder.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {name} in {folder}, found {matches}")
    return matches[0]


def inspect_trainable(path, arm):
    ckpt = load_ckpt(path)
    state = ckpt["model"]
    expected = sorted(k for k in state if k.startswith("target_key_adapter."))
    unexpected = sorted(set(state) - set(expected))
    if len(expected) != 4 or unexpected:
        raise RuntimeError(
            f"{arm}: checkpoint must contain ONLY 4 Key adapter tensors; "
            f"Key={len(expected)} other={unexpected[:8]}"
        )
    final = [k for k in expected if k.endswith(".network.2.weight")]
    if len(final) != 1:
        raise RuntimeError(f"{arm}: Key output layer missing in {expected}")
    norm = float(state[final[0]].float().norm())
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError(f"{arm}: no Key projector update (norm={norm})")
    if not all(torch.isfinite(x.float()).all() for x in state.values()):
        raise RuntimeError(f"{arm}: non-finite trained adapter")
    print(f"[K9-UPDATE] {arm}: 4 Key tensors, output norm={norm:.8g}", flush=True)
    return state


def parse_valid(log):
    line_re = re.compile(
        r"\*\*\*auc:\s*([0-9.eE+-]+).*?\*\*\*uauc:\(\s*([0-9.eE+-]+)"
    )
    pairs = []
    for ln in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Averaged stats:" not in ln:
            continue
        m = line_re.search(ln)
        if m:
            pairs.append(tuple(float(v) for v in m.groups()))
    if len(pairs) != 1:
        raise RuntimeError(
            f"Expected one Valid Small AUC/UAUC from {log}, got {len(pairs)}"
        )
    if not all(math.isfinite(v) for v in pairs[0]):
        raise RuntimeError(f"Non-finite validation metric in {log}")
    return pairs[0]


def git_head():
    p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                       capture_output=True, text=True, check=True)
    return p.stdout.strip()


def write_summary(logdir, statuses):
    header = "\t".join(OUTPUT_COLUMNS)
    lines = [header]
    for name, _, _ in ARMS:
        row = statuses.get(name)
        if not row or row.get("status") != "complete":
            continue
        values = [str(row.get(k, "")) for k in OUTPUT_COLUMNS]
        lines.append("\t".join(values))
    (logdir / "summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_state(logdir, state):
    path = logdir / "state.json"
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    temp.replace(path)


def setup_run(args, champion):
    if args.resume_dir:
        logs = Path(args.resume_dir).expanduser().resolve()
        state_file = logs / "state.json"
        if not state_file.is_file():
            raise FileNotFoundError(f"Cannot resume without {state_file}")
        state = json.loads(state_file.read_text(encoding="utf-8"))
        if state.get("repo_commit") != git_head():
            raise RuntimeError(
                "Cannot resume across different Git commits. Recreate run "
                "or restore the original experiment commit."
            )
        if state.get("champion") != str(champion):
            raise RuntimeError("Resume S2 checkpoint differs from previous run")
        if state.get("champion_sha256") != file_sha(champion):
            raise RuntimeError("S2 checkpoint bytes changed since the original run")
        if state.get("seed") != args.seed or state.get("lr") != args.lr:
            raise RuntimeError("Resume seed/LR must match the manifest")
        if state.get("smoke_steps") != args.smoke_steps:
            raise RuntimeError("Resume smoke_steps must match the manifest")
        output_root = Path(state["output_root"])
        if not output_root.is_dir():
            raise RuntimeError(f"Missing original outputs: {output_root}")
        return logs, output_root, state

    tag = "S2_K9_SERIAL_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logs = ROOT / "logs/collab_v1" / tag
    output_root = ROOT / "output" / tag
    if logs.exists() or output_root.exists():
        raise RuntimeError("Run name collision: " + tag)
    logs.mkdir(parents=True)
    output_root.mkdir(parents=True)
    state = {
        "tag": tag,
        "repo_commit": git_head(),
        "champion": str(champion),
        "champion_sha256": file_sha(champion),
        "output_root": str(output_root),
        "seed": args.seed,
        "lr": args.lr,
        "smoke_steps": args.smoke_steps,
        "arms": {},
    }
    write_state(logs, state)
    write_summary(logs, state["arms"])
    return logs, output_root, state


def preflight():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Only GPU0 allowed")
    for file in ("minigpt4/models/minigpt4rec_v2.py",
                 "experiments/collab_injection_v1/eval_target_k_anchor.py",
                 "experiments/collab_injection_v1/test_target_k_cpu.py",
                 "experiments/collab_injection_v1/compare_k_checkpoint_drift.py"):
        if not (ROOT / file).is_file():
            raise FileNotFoundError(ROOT / file)
    process = subprocess.run(
        ["pgrep", "-af", "[t]rain_collm_mf_din.py"],
        capture_output=True, text=True, check=False
    )
    if process.returncode == 0 and process.stdout.strip():
        raise RuntimeError("Another training job is active:\n" + process.stdout)
    # Prevent a common pitfall: running against locally modified source
    # would make the Git SHA in the manifest insufficient for reproducibility.
    tracked = subprocess.run(
        ["git", "diff", "--quiet", "--",
         "minigpt4/models/minigpt4rec_v2.py",
         "experiments/collab_injection_v1"],
        cwd=ROOT, check=False
    )
    if tracked.returncode:
        raise RuntimeError("Uncommitted source edits; commit/stash before running")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "history", "target",
                                           "history_target"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--smoke-steps", type=int, default=24)
    parser.add_argument("--resume-dir", default=None,
                        help="Path to prior logs/collab_v1/S2_K9_SERIAL_... folder")
    args = parser.parse_args()
    if args.seed < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Invalid seed or learning rate")
    if args.smoke_steps < 1:
        raise ValueError("smoke-steps must be positive")
    preflight()
    run([sys.executable, "-m", "py_compile",
         "minigpt4/models/minigpt4rec_v2.py",
         "experiments/collab_injection_v1/run_k9_serial.py"],
        ROOT / "logs/collab_v1/K9_SYNTAX.log")
    run([sys.executable, "experiments/collab_injection_v1/test_target_k_cpu.py"],
        ROOT / "logs/collab_v1/K9_CPU.log")
    run([sys.executable, "experiments/collab_injection_v1/prepare_e3_configs.py"],
        ROOT / "logs/collab_v1/K9_PREPARE.log")

    source = yaml.safe_load(
        (ROOT / "tran_configs/collab_v1/e3/S2_TARGET_FUSION.yaml"
         ).read_text(encoding="utf-8")
    )
    if source["model"]["collab_mode"] != "target_fusion":
        raise RuntimeError("Wrong source configuration")
    if int(source["model"]["gcn_layer"]) != 31 or float(source["model"]["gnn_scale"]) != 0.1:
        raise RuntimeError("Require historical S2 L31 gnn_scale=0.1")
    champion = Path(source["model"]["ckpt"]).resolve()
    if not champion.is_file():
        raise FileNotFoundError(champion)

    logs, outputs, state = setup_run(args, champion)
    arms = [v for v in ARMS if args.stage == "all" or v[1] == args.stage]
    print(f"[K9-PLAN] stage={args.stage} arms={[x[0] for x in arms]} "
          f"checkpoint={champion} GPU0 foreground; seed={args.seed} lr={args.lr}",
          flush=True)
    print(f"[K9-ROOT] logs={logs} outputs={outputs}", flush=True)

    for name, scope, mode in arms:
        prior = state["arms"].get(name)
        if prior and prior.get("status") == "complete":
            ckpt = Path(prior.get("merged_checkpoint", ""))
            if not ckpt.is_file():
                raise RuntimeError(f"Marked completed but no checkpoint: {ckpt}")
            expected_cfg = prior.get("config_sha256")
            config_file = logs / f"{name}_FULL.yaml"
            if expected_cfg != file_sha(config_file):
                raise RuntimeError(f"{name}: config changed; unsafe resume")
            print(f"[K9-SKIP] {name} verified complete: {ckpt}", flush=True)
            continue

        cfg = copy.deepcopy(source)
        m, r = cfg["model"], cfg["run"]
        m["collab_mode"] = mode
        m["target_key_scope"] = scope
        m["trainable_policy"] = "target_k_only"
        for old in ("target_fusion_scale", "target_fusion_mode",
                    "target_fusion_relative_cap"):
            m.pop(old, None)
        r.update({
            "seed": args.seed, "max_epoch": 3,
            "diagnostic_stop_after_epoch0": True, "stop_after_n_epochs": 1,
            "evaluate": False, "skip_validation": False,
            "resume_ckpt_path": None, "save_checkpoint_every_n_epochs": 1,
            "accum_grad_iters": 3,
            "init_lr": args.lr, "min_lr": args.lr * 0.4,
            "warmup_lr": min(1e-6, args.lr * 0.05),
            "lr_scale_rec": 0.0, "lr_scale_gnn": 0.0,
            "lr_scale_proj": 0.0, "lr_scale_lora": 0.0,
            "lr_scale_other": 1.0,
        })
        for stale in ("max_train_steps", "iters_per_epoch", "max_iters",
                      "mid_epoch_eval_iters"):
            r.pop(stale, None)
        if Path(m["ckpt"]).resolve() != champion:
            raise RuntimeError("Independent S2 warm-start invariant failed")
        output = outputs / name
        cfg["run"]["output_dir"] = str(output / "FULL_E1")
        full_cfg = logs / f"{name}_FULL.yaml"
        # Resume of an interrupted arm starts FROM SCRATCH; discard
        # partial outputs so checkpoint names cannot collide.
        if prior is not None:
            import shutil
            if output.exists():
                shutil.rmtree(output)
        full_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False),
                            encoding="utf-8")
        smoke = copy.deepcopy(cfg)
        smoke["run"]["max_train_steps"] = args.smoke_steps
        smoke["run"]["skip_validation"] = True
        smoke["run"]["output_dir"] = str(output / "SMOKE")
        smoke_cfg = logs / f"{name}_SMOKE.yaml"
        smoke_cfg.write_text(yaml.safe_dump(smoke, sort_keys=False),
                             encoding="utf-8")

        state["arms"][name] = {"status": "running", "scope": scope,
                               "condition": mode, "config_sha256": file_sha(full_cfg)}
        write_state(logs, state)
        print(f"[K9-START] {name} scope={scope} condition={mode}", flush=True)
        try:
            run([sys.executable, "-u",
                 "experiments/collab_injection_v1/eval_target_k_anchor.py",
                 "--config", full_cfg, "--checkpoint", champion,
                 "--expected-auc", S2_AUC,
                 "--result", logs / f"{name}_ANCHOR.json"],
                logs / f"{name}_ANCHOR.log")

            run([sys.executable, "-u", "train_collm_mf_din.py", "--cfg-path",
                 smoke_cfg], logs / f"{name}_SMOKE.log")
            smoke_path = sole_checkpoint(output / "SMOKE", "checkpoint_0.pth")
            inspect_trainable(smoke_path, f"{name}/SMOKE")

            run([sys.executable, "-u", "train_collm_mf_din.py", "--cfg-path",
                 full_cfg], logs / f"{name}_FULL.log")
            raw = sole_checkpoint(output / "FULL_E1", "checkpoint_best.pth")
            adapter = inspect_trainable(raw, name)

            champion_obj = load_ckpt(champion)
            trained_obj = load_ckpt(raw)
            merged_state = dict(champion_obj["model"])
            overlap = set(adapter) & set(merged_state)
            if overlap:
                raise RuntimeError(f"Key-only tensors unexpectedly overlap S2: {overlap}")
            merged_state.update(adapter)
            merged = output / "checkpoint_merged_best.pth"
            torch.save({
                "model": merged_state,
                "config": trained_obj["config"],
                "epoch": trained_obj["epoch"],
                "metadata": {
                    "source_champion": str(champion),
                    "key_checkpoint": str(raw),
                    "arm": name, "scope": scope,
                    "condition": mode, "seed": args.seed, "lr": args.lr,
                }
            }, merged)
            del champion_obj, trained_obj, merged_state, adapter

            run([sys.executable,
                 "experiments/collab_injection_v1/compare_k_checkpoint_drift.py",
                 "--champion", champion, "--trained", merged],
                logs / f"{name}_DRIFT.log")
            auc, uauc = parse_valid(logs / f"{name}_FULL.log")
            record = {
                "status": "complete", "arm": name,
                "scope": scope, "condition": mode,
                "auc": auc, "uauc": uauc,
                "delta_auc_vs_s2": auc - S2_AUC,
                "merged_checkpoint": str(merged),
                "config_sha256": file_sha(full_cfg),
            }
            state["arms"][name] = record
            write_state(logs, state)
            write_summary(logs, state["arms"])
            print(f"[K9-RESULT] {name} AUC={auc:.9f} UAUC={uauc:.9f} "
                  f"delta_S2={auc-S2_AUC:+.9f}", flush=True)
        except BaseException as exc:
            state["arms"][name] = {
                "status": "failed", "scope": scope,
                "condition": mode, "error": str(exc),
                "config_sha256": file_sha(full_cfg),
            }
            write_state(logs, state)
            write_summary(logs, state["arms"])
            print(f"[K9-STOP] {name} failed; no later arm was launched: {exc}",
                  flush=True)
            print(f"[K9-RESUME-CMD] python "
                  f"experiments/collab_injection_v1/run_k9_serial.py "
                  f"--resume-dir {logs} --stage {args.stage} "
                  f"--seed {args.seed} --lr {args.lr} "
                  f"--smoke-steps {args.smoke_steps}", flush=True)
            raise
    print("[K9-COMPLETE] All requested serial arms completed", flush=True)
    print((logs / "summary.tsv").read_text(encoding="utf-8"), flush=True)
    print(f"[K9-ROOT] logs={logs} outputs={outputs}", flush=True)


if __name__ == "__main__":
    main()
