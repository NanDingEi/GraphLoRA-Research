#!/usr/bin/env python3
"""Reproduce CoRA from the PINNED UNMODIFIED author's GitHub source.

Do not import the modified GraphLoRA CoRA adapter. Official CoRA is cloned
inside this existing NIGHT26 project's LOGS (not another working project).
The official training Python is launched with CWD+PYTHONPATH at official
source, and a generated YAML based directly on the official published YAML.

Strict Stage-2 requires the *official-compatible Stage-1 projection checkpoint*,
the pretrained MF state_dict, Vicuna and the original OOD2 data files. Missing
inputs are reported rather than fabricating an equivalent checkpoint.

Original reference: VanillaCreamer/CoRA 4d69c83504259ea8529725f95cfc277ffd1b8077
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PIN = "4d69c83504259ea8529725f95cfc277ffd1b8077"
UPSTREAM = "https://github.com/VanillaCreamer/CoRA.git"
SOURCE_DEFAULT = ROOT / "logs" / "CORA_OFFICIAL_UPSTREAM"
CONFIG_REL = Path("train_configs/plora_pretrain_mf_ood_movie.yaml")
DATA_FILES = ("train_ood2.pkl", "valid_small_ood2.pkl", "test_ood2.pkl")
MODEL_DEFAULT = Path("/home/bbx/user_lx/models/vicuna-7b-v0")
DATA_DEFAULT = Path("/home/bbx/user_lx/GraphLoRA/dataset/ml-1m")

def check_git(source, clone=False):
    """Pin the source commit and refuse to overwrite locally edited upstream."""
    if not source.exists():
        if not clone:
            raise FileNotFoundError(
                f"Original CoRA source absent at {source}. Run --prepare first.")
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", UPSTREAM, str(source)],
                       check=True)
        subprocess.run(["git", "checkout", "--detach", PIN], cwd=source, check=True)
    elif not (source / ".git").exists():
        raise RuntimeError(f"Expected a cloned upstream Git repository: {source}")
    rev = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if rev != PIN:
        raise RuntimeError(f"Wrong original CoRA source commit: {rev}; expected {PIN}. "
                           "Use a pristine clone at the pinned commit.")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=source, text=True).strip()
    if dirty:
        raise RuntimeError("Upstream CoRA files have local edits; refused: " + dirty[:400])
    for relative in (CONFIG_REL, Path("train_collm_mf_din.py"),
                     Path("minigpt4/models/minigpt4rec_vx.py"),
                     Path("minigpt4/models/ppeft/plora.py"),
                     Path("minigpt4/models/ppeft/weight_generator.py"),
                     Path("prompts/tallrec_movie.txt")):
        if not (source / relative).is_file():
            raise RuntimeError(f"Missing pinned official source {relative}")
    return rev

def input_preflight(args, require_checkpoint):
    problems = []
    for name, path in (("Vicuna", args.vicuna), ("MovieLens OOD2 dataset", args.data_root)):
        if not path.is_dir():
            problems.append(f"{name} directory not found: {path}")
    for fn in DATA_FILES:
        if not (args.data_root / fn).is_file():
            problems.append(f"Original OOD2 split missing: {args.data_root / fn}")
    for label, path in (("pretrained MF state_dict", args.mf_checkpoint),
                        ("Stage-1 projection checkpoint", args.stage1_checkpoint)):
        if path is None:
            if require_checkpoint:
                problems.append(f"{label} path not supplied")
        elif not path.is_file():
            problems.append(f"{label} file not found: {path}")
    if problems:
        raise RuntimeError("Original CoRA input preflight FAILED:\n  - "
                           + "\n  - ".join(problems))
    return dict(model=str(args.vicuna), dataset=str(args.data_root),
                mf=str(args.mf_checkpoint), stage1=str(args.stage1_checkpoint))

def verify_state_dicts(args):
    """Catch the two most damaging silent mismatches before loading Vicuna."""
    import torch
    import pandas as pd
    datasets = [pd.read_pickle(args.data_root / f) for f in DATA_FILES]
    for file, data in zip(DATA_FILES, datasets):
        need = {"uid", "iid", "label", "title", "his", "his_title"}
        if not need.issubset(data.columns):
            raise RuntimeError(
                f"Official MoiveOOData required columns missing in {file}: "
                f"{sorted(need-set(data.columns))}")
    n_user = max(int(d["uid"].max()) for d in datasets) + 1
    n_item = max(int(d["iid"].max()) for d in datasets) + 1
    mf = torch.load(str(args.mf_checkpoint), map_location="cpu")
    if not isinstance(mf, dict) or not all(
            key in mf for key in ("user_embedding.weight", "item_embedding.weight")):
        raise RuntimeError("MF checkpoint is not original MatrixFactorization "
                           "state_dict (user_embedding.weight / item_embedding.weight)")
    u = tuple(mf["user_embedding.weight"].shape)
    i = tuple(mf["item_embedding.weight"].shape)
    if u != (n_user, 256) or i != (n_item, 256):
        raise RuntimeError(f"Pretrained MF dimension mismatch: found U={u} I={i}, "
                           f"official OOD2 needs U={(n_user,256)} I={(n_item,256)}")
    del mf
    # Stage-1 must have projection layers to be frozen in CoRA stage-2.
    stage = torch.load(str(args.stage1_checkpoint), map_location="cpu")
    if not isinstance(stage, dict) or "model" not in stage:
        raise RuntimeError("Stage-1 checkpoint must be official runner "
                           "format containing the 'model' state dictionary")
    keys = list(stage["model"])
    for key_part in ("llama_proj.", "bias_proj."):
        if not any(key_part in key for key in keys):
            raise RuntimeError(f"Stage-1 checkpoint lacks {key_part} weights. "
                               "Do NOT freeze an untrained projector.")
    del stage
    print(f"[CORA-OFFICIAL-DATA] U={n_user} I={n_item} MF=256 "
          "stage1_proj_present=True datasets=train/valid_small/test_ood2",
          flush=True)

def build_config(source, args, output, evaluate=False):
    import yaml
    base = yaml.safe_load((source / CONFIG_REL).read_text())
    m, d, run = base["model"], base["datasets"]["movie_ood"], base["run"]
    if (m["arch"] != "mini_gpt4rec_vx" or
            m["lora_config"]["r"] != 16 or m["lora_config"]["alpha"] != 32):
        raise RuntimeError("Pinned CoRA config does not match documented MF reference")
    if set(m["lora_config"]["target_modules"]) != {
            "q_proj", "k_proj", "v_proj", "o_proj"}:
        raise RuntimeError("Official CoRA Q/K/V/O projection config changed")
    if m["freeze_rec"] is not True or m["freeze_proj"] is not True:
        raise RuntimeError("Original CoRA stage-2 should freeze MF and projector")
    if m["freeze_lora"] is not False:
        raise RuntimeError("Original CoRA stage-2 PLoRA was unexpectedly frozen")
    m["llama_model"] = str(args.vicuna)
    m["rec_config"]["pretrained_path"] = str(args.mf_checkpoint)
    m["ckpt"] = str(args.eval_checkpoint if evaluate else args.stage1_checkpoint)
    # Original code concatenates data_dir + 'train_ood2.pkl': trailing slash
    # is mandatory. No format conversions or data copying.
    d["path"] = str(args.data_root) + os.sep
    d["build_info"]["storage"] = str(args.data_root) + os.sep
    run["output_dir"] = str(output)
    run["device"] = "cuda"
    run["distributed"] = False
    run["world_size"] = 1
    run["seed"] = 42
    run["resume_ckpt_path"] = None
    if evaluate:
        # Official evaluate_only calls self.test_splits[0]; supply 'valid'
        # as a test-like evaluation target to avoid using TEST labels.
        run["evaluate"] = True
        run["valid_splits"] = []
        run["test_splits"] = ["valid"]
    else:
        run["evaluate"] = False
        run["valid_splits"] = ["valid"]
        run["test_splits"] = []
    deviations = []
    if args.profile == "gpu0_24gb":
        run["batch_size_train"] = 2
        run["batch_size_eval"] = 3
        deviations.append("train batch 8 -> 2, eval batch 32 -> 3 for GPU0 24GB")
    elif args.profile == "smoke":
        run["batch_size_train"] = 2
        run["batch_size_eval"] = 3
        run["max_epoch"] = 1
        run["iters_per_epoch"] = 10
        deviations.append("SMOKE ONLY: batch=2/3, epochs=1, iters=10")
    if args.max_epochs is not None:
        run["max_epoch"] = args.max_epochs
        deviations.append(f"max_epoch override={args.max_epochs}")
    if args.iters_per_epoch is not None:
        run["iters_per_epoch"] = args.iters_per_epoch
        deviations.append(f"iters_per_epoch override={args.iters_per_epoch}")
    # The official training yaml evaluates test splits only when evaluate=True;
    # disabling premature test usage does not alter training math.
    deviations.append("device restricted to GPU0, no distributed")
    output.parent.mkdir(parents=True, exist_ok=True)
    return base, deviations

def show_plan(args):
    print("[CORA-OFFICIAL-PLAN] VanillaCreamer/CoRA, pinned", PIN)
    print(f"[CORA-OFFICIAL-PLAN] Source={args.source_dir}")
    print("[CORA-OFFICIAL-PLAN] Original architecture=MiniGPT4Rec_vx, "
          "official PLoRA Q/K/V/O, frozen MF+proj, rank16 alpha32 dropout.05")
    print("[CORA-OFFICIAL-PLAN] Published MF stage2 training: "
          "seed42 lr=1e-3 min_lr=8e-5 warmup_lr=1e-5 "
          "epochs=1000 iters/epoch=100 bs=8 eval_bs=32")
    print(f"[CORA-OFFICIAL-PLAN] profile={args.profile}; "
          f"data={args.data_root}; vicuna={args.vicuna}")
    print(f"[CORA-OFFICIAL-PLAN] MF={args.mf_checkpoint or 'MISSING'}; "
          f"stage1={args.stage1_checkpoint or 'MISSING'}")
    print("[CORA-OFFICIAL-PLAN] No GraphLoRA modules, no CTR/KD/GNN "
          "auxiliary losses; original prompt and official PLoRA only")

def run_official(args, evaluate=False):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Explicit CUDA_VISIBLE_DEVICES=0 required.")
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("Cannot check GPU0 compute processes (nvidia-smi missing)")
    result = subprocess.run(
        ["nvidia-smi", "-i", "0", "--query-compute-apps=pid",
         "--format=csv,noheader,nounits"], capture_output=True, text=True,
        check=True)
    if any(line.strip().isdigit() for line in result.stdout.splitlines()):
        raise RuntimeError("GPU0 is occupied: refusing parallel training.")
    src = args.source_dir
    check_git(src)
    input_preflight(args, require_checkpoint=True)
    if evaluate and (args.eval_checkpoint is None or
                     not args.eval_checkpoint.is_file()):
        raise RuntimeError("Official evaluation needs --eval-checkpoint")
    verify_state_dicts(args)
    import yaml
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "EVAL" if evaluate else "TRAIN"
    root = ROOT / "logs" / f"CORA_OFFICIAL_{label}_{stamp}"
    if root.exists():
        raise RuntimeError(f"Refuse collision with previous run {root}")
    root.mkdir(parents=True, exist_ok=False)
    output = ROOT / "output" / root.name
    cfg, deviations = build_config(src, args, output, evaluate=evaluate)
    file = root / "config_official.yaml"
    file.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    manifest = dict(upstream=UPSTREAM, commit=PIN,
                    source=str(src), runner="train_collm_mf_din.py",
                    profile=args.profile, deviations=deviations,
                    stage1_ckpt=str(args.stage1_checkpoint),
                    mf_checkpoint=str(args.mf_checkpoint),
                    vicuna=str(args.vicuna), data_root=str(args.data_root),
                    output=str(output), cfg=str(file),
                    original_config=str(CONFIG_REL),
                    original_code_untouched=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("[CORA-OFFICIAL-START] " + json.dumps(manifest), flush=True)
    # Ensure imports always select official MiniGPT4Rec_vx and ppeft.
    # External installed packages are used only for their dependencies.
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONPATH"] = str(src)
    env["PYTHONUNBUFFERED"] = "1"
    command = [args.python_bin, "-u", "train_collm_mf_din.py",
               "--cfg-path", str(file)]
    start = time.monotonic()
    with (root / "train.log").open("x", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=src, env=env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True, errors="replace", bufsize=1)
        try:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = proc.wait()
        except BaseException:
            proc.terminate()
            proc.wait()
            raise
    manifest["elapsed_seconds"] = round(time.monotonic()-start, 2)
    manifest["exit_code"] = code
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[CORA-OFFICIAL-EXIT] code={code} log={root/'train.log'} "
          f"seconds={manifest['elapsed_seconds']}", flush=True)
    if code:
        raise SystemExit(code)

def main():
    p = argparse.ArgumentParser(
        description="Pinned upstream CoRA MF/ML-1M OOD2 faithful reproduce")
    m = p.add_mutually_exclusive_group(required=True)
    for action in ("plan", "prepare", "train", "evaluate"):
        m.add_argument("--" + action, action="store_true")
    p.add_argument("--source-dir", type=Path, default=SOURCE_DEFAULT)
    p.add_argument("--data-root", type=Path, default=DATA_DEFAULT)
    p.add_argument("--vicuna", type=Path, default=MODEL_DEFAULT)
    p.add_argument("--mf-checkpoint", type=Path)
    p.add_argument("--stage1-checkpoint", type=Path)
    p.add_argument("--eval-checkpoint", type=Path)
    p.add_argument("--profile", choices=("official", "gpu0_24gb", "smoke"),
                   default="official")
    p.add_argument("--max-epochs", type=int)
    p.add_argument("--iters-per-epoch", type=int)
    p.add_argument("--python-bin", default=sys.executable)
    args = p.parse_args()
    for name in ("source_dir", "data_root", "vicuna", "mf_checkpoint",
                 "stage1_checkpoint", "eval_checkpoint"):
        val = getattr(args, name)
        if val is not None:
            setattr(args, name, val.expanduser().resolve())
    for val in (args.max_epochs, args.iters_per_epoch):
        if val is not None and val <= 0:
            p.error("Epoch / iteration overrides must be positive integers")
    if args.plan:
        show_plan(args)
        return
    if args.prepare:
        check_git(args.source_dir, clone=True)
        show_plan(args)
        print("[CORA-OFFICIAL-SOURCE-READY] Unmodified pinned upstream: "
              + str(args.source_dir))
        return
    run_official(args, evaluate=args.evaluate)

if __name__ == "__main__":
    main()
