#!/usr/bin/env python3
"""8 one-epoch NO31+original-CoRA ablations; GPU0, foreground, Valid only.

Original-CoRA is still the exact default of adapter.py.
Each R1..R8 changes ONE declared axis (R4 is the intentional R1+R3
interaction). All eight start from the SAME pretrained MF/Vicuna
initialization and fresh zero-P CoRA; never resume a competing checkpoint.
Does not alter previous experiments or their optimizer/checkpoint files.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime

import yaml

ROOT = Path(__file__).resolve().parents[2]
READY = ROOT / "logs/NO31_CORA_ORIGINAL_READY/train.yaml"
AUC_PATTERN = re.compile(r"\*\*\*auc:\s*([0-9.]+).*?\*\*\*uauc:\s*\(\s*([0-9.]+)")
BASELINE_AUC = 0.719760366
BASELINE_UAUC = 0.685106415
CORA_R0_AUC = 0.6337559472796233
CORA_R0_UAUC = 0.5931980355032019

# R0 (not rerun): QKVO layers 0..31, concatenated [u;i] as one CF token,
# alpha32, rank16, static P LR multiplier=1, fix_autocast_cache=True.
EXPERIMENTS = [
    dict(id="R1", title="Full-depth Q/V", projections="q_proj,v_proj",
         layers="0-31", cf_input="concat", alpha=32, p_lr=1.0),
    dict(id="R2", title="Full-depth V/O", projections="v_proj,o_proj",
         layers="0-31", cf_input="concat", alpha=32, p_lr=1.0),
    dict(id="R3", title="Upper-half Q/K/V/O", projections="q_proj,k_proj,v_proj,o_proj",
         layers="16-31", cf_input="concat", alpha=32, p_lr=1.0),
    dict(id="R4", title="Upper-half Q/V (R1 x R3)", projections="q_proj,v_proj",
         layers="16-31", cf_input="concat", alpha=32, p_lr=1.0),
    dict(id="R5", title="Two separate CF encoder tokens", projections="q_proj,k_proj,v_proj,o_proj",
         layers="0-31", cf_input="two_tokens", alpha=32, p_lr=1.0),
    dict(id="R6", title="Full-depth alpha16", projections="q_proj,k_proj,v_proj,o_proj",
         layers="0-31", cf_input="concat", alpha=16, p_lr=1.0),
    dict(id="R7", title="Full-depth alpha64", projections="q_proj,k_proj,v_proj,o_proj",
         layers="0-31", cf_input="concat", alpha=64, p_lr=1.0),
    dict(id="R8", title="Full-depth CoRA-P learning rate x2", projections="q_proj,k_proj,v_proj,o_proj",
         layers="0-31", cf_input="concat", alpha=32, p_lr=2.0),
]
BY_ID = {e["id"]: e for e in EXPERIMENTS}


def prepared_config():
    if not READY.is_file():
        raise FileNotFoundError(f"Missing prepared CoRA config: {READY}")
    cfg = yaml.safe_load(READY.read_text())
    m, r = cfg["model"], cfg["run"]
    expectations = [
        (float(m.get("gnn_scale", -1)) == 0, "NO31 gnn_scale=0"),
        (float(m.get("gnn_conf_loss_weight", -1)) == 0.05, "CTR weight=0.05"),
        (m.get("gnn_semantic_align") is True, "gnn_semantic_align=True"),
        (float(m.get("gnn_semantic_weight", -1)) == 1e-4, "semantic weight=1e-4"),
        (m.get("fix_autocast_cache") is True, "autocast fix must be ON"),
        (m.get("rec_model") == "MF", "MF pretrained CF"),
        (m.get("lora_gnn") == "ngcf", "NGCF NO31"),
        (int(m["rec_config"]["embedding_size"]) == 256, "CF dim=256"),
        (float(m.get("proj_reg_weight", -1)) == 0.0005, "projector reg unchanged"),
        (int(r.get("seed", -1)) == 42, "seed42"),
        (int(r.get("batch_size_train", -1)) == 2, "train BS2"),
        (int(r.get("batch_size_eval", -1)) == 3, "eval BS3"),
        (int(r.get("accum_grad_iters", -1)) == 4, "accum4"),
        (int(r.get("max_epoch", -1)) == 3, "3 epoch LR horizon"),
        (float(r.get("init_lr", -1)) == 1e-4, "NO31 init LR=1e-4"),
        (not r.get("resume_ckpt_path"), "no resume"),
        (not m.get("ckpt"), "no CoRA checkpoint preload"),
        (not r.get("test_splits"), "Valid only, no Test"),
    ]
    for good, msg in expectations:
        if not good:
            raise RuntimeError(f"[NIGHT8-CONFIG-FAIL] {msg}. Fix {READY} first.")
    return cfg


def selected_targets(row):
    lo, hi = map(int, row["layers"].split("-"))
    return (hi - lo + 1) * len(row["projections"].split(","))


def build_environment(row):
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "S2_SMALL_KD": "0.05",
        "S2_INPUT_ALPHA": "0",
        "S2_NIGHT_EPOCHS": "1",
        "NO31_CORA_ALPHA": str(row["alpha"]),
        "NO31_CORA_PROJECTIONS": row["projections"],
        "NO31_CORA_LAYERS": row["layers"],
        "NO31_CORA_INPUT": row["cf_input"],
        "NO31_CORA_RESIDUAL_RATIO": str(row.get("residual_ratio", 0.0)),
        "NO31_CORA_DEPTH_PROFILE": row.get("depth_profile", "uniform"),
        "PYTHONUNBUFFERED": "1",
    })
    return env


def gpu_is_in_use():
    try:
        response = subprocess.check_output(
            ["nvidia-smi", "-i", "0", "--query-compute-apps=pid",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError("Cannot verify GPU compute processes with nvidia-smi")
    return bool(re.search(r"(?m)^\s*\d+\s*$", response))


def precheck(path):
    import torch
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    norms = {}
    for layer in range(32):
        layer_norms = []
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            tail = f"layers.{layer}.self_attn.{projection}.cora_lora_P.weight"
            matching = [key for key in sd if key.endswith(tail)]
            if len(matching) != 1:
                raise RuntimeError(f"Expected one original CoRA-P key for {tail}; got {matching}")
            layer_norms.append(float(sd[matching[0]].float().norm()))
        norms[layer] = layer_norms
    for layer in (0, 1, 3, 7, 8, 9, 16, 24, 31):
        print(f"[NIGHT8-PRECHECK] L{layer:02d} "
              + " ".join(f"{name}={norm:.6f}" for name,norm in
                         zip(("q","k","v","o"),norms[layer])), flush=True)
    if any(all(v==0 for v in norms[layer]) for layer in range(8)):
        raise RuntimeError(
            "[NIGHT8-PRECHECK-FAIL] At least one early CoRA layer remains "
            "zero after supposed autocast fix; do not run 8 expensive trials."
        )
    print("[NIGHT8-PRECHECK-PASS] early L0-L7 CoRA-P weights are nonzero", flush=True)


def plan(rows):
    cfg = prepared_config()
    print("[NIGHT8-PLAN] Baselines: NO31 AUC=%.9f UAUC=%.9f; "
          "repaired CoRA R0 AUC=%.9f UAUC=%.9f"
          % (BASELINE_AUC, BASELINE_UAUC,CORA_R0_AUC,CORA_R0_UAUC))
    print("[NIGHT8-COMMON] prepared=%s train_bs=2 eval_bs=3 accum=4 "
          "seed42 CTR=.05 KD=.05 fix_autocast_cache=True rank16 "
          "fresh_init=YES 1epoch Valid-only GPU0 foreground" % READY)
    for row in rows:
        print("[NIGHT8-PLAN] %s %-35s layers=%s proj=%s "
              "input=%s alpha=%d p_lr=%.1f P=%d"
              % (row["id"],row["title"],row["layers"],row["projections"],
                 row["cf_input"],row["alpha"],row["p_lr"],selected_targets(row)))
    return cfg


def run_one(row, common_cfg, root):
    ident = row["id"]
    exp_dir = root / ident
    exp_dir.mkdir(parents=False, exist_ok=False)
    cfg = yaml.safe_load(yaml.safe_dump(common_cfg))
    cfg["run"]["lr_scale_cora_p"] = float(row["p_lr"])
    # Follow-up experiments vary only the explicitly listed random seed.
    # Existing R1-R8 recipes omit seed and remain seed42.
    cfg["run"]["seed"] = int(row.get("seed", 42))
    if cfg["run"]["seed"] not in (42,43):
        raise ValueError(f"Unapproved seed for CoRA follow-up: {cfg['run']['seed']}")
    cfg["run"]["output_dir"] = str(ROOT / "output" / root.name / ident)
    cfg["run"]["valid_splits"] = ["valid"]
    cfg["run"]["test_splits"] = []
    cfg["run"]["save_checkpoint_every_n_epochs"] = 1
    cfg["run"]["evaluate"] = False
    config_path = exp_dir / "train.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    (exp_dir / "experiment.json").write_text(json.dumps(row, indent=2))
    command = [
        sys.executable, "-u", "experiments/no31_cora_original/entry.py",
        "--cfg-path", str(config_path),
    ]
    print(f"[NIGHT8-START] {ident} {row['title']} "
          f"seed={cfg['run']['seed']} selected_projections={selected_targets(row)} "
          f"log={exp_dir/'train.log'}",
          flush=True)
    log_file=exp_dir / "train.log"
    with log_file.open("x", encoding="utf-8") as saved:
        child = subprocess.Popen(
            command, cwd=str(ROOT), env=build_environment(row),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            encoding="utf-8", errors="replace", bufsize=1,
        )
        try:
            for line in child.stdout:
                saved.write(line)
                saved.flush()
                if any(word in line for word in
                       ("[CORA-ORIGINAL", "[NIGHT8-", "[FX]",
                        "[lr-group]", "Averaged stats:",
                        "OutOfMemory", "Traceback", "RuntimeError")):
                    print(f"[{ident}] {line.rstrip()[:700]}", flush=True)
            code = child.wait()
        except BaseException:
            child.terminate()
            child.wait()
            raise
    if code:
        raise RuntimeError(f"[{ident}] training failed exit={code}: {log_file}")

    text = log_file.read_text(errors="replace")
    if ("[CORA-ORIGINAL-PROBE-OK]" not in text or
        "[CORA-ORIGINAL-COUNTERS]" not in text or
        "[NIGHT8-STOP] trained_epochs=1" not in text or
        "[FX] autocast cache fix ON" not in text):
        raise RuntimeError(f"[{ident}] verification marker missing: {log_file}")
    if (f"static_P={selected_targets(row)} x [4096,16]" not in text or
        f"alpha={row['alpha']}" not in text or
        f"input={row['cf_input']}" not in text or
        f"projections={row['projections']}" not in text or
        f"layers={row['layers']}" not in text or
        f"residual_ratio={float(row.get('residual_ratio', 0.0)):g}" not in text or
        f"depth_profile={row.get('depth_profile', 'uniform')}" not in text):
        raise RuntimeError(f"[{ident}] requested CoRA architecture not loaded")
    lr_group_lines = [s for s in text.splitlines() if "[lr-group]" in s]
    target_group = [s for s in lr_group_lines if "cora_p_wd" in s]
    expected_scale = f"scale={row['p_lr']:.4g}"
    if not target_group or not any(expected_scale in s for s in target_group):
        raise RuntimeError(f"[{ident}] CoRA-P LR scale missing: {expected_scale}, {target_group}")

    metric_rows = []
    for line in text.splitlines():
        if "Averaged stats:" in line:
            match = AUC_PATTERN.search(line)
            if match:
                metric_rows.append((float(match.group(1)),float(match.group(2))))
    if len(metric_rows) != 1:
        raise RuntimeError(f"[{ident}] expected exactly 1 Valid: {metric_rows}")

    ctr_logs = re.findall(r"\[CORA-ORIGINAL-COUNTERS\]\s*(\{[^\n]*\})",text)
    counts = json.loads(ctr_logs[-1])
    if counts.get("projection_forwards",0) < selected_targets(row):
        raise RuntimeError(f"[{ident}] CoRA projection not executed: {counts}")
    checkpoints = sorted(Path(cfg["run"]["output_dir"]).rglob("checkpoint_0.pth"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"[{ident}] expected 1 checkpoint_0, got {checkpoints}")
    auc, uauc = metric_rows[0]
    result = dict(
        id=ident, title=row["title"], auc=auc, uauc=uauc,
        delta_vs_no31=auc-BASELINE_AUC,
        delta_vs_cora_r0=auc-CORA_R0_AUC,
        checkpoint=str(checkpoints[0]),
        log=str(log_file),
        targets=selected_targets(row), counts=counts,
        projections=row["projections"], layers=row["layers"],
        cf_input=row["cf_input"], alpha=row["alpha"], p_lr=row["p_lr"],
        residual_ratio=float(row.get("residual_ratio", 0.0)),
        depth_profile=row.get("depth_profile", "uniform"),
        seed=cfg["run"]["seed"],
    )
    (exp_dir / "result.json").write_text(json.dumps(result, indent=2))
    print("[NIGHT8-RESULT] %s ValidAUC=%.9f ValidUAUC=%.9f "
          "delta_vs_R0=%+.9f delta_vs_NO31=%+.9f"
          % (ident,auc,uauc,auc-CORA_R0_AUC,auc-BASELINE_AUC), flush=True)
    return result


def run_trials(rows, common_cfg, name_prefix="NO31_CORA_NIGHT8"):
    if name_prefix not in ("NO31_CORA_NIGHT8","NO31_CORA_FOLLOWUP8","NO31_CORA_NIGHT10"):
        raise ValueError(f"Invalid CoRA experiment folder prefix: {name_prefix}")
    # Refuse costly runs unless original CoRA's full-matrix CPU smoke
    # and every R1-R8 conditional configuration work in this environment.
    subprocess.run(
        [sys.executable, "experiments/no31_cora_original/test_night8_cpu.py"],
        cwd=str(ROOT), check=True,
    )
    if gpu_is_in_use():
        raise RuntimeError("GPU0 already has a compute process: refusing overlapping training")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = ROOT / "logs" / f"{name_prefix}_{stamp}"
    out.mkdir(exist_ok=False, parents=True)
    (out / "plan.json").write_text(json.dumps(
        dict(experiments=rows, source_config=str(READY),
             NO31_auc=BASELINE_AUC,CoRA_R0_auc=CORA_R0_AUC),
        indent=2,
    ))
    print("[NIGHT8-ROOT] "+str(out),flush=True)
    records = []
    for row in rows:
        try:
            result = run_one(row, common_cfg, out)
            # NIGHT10 begins with two exact historical anchors. Stop the sweep
            # early if either is far from its reference, rather than spending
            # the remaining GPU budget on an invalid comparison.
            if name_prefix == "NO31_CORA_NIGHT10" and row.get("anchor_auc") is not None:
                da = result["auc"] - float(row["anchor_auc"])
                du = result["uauc"] - float(row["anchor_uauc"])
                tol = float(row.get("anchor_tolerance", 0.02))
                print(f"[NIGHT10-ANCHOR] {row['id']} delta_auc={da:+.6f} "
                      f"delta_uauc={du:+.6f} tolerance={tol:.3f}", flush=True)
                if abs(da) > tol or abs(du) > tol:
                    raise RuntimeError(
                        f"{row['id']} anchor drift exceeds tolerance: "
                        f"delta_auc={da:+.6f}, delta_uauc={du:+.6f}, tolerance={tol:.3f}. "
                        "Stop and inspect initialization/data/config before interpreting this sweep."
                    )
        except BaseException as exc:
            (out / "failed.json").write_text(json.dumps(
                dict(id=row["id"], reason=str(exc)),indent=2))
            print(f"[NIGHT8-ABORT] {row['id']} failed. Previous "
                  f"results preserved in {out}; resume selectively "
                  "with --start/--only after fixing the cause.",flush=True)
            raise
        records.append(result)
        with (out/"results.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(result)+"\n")
        with (out/"summary.csv").open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=(
                "id","auc","uauc","delta_vs_cora_r0","delta_vs_no31",
                "layers","projections","cf_input","alpha","p_lr",
                "residual_ratio","depth_profile","seed","checkpoint"))
            writer.writeheader()
            for r in records:writer.writerow({k:r[k] for k in writer.fieldnames})
    print(f"[NIGHT8-COMPLETE] {len(records)} trials finished. "
          f"Summary: {out/'summary.csv'}",flush=True)


def main():
    parser=argparse.ArgumentParser()
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan",action="store_true")
    action.add_argument("--run",action="store_true")
    action.add_argument("--precheck",metavar="CKPT")
    parser.add_argument("--only",choices=tuple(BY_ID))
    parser.add_argument("--start",choices=tuple(BY_ID))
    args=parser.parse_args()
    if args.precheck:
        precheck(args.precheck)
        return
    if args.only and args.start:
        parser.error("Use --only or --start, not both")
    rows=EXPERIMENTS
    if args.only:rows=[BY_ID[args.only]]
    if args.start:
        ids=list(BY_ID)
        rows=EXPERIMENTS[ids.index(args.start):]
    cfg=plan(rows)
    if args.run:
        run_trials(rows,cfg)


if __name__=="__main__":
    main()
