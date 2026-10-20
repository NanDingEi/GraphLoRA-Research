#!/usr/bin/env python3
"""VG0/VG1: candidate-aware rank-gated GraphLoRA L31 V injection.

VG0 has user/current-node conditioning only. VG1 adds candidate and
candidate*node conditioning. Both preserve historical L31 Q/V injection,
warm-start S2-L31, train 1 epoch serially on GPU0, and evaluate Valid Small.
Original LoRA, GNN, rec and projector parameters are frozen. CPU unit test,
zero-init valid AUC anchor, and 24-batch gradient smoke are mandatory.
"""
import copy
import datetime
import os
import re
import subprocess
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"

ARMS = [
    ("VG0_NO_CANDIDATE", "target_v_gate_no_candidate"),
    ("VG1_CANDIDATE", "target_v_gate_candidate"),
]

def run(cmd, log):
    print("[NOON-CMD]", " ".join(map(str, cmd)), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        process = subprocess.Popen(
            list(map(str, cmd)), cwd=ROOT, env=os.environ.copy(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            errors="replace", bufsize=1,
        )
        try:
            for line in process.stdout:
                f.write(line)
                f.flush()
                if len(line) > 2500:
                    print(line[:220] + " ... [full log line saved]", flush=True)
                else:
                    print(line, end="", flush=True)
            rc = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
    if rc != 0:
        raise RuntimeError(f"Command returned {rc}; log: {log}")

def ckpt_in(directory, filename):
    paths = list(directory.rglob(filename))
    if len(paths) != 1:
        raise RuntimeError(f"Expected 1 {filename} in {directory}, got {len(paths)}")
    return paths[0]

def load_ckpt(path):
    return torch.load(str(path), map_location="cpu", weights_only=False)

def inspect_trainable(ckpt_path, arm):
    state = load_ckpt(ckpt_path)["model"]
    gate = sorted(k for k in state if ".target_v_gate." in k)
    unexpected = sorted(set(state) - set(gate))
    if len(gate) != 4 or unexpected:
        raise RuntimeError(
            f"{arm}: expected only four V gate tensors, found "
            f"{len(gate)}; unexpected={unexpected[:8]}"
        )
    last = [k for k in gate if k.endswith(".network.2.weight")]
    if len(last) != 1:
        raise RuntimeError(f"Missing V gate output weight in {gate}")
    norm = float(state[last[0]].float().norm())
    if not (0 < norm < float("inf")):
        raise RuntimeError(f"{arm}: zero or non-finite output gate after training: {norm}")
    print(f"[VG-UPDATE-PASS] {arm} gate=4 other=0 output_norm={norm:.8g}", flush=True)
    return state

def main():
    others = subprocess.run(
        ["pgrep", "-af", "[t]rain_collm_mf_din.py"],
        capture_output=True, text=True, check=False,
    )
    if others.returncode == 0 and others.stdout.strip():
        raise RuntimeError("Another GraphLoRA training process is running:\n" + others.stdout)

    model_py = ROOT / "minigpt4/models/minigpt4rec_v2.py"
    if "class TargetConditionedVGate" not in model_py.read_text(encoding="utf-8"):
        raise RuntimeError("Missing VG model code; fetch latest branch")

    run([sys.executable, "-m", "py_compile", str(model_py)], ROOT / "logs/collab_v1/noon_syntax.log")
    run([sys.executable, "experiments/collab_injection_v1/test_target_v_gate_cpu.py"],
        ROOT / "logs/collab_v1/vg_cpu_unit.log")
    tag = "S2_VG_2EP_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logs = ROOT / "logs/collab_v1" / tag
    outputs = ROOT / "output" / tag
    logs.mkdir(parents=True)
    outputs.mkdir(parents=True)

    run([sys.executable, "experiments/collab_injection_v1/prepare_e3_configs.py"], logs / "prepare.log")
    source_cfg = yaml.safe_load(
        (ROOT / "tran_configs/collab_v1/e3/S2_TARGET_FUSION.yaml").read_text(encoding="utf-8")
    )
    champion = Path(source_cfg["model"]["ckpt"]).resolve()
    if not champion.is_file():
        raise FileNotFoundError(champion)
    assert source_cfg["model"]["collab_mode"] == "target_fusion"
    assert int(source_cfg["model"]["gcn_layer"]) == 31
    assert float(source_cfg["model"]["gnn_scale"]) == 0.1

    print(f"[VG] {tag}: 2 independent arms x 1 real epoch, GPU0", flush=True)
    print(f"[VG] S2 champion={champion}", flush=True)
    summary = logs / "summary.tsv"
    summary.write_text("arm\tauc\tuauc\tmerged_checkpoint\n", encoding="utf-8")

    for name, mode in ARMS:
        cfg = copy.deepcopy(source_cfg)
        m, r = cfg["model"], cfg["run"]
        m["collab_mode"] = mode
        m["trainable_policy"] = "target_v_gate_only"
        m.pop("target_fusion_scale", None)
        m.pop("target_fusion_mode", None)
        m.pop("target_fusion_relative_cap", None)
        r.update({
            "seed": 42, "max_epoch": 3, "diagnostic_stop_after_epoch0": True,
            "stop_after_n_epochs": 1,
            "evaluate": False, "skip_validation": False,
            "resume_ckpt_path": None, "save_checkpoint_every_n_epochs": 1,
            "accum_grad_iters": 3, "init_lr": 1e-5,
            "min_lr": 5e-6, "warmup_lr": 1e-6,
            "lr_scale_lora": 0.0, "lr_scale_other": 1.0,
            "lr_scale_rec": 0.0, "lr_scale_proj": 0.0, "lr_scale_gnn": 0.0,
        })
        for stale in ("max_train_steps", "iters_per_epoch", "max_iters", "mid_epoch_eval_iters"):
            r.pop(stale, None)
        # Never resume optimizer or use previous experiment's checkpoint.
        assert Path(m["ckpt"]).resolve() == champion
        assert not r["resume_ckpt_path"]
        out = outputs / name

        smoke = copy.deepcopy(cfg)
        smoke["run"]["max_train_steps"] = 24
        smoke["run"]["skip_validation"] = True
        smoke["run"]["output_dir"] = str(out / "SMOKE_24")
        smoke_cfg = logs / f"{name}_SMOKE.yaml"
        smoke_cfg.write_text(yaml.safe_dump(smoke, sort_keys=False), encoding="utf-8")

        cfg["run"]["output_dir"] = str(out / "FULL_E1")
        full_cfg = logs / f"{name}_FULL.yaml"
        full_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        print(f"[VG-START] {name}: zero-init valid S2 anchor check", flush=True)
        anchor_result = logs / f"{name}_ANCHOR.json"
        run([sys.executable, "-u",
             "experiments/collab_injection_v1/eval_target_v_gate_anchor.py",
             "--config", full_cfg, "--checkpoint", champion,
             "--expected-auc", "0.7296752596287361",
             "--result", anchor_result],
            logs / f"{name}_ANCHOR.log")

        print(f"[VG-START] {name}: 24-batch gate gradient smoke", flush=True)
        run([sys.executable, "-u", "train_collm_mf_din.py", "--cfg-path", smoke_cfg],
            logs / f"{name}_SMOKE.log")
        inspect_trainable(ckpt_in(out / "SMOKE_24", "checkpoint_0.pth"), name)

        print(f"[VG-START] {name}: 1 full epoch + Valid Small", flush=True)
        full_log = logs / f"{name}_FULL.log"
        run([sys.executable, "-u", "train_collm_mf_din.py", "--cfg-path", full_cfg], full_log)
        ckpt_path = ckpt_in(out / "FULL_E1", "checkpoint_best.pth")
        train_state = inspect_trainable(ckpt_path, name)

        champ_checkpoint = load_ckpt(champion)
        trained_checkpoint = load_ckpt(ckpt_path)
        state = dict(champ_checkpoint["model"])
        state.update(train_state)
        merged_path = out / "checkpoint_merged_best.pth"
        torch.save({
            "model": state,
            "config": trained_checkpoint["config"],
            "epoch": trained_checkpoint["epoch"],
            "metadata": {"champion": str(champion), "trained": str(ckpt_path)},
        }, str(merged_path))
        del state, champ_checkpoint, trained_checkpoint, train_state

        num = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
        pattern = re.compile(r"\*\*\*auc:\s*" + num + r".*?\*\*\*uauc:\(\s*" + num)
        scores = []
        for line in full_log.read_text(encoding="utf-8", errors="replace").splitlines():
            if "Averaged stats:" not in line:
                continue
            match = pattern.search(line)
            if match:
                scores.append((float(match.group(1)), float(match.group(2))))
        if len(scores) != 1:
            raise RuntimeError(f"{name}: wanted one validation AUC/UAUC; found {len(scores)}")
        auc, uauc = scores[0]
        with summary.open("a", encoding="utf-8") as f:
            f.write(f"{name}\t{auc:.9f}\t{uauc:.9f}\t{merged_path}\n")
        print(f"[VG-VALID] {name} AUC={auc:.9f} UAUC={uauc:.9f}", flush=True)
        print(f"[VG-MERGED] {name} checkpoint: {merged_path}", flush=True)

        run([sys.executable,
             "experiments/collab_injection_v1/compare_vg_checkpoint_drift.py",
             "--champion", champion, "--trained", merged_path],
            logs / f"{name}_DRIFT.log")

    print("[VG-2EP-COMPLETE] two independent experiments completed", flush=True)
    print(summary.read_text(encoding="utf-8"), flush=True)
    print(f"[VG] logs={logs} checkpoints={outputs}", flush=True)


if __name__ == "__main__":
    main()
