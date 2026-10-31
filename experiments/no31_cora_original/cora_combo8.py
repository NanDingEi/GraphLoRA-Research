#!/usr/bin/env python3
"""Third-round CoRA/MF controlled loss combinations (8 independent VALID trials).

Dependencies: cora_joint8.py from the same directory, with
install_compact_atomic_best_checkpoint() and the earlier AMP/early-stop patches.
No official CoRA source changes, no TEST evaluation, no full-model checkpoints.
"""
import argparse
import json
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path

import yaml
import cora_joint8 as core

ROOT = Path(__file__).resolve().parent
PROTOCOL = "CORA_COMBO8_20261031"
BASE_LR = 3e-4
MF_LR_RATIO = 0.1
SEED = 42
STEPS_PER_EPOCH = 400
CONF_MARGIN = 0.20

# CTR and KD both supervise MF. "reverse" supervises the LLM Yes logit with
# the detached pretrained/updated MF probability and updates CoRA, NOT just MF.
# All weights multiply unreduced binary-cross-entropy, averaged over batch size.
TRIALS = [
    dict(name="C1_WEAK_BOTH", ctr=.005, kd=.005, kd_gate="all", rev=0., gamma=1.),
    dict(name="C2_CTR_DOM", ctr=.010, kd=.002, kd_gate="all", rev=0., gamma=1.),
    dict(name="C3_KD_DOM", ctr=.002, kd=.010, kd_gate="all", rev=0., gamma=1.),
    dict(name="C4_KD_AGREE", ctr=.005, kd=.010, kd_gate="agree", rev=0., gamma=1.),
    dict(name="C5_KD_CONF", ctr=.005, kd=.010, kd_gate="conf", rev=0., gamma=1.),
    dict(name="C6_KD_BOTH", ctr=.005, kd=.010, kd_gate="agree_conf", rev=0., gamma=1.),
    dict(name="C7_BIDIR", ctr=.005, kd=.010, kd_gate="all", rev=.005, gamma=1.),
    dict(name="C8_MAIN_HALF", ctr=.005, kd=.010, kd_gate="all", rev=0., gamma=.5),
]
NAMES = {x["name"]: x for x in TRIALS}

# Reuse the original verified config/optimizer/early-stop functions; all output
# goes into a new directory. This code must never overwrite earlier rounds.
core.RUNS = [(t["name"], False, t["ctr"], t["kd"], 0., 0.) for t in TRIALS]
core.PROTOCOL = PROTOCOL
core.CONFIG_DIR = ROOT / "local_configs" / PROTOCOL
core.LOG_DIR = ROOT / "logs" / PROTOCOL
core.OUT_DIR = ROOT / "output" / PROTOCOL


def preflight():
    if Path(core.__file__).resolve() != ROOT / "cora_joint8.py":
        raise RuntimeError("cora_joint8.py must be alongside cora_combo8.py")
    if not hasattr(core, "install_compact_atomic_best_checkpoint"):
        raise RuntimeError("Please update cora_joint8.py: trainable-only checkpoint patch missing")
    if core.CORA_LR != BASE_LR or core.MF_LR_RATIO != MF_LR_RATIO:
        raise RuntimeError("CoRA LR / MF LR ratio differs from experiment protocol")
    core.check_sources()
    if len(NAMES) != 8:
        raise RuntimeError("Expected exactly 8 unique trial names")
    if any(t["kd_gate"] not in ("all", "agree", "conf", "agree_conf")
           for t in TRIALS):
        raise RuntimeError("Unrecognized KD sample gate")


def prepare():
    preflight()
    core.prepare()  # new protocol directory, only; does not train any model
    for t in TRIALS:
        cfg = core.CONFIG_DIR / (t["name"] + ".yaml")
        c = yaml.safe_load(cfg.read_text())
        if (c["model"]["freeze_rec"] or c["run"]["seed"] != SEED or
            abs(float(c["run"]["init_lr"]) - BASE_LR) > 1e-12 or
            c["run"]["iters_per_epoch"] != STEPS_PER_EPOCH or
            c["run"].get("test_splits") or c["run"].get("resume_ckpt_path")):
            raise RuntimeError(f"Trial config violates controls: {cfg}")
    manifest = {
        "protocol": PROTOCOL, "seed": SEED, "gpu": "0",
        "main_loss": "original CoRA Yes-logit BCE with true labels",
        "ctr": "BCEWithLogits(MF user-item dot, ground-truth label)",
        "kd": "BCEWithLogits(MF dot, stopgrad(sigmoid(LLM Yes logit)))",
        "kd_agree": "only when LLM p>=.5 matches true y",
        "kd_conf": "only when abs(LLM p-.5)>=.20",
        "reverse": "BCEWithLogits(LLM Yes logit, stopgrad(sigmoid(MF dot)))",
        "reverse_mask": "MF prob agreement with label AND abs(MF p-.5)>=.20",
        "masked_loss": "per-example BCE, inactive examples zeroed, mean over full batch",
        "gamma": "gradient gain from CoRA main loss to MF through weight-generator input (value unchanged)",
        "best_model_checkpoint": "trainable-parameters only; atomic save; no optimizer/scaler/frozen LLM",
        "controls": {
            "cora_lr": BASE_LR, "mf_lr": BASE_LR * MF_LR_RATIO,
            "bs": 2, "accum": 4, "iters_per_epoch": STEPS_PER_EPOCH,
            "patience": 10, "valid": "valid_small", "test": False,
        },
        "trials": TRIALS,
    }
    (core.LOG_DIR / "combo8_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[COMBO8 READY] 8 independent configs; logs and checkpoints isolated.", flush=True)
    for t in TRIALS:
        print("[COMBO8 TRIAL] " + json.dumps(t, sort_keys=True), flush=True)


def register_mf_main_gradient_gain(model, gamma):
    """Keep ui_embedding forward value unchanged; scale only its main-loss grad.

    generator gradients stay fully enabled. This affects only the CoRA
    weight-generator input, not the independent MF auxiliary losses.
    """
    if gamma == 1:
        return
    generator = model.llama_model_lora.base_model.weight_generator
    if getattr(generator, "_combo8_scaled", False):
        return
    base_forward = generator.forward

    def gradscaled_forward(ui_embedding):
        u = ui_embedding.detach() + gamma * (ui_embedding - ui_embedding.detach())
        return base_forward(u)

    generator.forward = gradscaled_forward
    generator._combo8_scaled = True
    print(f"[COMBO8 MF MAIN GRAD] gamma={gamma:.3f}; "
          "input value unchanged; CoRA generator remains trainable", flush=True)


def install_combo_model(t):
    import torch
    import torch.nn.functional as F
    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx

    original = MiniGPT4Rec_vx.forward_v2
    yes_token = None

    def forward(self, samples):
        nonlocal yes_token
        if not hasattr(self, "_combo8_ready"):
            self._combo8_ready = True
            mf_trainable = sum(p.numel() for p in self.rec_encoder.parameters()
                               if p.requires_grad)
            if not mf_trainable:
                raise RuntimeError("[COMBO8] MF must be trainable in every combo")
            if self.rec_encoder.user_embedding.weight.dtype != torch.float32:
                raise RuntimeError("[COMBO8] trainable MF embeddings must be float32")
            register_mf_main_gradient_gain(self, t["gamma"])
            self._combo8_microsteps = 0
            print(f"[COMBO8 MODEL] {t['name']} trainable_mf={mf_trainable} "
                  f"ctr={t['ctr']} kd={t['kd']} gate={t['kd_gate']} "
                  f"reverse={t['rev']} gamma={t['gamma']}", flush=True)

        if yes_token is None:
            yes_token = int(self.llama_tokenizer(
                self.pos_ans[0], add_special_tokens=False).input_ids[0])
        answers = [self.pos_ans[0] if int(y) == 1 else self.neg_ans[0]
                   for y in samples["label"]]
        tokens = self.llama_tokenizer(
            answers, return_tensors="pt", padding="longest", truncation=True,
            max_length=self.max_txt_len, add_special_tokens=False)
        offset = tokens.input_ids.shape[-1] + 1

        captured = {}
        def capture_yes(_module, _inputs, output):
            captured["yes"] = output.logits[:, -offset, yes_token]

        # A short-lived hook captures the precise same Yes logit used by
        # official forward_v2() without changing official CoRA implementation.
        hook = self.llama_model_lora.register_forward_hook(capture_yes)
        try:
            result = original(self, samples)
        finally:
            hook.remove()
        if "yes" not in captured:
            raise RuntimeError("[COMBO8] missing LLM Yes-logit hook output")

        llm_yes = captured["yes"].float()
        if not bool(torch.isfinite(llm_yes).all()):
            raise RuntimeError("[COMBO8] nonfinite LLM Yes logits; refusing to mask NaNs")
        labels = samples["label"].float().to(llm_yes.device)
        mf_logits = self.rec_encoder(
            samples["UserID"].long(),
            samples["TargetItemID"].long()).float()
        if not bool(torch.isfinite(mf_logits).all()):
            raise RuntimeError("[COMBO8] nonfinite MF logits")
        llm_p = llm_yes.detach().sigmoid()
        mf_p = mf_logits.detach().sigmoid()

        # Main CoRA loss remains the original output BCE, and always trains
        # CoRA parameters. For gamma=1 its MF gradient pathway is unchanged.
        extra = mf_logits.sum() * 0.
        details = {}
        ctr = F.binary_cross_entropy_with_logits(mf_logits, labels)
        extra = extra + t["ctr"] * ctr
        details["ctr"] = float(ctr.detach())

        kd_per = F.binary_cross_entropy_with_logits(
            mf_logits, llm_p, reduction="none")
        agree = (llm_p >= .5) == (labels >= .5)
        confidence = (llm_p - .5).abs() >= CONF_MARGIN
        gate = t["kd_gate"]
        active = (
            torch.ones_like(agree, dtype=torch.bool) if gate == "all" else
            agree if gate == "agree" else
            confidence if gate == "conf" else
            (agree & confidence)
        )
        kd = (kd_per * active.float()).mean()
        extra = extra + t["kd"] * kd
        details["kd"] = float(kd.detach())
        details["kd_active"] = float(active.float().mean().detach())

        if t["rev"] > 0:
            # Only trusted, label-consistent MF predictions supervise the LLM.
            # MF p is stop-gradient; LLM logits are the student.
            mf_agree = (mf_p >= .5) == (labels >= .5)
            mf_conf = (mf_p - .5).abs() >= CONF_MARGIN
            reverse_active = mf_agree & mf_conf
            rev_each = F.binary_cross_entropy_with_logits(
                llm_yes, mf_p, reduction="none")
            rev = (rev_each * reverse_active.float()).mean()
            extra = extra + t["rev"] * rev
            details["rev"] = float(rev.detach())
            details["rev_active"] = float(reverse_active.float().mean().detach())

        result["loss"] = result["loss"] + extra
        if not bool(torch.isfinite(result["loss"]).all()):
            raise RuntimeError("[COMBO8] nonfinite joint loss")
        self._combo8_microsteps += 1
        if self._combo8_microsteps == 1 or self._combo8_microsteps % 100 == 0:
            print(f"[COMBO8 LOSS] {t['name']} step={self._combo8_microsteps} "
                  f"main={float((result['loss']-extra).detach()):.6g} "
                  f"extra={float(extra.detach()):.6g} terms={details}",
                  flush=True)
        return result

    MiniGPT4Rec_vx.forward_v2 = forward


def worker(name, patience, epoch_cap):
    preflight()
    if name not in NAMES:
        raise ValueError("Unknown trial: " + name)
    core.patch_runner(
        screen_epochs=epoch_cap, mf_lr_ratio=MF_LR_RATIO, patience=patience)
    core.install_compact_atomic_best_checkpoint()
    install_combo_model(NAMES[name])
    cfg_path = core.CONFIG_DIR / (name + ".yaml")
    if not cfg_path.is_file():
        raise FileNotFoundError("Run 'python cora_combo8.py' first: " + str(cfg_path))
    c = yaml.safe_load(cfg_path.read_text())
    if c["run"]["seed"] != SEED or c["model"]["freeze_rec"] is not False:
        raise RuntimeError("Unexpected run config in " + str(cfg_path))
    sys.argv = ["train_collm_mf_din.py", "--cfg-path", str(cfg_path)]
    runpy.run_path(str(ROOT / "train_collm_mf_din.py"), run_name="__main__")


EVAL = re.compile(
    r"\*\*\*auc:\s*([0-9]+(?:\.[0-9]+)?)\s+\*\*\*uauc:\s*([0-9]+(?:\.[0-9]+)?)")

def summarize():
    print(f"{'Trial':20} {'Best Valid AUC':>15} {'UAUC@AUCbest':>16} "
          f"{'Epoch':>6} {'Status':>10}")
    for t in TRIALS:
        name = t["name"]
        log = core.LOG_DIR / (name + ".log")
        if not log.exists():
            print(f"{name:20} {'pending':>15}")
            continue
        text = log.read_text(errors="replace")
        points = [(float(x), float(y)) for x,y in EVAL.findall(text)]
        if not points:
            print(f"{name:20} {'no-valid':>15} {'':>16} {'':>6} "
                  f"{'FAILED' if 'Traceback' in text else 'INCOMPLETE'}")
            continue
        idx = max(range(len(points)), key=lambda i: points[i][0])
        status = "FAILED" if "Traceback (most recent call last)" in text else (
                 "DONE" if "Training time" in text else "INCOMPLETE")
        # Unlike a numbered DataFrame, each evaluation starts at epoch 0.
        # In the unmodified runner, one full Valid evaluation follows one epoch.
        print(f"{name:20} {points[idx][0]:15.6f} {points[idx][1]:16.6f} "
              f"{idx:6d} {status:>10}")


def run_selected(names, patience, epoch_cap):
    for name in names:
        if name not in NAMES:
            raise ValueError("Unknown trial: " + name)
    for name in names:
        path = core.LOG_DIR / (name + ".log")
        if path.exists():
            raise RuntimeError(f"Run log exists; preserving it: {path}.\n"
                               "Use --only with unstarted trial names. "
                               "Do not overwrite checkpoints or results.")
        command = [sys.executable, "-u", str(ROOT / "cora_combo8.py"),
                   "--_worker", name, "--patience", str(patience),
                   "--screen-epochs", str(epoch_cap)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
        print(f"[COMBO8 START] {name}, GPU0 serial foreground; log={path}", flush=True)
        with path.open("x", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command, env=env, cwd=ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, encoding="utf-8",
                errors="replace", bufsize=1)
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
            code = process.wait()
        if code != 0:
            raise SystemExit(f"[COMBO8 STOP ON FAILURE] {name}, exit={code}; "
                             f"see {path}")
        summarize()
    print("[COMBO8 COMPLETE] requested trials completed; VALID-only.", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true")
    p.add_argument("--only", default="")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--screen-epochs", type=int, default=0)
    p.add_argument("--summary", action="store_true")
    p.add_argument("--_worker", default="", help=argparse.SUPPRESS)
    args = p.parse_args()
    os.chdir(ROOT)
    if args._worker:
        worker(args._worker, args.patience, args.screen_epochs)
        return
    if args.summary:
        summarize()
        return
    if args.patience < 1 or args.screen_epochs < 0:
        p.error("patience >= 1 and screen-epochs >= 0 required")
    prepare()
    if not args.run:
        print("[COMBO8 PLAN ONLY] no GPU training started", flush=True)
        return
    names = [x.strip() for x in args.only.split(",") if x.strip()] if args.only \
        else [t["name"] for t in TRIALS]
    run_selected(names, args.patience, args.screen_epochs)

if __name__ == "__main__":
    main()
