#!/usr/bin/env python3
"""D4-based late-layer voting and mutual-check CoRA experiments.

The official CoRA forward predicts the Yes-token at sequence position -t_posi.
This runner hooks only the last four decoder outputs (layers 24, 28, 31, 32),
applies the SAME final RMSNorm and lm_head, and extracts their Yes logits.
It retains C6 losses and D4 TopK40/20/10/5 + 25% MF residual inputs.

L1: train/evaluate with the mean late-layer Yes probability (soft voting).
L2: final-layer BCE + pairwise symmetric Bernoulli KL (lambda=.02).
L3: same consistency objective, but evaluate with 3-of-4 consensus gating.
L4: final-layer BCE + auxiliary late-layer BCE (.10), evaluate with soft voting.

No all-layer output_hidden_states tensor is retained; hooks keep only one
hidden vector per selected layer and sample. VALID only, seed42, GPU0.
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
import torch.nn.functional as F
import yaml

import cora_rankgran7 as base
import cora_struct10 as gran

ROOT = Path(__file__).resolve().parent
PROTOCOL = "CORA_LATEVOTE4_20261102"
SEED = 42
CORA_LR = 3e-4
MF_LR_RATIO = 0.1
PATIENCE = 10
LATE_LAYERS = (23, 27, 30, 31)  # 0-based: decoder layers 24, 28, 31, 32

TRIALS = [
    dict(name="L1_LATE_SOFTVOTE", mode="neighbor_mix", personal_share=.25,
         late_mode="soft_vote", consistency_lambda=0.0, layer_aux_lambda=0.0),
    dict(name="L2_LATE_PAIRKL", mode="neighbor_mix", personal_share=.25,
         late_mode="final", consistency_lambda=.02, layer_aux_lambda=0.0),
    dict(name="L3_LATE_CONSENSUS", mode="neighbor_mix", personal_share=.25,
         late_mode="consensus", consistency_lambda=.02, layer_aux_lambda=0.0),
    dict(name="L4_LATE_AUXBCE", mode="neighbor_mix", personal_share=.25,
         late_mode="soft_vote", consistency_lambda=0.0, layer_aux_lambda=.10),
]
TRIAL_BY_NAME = {t["name"]: t for t in TRIALS}
ORDER = [t["name"] for t in TRIALS]


def configure():
    base.PROTOCOL = PROTOCOL
    base.TRIALS = TRIALS
    base.TRIAL_BY_NAME = TRIAL_BY_NAME
    base.ALL_ORDER = ORDER
    base._make_group_inputs = gran.group_inputs
    base._configure_initialized_model = gran.configure_initialized_model
    base.configure_core()


def _answer_offset(model, samples):
    answers = [model.pos_ans[0] if int(y) == 1 else model.neg_ans[0]
               for y in samples["label"]]
    tokens = model.llama_tokenizer(
        answers, return_tensors="pt", padding="longest",
        truncation=True, max_length=model.max_txt_len,
        add_special_tokens=False)
    return int(tokens.input_ids.shape[-1]) + 1


def _capture_late(model, samples, fn, *args, **kwargs):
    """Capture only the target-position hidden vector at selected late layers."""
    causal_lm = model.llama_model_lora.base_model.model
    decoder = causal_lm.model
    if len(decoder.layers) != 32:
        raise RuntimeError(f"Expected 32 Vicuna layers, got {len(decoder.layers)}")
    if getattr(decoder, "gradient_checkpointing", False):
        raise RuntimeError("Late-layer hooks require gradient_checkpointing=False")
    offset = _answer_offset(model, samples)
    yes_token = int(model.llama_tokenizer(
        model.pos_ans[0], add_special_tokens=False).input_ids[0])
    captured = {"layers": {}}

    def make_layer_hook(layer_idx):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if hidden.ndim != 3 or hidden.shape[1] < offset:
                raise RuntimeError(
                    f"Invalid hidden shape at layer {layer_idx}: {tuple(hidden.shape)}")
            captured["layers"][layer_idx] = hidden[:, -offset, :]
        return hook

    def final_hook(_module, _inputs, output):
        logits = output.logits if hasattr(output, "logits") else output[0]
        captured["final"] = logits[:, -offset, yes_token].float()

    handles = [
        decoder.layers[idx].register_forward_hook(make_layer_hook(idx))
        for idx in LATE_LAYERS
    ]
    handles.append(model.llama_model_lora.register_forward_hook(final_hook))
    try:
        result = fn(model, samples, *args, **kwargs)
    finally:
        for handle in handles:
            handle.remove()

    missing = [idx for idx in LATE_LAYERS if idx not in captured["layers"]]
    if missing or "final" not in captured:
        raise RuntimeError(
            f"Late-layer capture incomplete: missing_layers={missing}, "
            f"final={'final' in captured}")
    layer_logits = []
    for idx in LATE_LAYERS:
        hidden = captured["layers"][idx]
        normalized = decoder.norm(hidden)
        logits = causal_lm.lm_head(normalized)
        layer_logits.append(logits[:, yes_token].float())
    layer_logits = torch.stack(layer_logits, dim=0)  # [4, batch]
    final_logits = captured["final"]
    if not torch.isfinite(layer_logits).all() or not torch.isfinite(final_logits).all():
        raise RuntimeError("Nonfinite final/late-layer Yes logits")
    # Layer 32, after the shared final RMSNorm/head, must reproduce final logits.
    max_error = float((layer_logits[-1] - final_logits).detach().abs().max())
    if max_error > 0.05:
        raise RuntimeError(
            f"Layer-32 reconstruction differs from official output: max_abs={max_error:.6g}")
    return result, final_logits, layer_logits, max_error


def _probs(layer_logits):
    return torch.sigmoid(layer_logits.float()).clamp(1e-6, 1.0 - 1e-6)


def _prob_to_logit(prob):
    prob = prob.float().clamp(1e-6, 1.0 - 1e-6)
    return torch.log(prob) - torch.log1p(-prob)


def _soft_vote(layer_logits):
    return _probs(layer_logits).mean(dim=0)


def _pairwise_symmetric_bernoulli_kl(layer_logits):
    p = _probs(layer_logits)
    values = []
    for i in range(p.shape[0]):
        for j in range(i + 1, p.shape[0]):
            a, b = p[i], p[j]
            kl_ab = a * (a.log() - b.log()) + (1-a) * ((1-a).log() - (1-b).log())
            kl_ba = b * (b.log() - a.log()) + (1-b) * ((1-b).log() - (1-a).log())
            values.append(0.5 * (kl_ab + kl_ba).mean())
    return torch.stack(values).mean()


def _consensus_probability(layer_logits, final_logits):
    p = _probs(layer_logits)
    votes_yes = (p >= 0.5).sum(dim=0)
    # 4 experts: 3/4 or 4/4 agree. A 2:2 tie falls back to the final layer.
    confident_consensus = (votes_yes >= 3) | (votes_yes <= 1)
    vote_p = p.mean(dim=0)
    return torch.where(confident_consensus, vote_p, torch.sigmoid(final_logits.float()))


def install_late_logits(trial):
    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx

    for method_name in ("forward_v2", "generate_for_samples_v2"):
        marker = "_latevote4_wrapped_" + method_name
        if getattr(MiniGPT4Rec_vx, marker, False):
            raise RuntimeError(f"Refusing to double-wrap {method_name}")
        original = getattr(MiniGPT4Rec_vx, method_name)

        def make_wrapper(fn, name):
            def wrapped(self, samples, *args, **kwargs):
                result, final_logits, layer_logits, max_error = _capture_late(
                    self, samples, fn, *args, **kwargs)
                labels = samples["label"].float().to(final_logits.device)
                final_main = F.binary_cross_entropy_with_logits(final_logits, labels)
                vote_p = _soft_vote(layer_logits)
                vote_logit = _prob_to_logit(vote_p)
                consistency = _pairwise_symmetric_bernoulli_kl(layer_logits)
                aux_bce = F.binary_cross_entropy_with_logits(
                    layer_logits, labels.unsqueeze(0).expand_as(layer_logits))
                mode = trial["late_mode"]

                if name == "forward_v2":
                    if not isinstance(result, dict) or "loss" not in result:
                        raise RuntimeError("Expected training forward_v2 to return a loss dict")
                    if mode == "soft_vote":
                        # Replace only the original final-layer main BCE.
                        # Preserve C6 CTR/KD auxiliary losses from combo8.
                        c6_extras = result["loss"] - final_main
                        main = F.binary_cross_entropy_with_logits(vote_logit, labels)
                        result["loss"] = c6_extras + main
                    else:
                        result["loss"] = result["loss"] + (
                            float(trial["consistency_lambda"]) * consistency
                        ) + (float(trial["layer_aux_lambda"]) * aux_bce)
                    if not torch.isfinite(result["loss"]).all():
                        raise RuntimeError("Nonfinite late-layer joint loss")
                    self._latevote4_steps = getattr(self, "_latevote4_steps", 0) + 1
                    if self._latevote4_steps == 1 or self._latevote4_steps % 100 == 0:
                        means = _probs(layer_logits).mean(dim=1).detach().cpu().tolist()
                        print(
                            f"[LATEVOTE4 LOSS] {trial['name']} "
                            f"step={self._latevote4_steps} mode={mode} "
                            f"final_bce={float(final_main.detach()):.6f} "
                            f"pair_kl={float(consistency.detach()):.6g} "
                            f"aux_bce={float(aux_bce.detach()):.6f} "
                            f"layer_mean_p={[round(x,5) for x in means]} "
                            f"layer32_max_error={max_error:.6g}",
                            flush=True)
                    return result

                # Evaluation: retain the original raw-logit convention.
                if mode == "soft_vote":
                    selected_logit = vote_logit
                elif mode == "consensus":
                    selected_logit = _prob_to_logit(
                        _consensus_probability(layer_logits, final_logits))
                else:
                    selected_logit = final_logits
                if isinstance(result, tuple) and len(result) == 2:
                    return result[0], selected_logit
                if not isinstance(result, dict):
                    raise RuntimeError("Unexpected generate_for_samples_v2 return type")
                result["logits"] = selected_logit
                result["loss"] = F.binary_cross_entropy_with_logits(selected_logit, labels)
                return result
            return wrapped

        setattr(MiniGPT4Rec_vx, method_name, make_wrapper(original, method_name))
        setattr(MiniGPT4Rec_vx, marker, True)

    print(
        f"[LATEVOTE4 INSTALLED] {trial['name']} layers={LATE_LAYERS} "
        f"mode={trial['late_mode']} pair_kl={trial['consistency_lambda']} "
        f"layer_aux={trial['layer_aux_lambda']}; D4 input and C6 losses retained",
        flush=True)


def _cpu_test():
    torch.manual_seed(42)
    z = torch.randn(4, 7, requires_grad=True)
    p = _probs(z)
    vote = _soft_vote(z)
    assert vote.shape == (7,) and torch.isfinite(vote).all()
    loss = F.binary_cross_entropy_with_logits(_prob_to_logit(vote),
                                               torch.randint(0, 2, (7,)).float())
    kl = _pairwise_symmetric_bernoulli_kl(z)
    (loss + .02 * kl).backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    final = z[-1].detach()
    gated = _consensus_probability(z.detach(), final)
    assert gated.shape == (7,) and torch.isfinite(gated).all()
    assert len(TRIALS) == 4 and len(set(ORDER)) == 4
    assert all(t["mode"] == "neighbor_mix" and t["personal_share"] == .25
               for t in TRIALS)
    print("[LATEVOTE4 CPU PASS] soft vote, symmetric Bernoulli KL, "
          "consensus gate and gradients", flush=True)


def prepare():
    configure()
    base.core.check_sources()
    base._model_api_preflight()
    base.core.prepare()
    for trial in TRIALS:
        cfg_path = base.core.CONFIG_DIR / (trial["name"] + ".yaml")
        cfg = yaml.safe_load(cfg_path.read_text())
        if (cfg["run"]["seed"] != SEED or cfg["model"]["freeze_rec"] is not False or
                cfg["run"].get("test_splits") or cfg["run"].get("resume_ckpt_path") or
                cfg["run"]["iters_per_epoch"] != 400 or
                abs(float(cfg["run"]["init_lr"]) - CORA_LR) > 1e-12):
            raise RuntimeError(f"Control mismatch in {cfg_path}")
    manifest = {
        "protocol": PROTOCOL, "seed": SEED, "gpu": "0",
        "backbone": "D4_TOPK_RES025: TopK40/20/10/5, MF residual share=.25",
        "loss": "C6_KD_BOTH: original LLM BCE + CTR=.005 + agreement/confidence gated KD=.01",
        "rank": 16, "alpha": 32, "late_layers_zero_based": list(LATE_LAYERS),
        "late_layers_one_based": [x+1 for x in LATE_LAYERS],
        "late_logit": "capture each selected layer output at the exact official -t_posi position; apply shared final RMSNorm and lm_head; extract Yes token",
        "training": {"seed": SEED, "cora_lr": CORA_LR, "mf_lr": CORA_LR*MF_LR_RATIO,
                     "batch_size": 2, "accum_grad_iters": 4, "iters_per_epoch": 400,
                     "patience": PATIENCE, "valid_only": True},
        "checkpoint": "atomic trainable-only; no frozen Vicuna or optimizer/scaler",
        "trials": TRIALS,
        "note": "No all-layer output_hidden_states; only four [batch,hidden] vectors retained by hooks. No TEST.",
    }
    base.core.LOG_DIR.mkdir(parents=True, exist_ok=True)
    (base.core.LOG_DIR / "latevote4_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[LATEVOTE4 PLAN] configs={base.core.CONFIG_DIR}", flush=True)
    print(f"[LATEVOTE4 LOGS] {base.core.LOG_DIR}", flush=True)
    print(f"[LATEVOTE4 OUTPUT] {base.core.OUT_DIR}", flush=True)
    for trial in TRIALS:
        print("[LATEVOTE4 TRIAL]", json.dumps(trial, sort_keys=True), flush=True)


def worker(name, patience, screen_epochs):
    configure()
    if name not in TRIAL_BY_NAME:
        raise ValueError("Unknown trial: " + name)
    trial = TRIAL_BY_NAME[name]
    cfg_path = base.core.CONFIG_DIR / (name + ".yaml")
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Run --plan first: {cfg_path}")
    base.core.patch_runner(screen_epochs=screen_epochs,
                           mf_lr_ratio=MF_LR_RATIO, patience=patience)
    base.core.install_compact_atomic_best_checkpoint()
    base.combo.install_combo_model(base.combo.NAMES["C6_KD_BOTH"])
    base._patch_runtime_classes(trial)
    install_late_logits(trial)
    cfg = yaml.safe_load(cfg_path.read_text())
    if cfg["model"]["freeze_rec"] is not False or cfg["run"]["seed"] != SEED:
        raise RuntimeError("MF must be trainable and seed must remain 42")
    sys.argv = ["train_collm_mf_din.py", "--cfg-path", str(cfg_path)]
    runpy.run_path(str(ROOT / "train_collm_mf_din.py"), run_name="__main__")


def _refresh_summary():
    script = ROOT / "cora_all_summary.py"
    if script.is_file():
        subprocess.run([sys.executable, str(script), "--sort", "auc"],
                       cwd=ROOT, check=False)


def run_selected(names, patience, screen_epochs):
    configure()
    if not base.core.CONFIG_DIR.is_dir():
        raise RuntimeError("Run python cora_latevote4.py --plan first")
    for name in names:
        log = base.core.LOG_DIR / (name + ".log")
        if log.exists():
            raise RuntimeError(f"Refusing to overwrite existing log: {log}")
    for name in names:
        log = base.core.LOG_DIR / (name + ".log")
        command = [sys.executable, "-u", str(ROOT / "cora_latevote4.py"),
                   "--_worker", name, "--patience", str(patience),
                   "--screen-epochs", str(screen_epochs)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
        print(f"[LATEVOTE4 START] {name}; serial GPU0; log={log}", flush=True)
        with log.open("x", encoding="utf-8") as handle:
            proc = subprocess.Popen(command, cwd=ROOT, env=env,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8",
                                    errors="replace", bufsize=1)
            for line in proc.stdout:
                print(line, end="", flush=True)
                handle.write(line)
            code = proc.wait()
        _refresh_summary()
        if code != 0:
            raise SystemExit(f"[LATEVOTE4 STOP ON FAILURE] {name} exit={code}; see {log}")
    print("[LATEVOTE4 DONE] Valid-only experiments completed.", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--only", default="")
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--screen-epochs", type=int, default=0)
    p.add_argument("--cpu-test", action="store_true")
    p.add_argument("--summary", action="store_true")
    p.add_argument("--_worker", default="", help=argparse.SUPPRESS)
    args = p.parse_args()
    os.chdir(ROOT)
    if args.cpu_test:
        _cpu_test()
        return
    if args.summary:
        _refresh_summary()
        return
    if args._worker:
        worker(args._worker, args.patience, args.screen_epochs)
        return
    if args.patience < 1 or args.screen_epochs < 0:
        p.error("patience >=1 and screen-epochs >=0 required")
    if not args.plan and not args.run:
        p.error("Specify --plan or --run")
    names = [x.strip() for x in args.only.split(",") if x.strip()] if args.only else ORDER[:]
    if len(names) != len(set(names)) or any(x not in TRIAL_BY_NAME for x in names):
        p.error("Invalid or duplicate --only IDs")
    if args.plan:
        prepare()
        print("[LATEVOTE4 PLAN ONLY] no training started", flush=True)
        return
    configure()
    if not base.core.CONFIG_DIR.is_dir():
        prepare()
    run_selected(names, args.patience, args.screen_epochs)


if __name__ == "__main__":
    main()
