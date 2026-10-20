#!/usr/bin/env python3
"""No-training paired VG1 diagnostic with FP32 final RMSNorm + Yes-token head.

On the SAME validation samples, restore Python/NumPy/CPU/CUDA RNG before:
    off-A, off-B, learned x1, learned x50
Then compare LoRA-B/V projection at injected ID tokens, L31 attention
at the answer-decision position, L31 decoder hidden at that position,
and the Yes/No logit. off-B vs off-A is a same-input noise control.

Only the first --diagnostic-batches batches get four paired forwards.
All 5200 valid samples are scored with the normal trained factor x1
exactly once for validation AUC/UAUC. No train/test runs, no updates.
"""
import argparse
import copy
import json
import os
import random
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import torch
import train_collm_mf_din as entry
from minigpt4.common.config import Config


def rng_snapshot():
    return (
        random.getstate(),
        copy.deepcopy(np.random.get_state()),
        torch.get_rng_state().clone(),
        [s.clone() for s in torch.cuda.get_rng_state_all()],
    )


def rng_restore(s):
    random.setstate(s[0])
    np.random.set_state(s[1])
    torch.set_rng_state(s[2])
    torch.cuda.set_rng_state_all(s[3])


def rng_equal(a, b):
    if a[0] != b[0] or not torch.equal(a[2], b[2]):
        return False
    if len(a[3]) != len(b[3]) or any(
            not torch.equal(x, y) for x, y in zip(a[3], b[3])):
        return False
    return all(
        np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y
        for x, y in zip(a[1], b[1])
    )


def prompt_state_snapshot(model):
    """Preserve stateful first-prompt bookkeeping across paired forwards.

    recprompt_wrap_v2 uses an extra random.choice the FIRST time it prints a
    prompt example. Resetting RNG alone is insufficient because the flag
    changes between the off-A and off-B passes.
    """
    return {
        "has_print_prompt": bool(model.has_print_prompt),
        "has_pri_decode": bool(model.has_pri_decode),
        "prompt_list_p": model.prompt_list_p,
    }


def prompt_state_restore(model, state):
    for name, value in state.items():
        setattr(model, name, value)


def rng_differences(a, b):
    """Identify stream divergence if a future model change breaks pairing."""
    names = []
    if a[0] != b[0]:
        names.append("python_random")
    if not all(
        np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y
        for x, y in zip(a[1], b[1])
    ):
        names.append("numpy")
    if not torch.equal(a[2], b[2]):
        names.append("torch_cpu")
    if len(a[3]) != len(b[3]):
        names.append("cuda_device_count")
    else:
        names.extend(f"torch_cuda_{i}" for i, (x, y) in
                     enumerate(zip(a[3], b[3])) if not torch.equal(x, y))
    return names


def fp32_yes_from_decoder(decision_hidden, final_norm, lm_head, yes_token_id):
    """Recompute ONLY the last RMSNorm and Yes-token projection in FP32.

    The earlier transformer activations remain their original inference dtype.
    Disable autocast explicitly or torch will cast the FP32 matmul back to FP16.
    This is a post-hoc sensitivity probe, NOT full-model FP32 inference.
    """
    if decision_hidden.ndim != 2:
        raise RuntimeError("Expected final-layer decision state [batch, hidden]")
    eps = getattr(final_norm, "variance_epsilon", None)
    if eps is None:
        eps = getattr(final_norm, "eps", None)
    if eps is None:
        raise RuntimeError("Cannot identify final RMSNorm epsilon")
    with torch.autocast(device_type=decision_hidden.device.type, enabled=False):
        x = decision_hidden.detach().float()
        x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
        x = x * final_norm.weight.detach().float()
        row = lm_head.weight[yes_token_id].detach().float()
        return (x * row.unsqueeze(0)).sum(dim=-1, keepdim=True).cpu()


def diff_stat(current, base):
    if current.shape != base.shape:
        raise RuntimeError(
            f"Hook shapes changed between paired forwards: {current.shape}, {base.shape}"
        )
    x, y = current.float(), base.float()
    diff = (x - y).abs()
    norm = torch.linalg.vector_norm(x - y, dim=-1)
    denom = torch.linalg.vector_norm(y, dim=-1).clamp_min(1e-12)
    return {
        "elements": int(diff.numel()),
        "mean_abs": float(diff.mean()),
        "max_abs": float(diff.max()),
        "nonzero_fraction": float((diff > 0).float().mean()),
        "vectors": int(norm.numel()),
        "mean_relative_l2": float((norm / denom).mean()),
    }


class Aggregate:
    def __init__(self):
        self.totals = {}
        self.weight = {}
        self.max_abs = {}

    def add(self, name, stat):
        count = stat["elements"]
        vectors = stat["vectors"]
        if name not in self.totals:
            self.totals[name] = {
                "sum_abs": 0., "sum_nonzero": 0., "sum_relative": 0.,
            }
            self.weight[name] = [0, 0]
            self.max_abs[name] = 0.
        d = self.totals[name]
        d["sum_abs"] += stat["mean_abs"] * count
        d["sum_nonzero"] += stat["nonzero_fraction"] * count
        d["sum_relative"] += stat["mean_relative_l2"] * vectors
        self.weight[name][0] += count
        self.weight[name][1] += vectors
        self.max_abs[name] = max(self.max_abs[name], stat["max_abs"])

    def output(self):
        r = {}
        for name, d in self.totals.items():
            elems, vecs = self.weight[name]
            r[name] = {
                "num_elements": elems,
                "mean_abs": d["sum_abs"] / elems,
                "nonzero_fraction": d["sum_nonzero"] / elems,
                "mean_relative_l2": d["sum_relative"] / vecs,
                "max_abs": self.max_abs[name],
            }
        return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--diagnostic-batches", type=int, default=64)
    ap.add_argument("--intervention", type=float, default=50.0)
    ap.add_argument("--result", required=True)
    ap.add_argument("--expected-auc", type=float, default=0.7296748139587983)
    ap.add_argument("--auc-tolerance", type=float, default=0.00003)
    args = ap.parse_args()
    if args.diagnostic_batches < 1:
        raise ValueError("--diagnostic-batches must be positive")
    if not 1 < args.intervention <= 100:
        raise ValueError("--intervention must be in (1, 100]")

    config_path = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    result_path = Path(args.result).resolve()
    for path in (config_path, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = Config(Namespace(cfg_path=str(config_path), options=None))
    if cfg.model_cfg.collab_mode != "target_v_gate_candidate":
        raise RuntimeError("This paired diagnosis requires VG1_CANDIDATE")
    cfg.model_cfg.ckpt = str(checkpoint)
    cfg.run_cfg.resume_ckpt_path = None
    cfg.run_cfg.distributed = False
    cfg.run_cfg.evaluate = False
    cfg.run_cfg.output_dir = str(result_path.parent / "eval_artifacts")

    entry.init_distributed_mode(cfg.run_cfg)
    entry.setup_seeds(cfg)
    entry.setup_logger()
    task = entry.tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    try:
        data_path = cfg.datasets_cfg.movie_ood.path
    except Exception:
        data_path = cfg.datasets_cfg.amazon_ood.path
    # The historical runner derives user/item vocab from all split metadata.
    # Reading these files does NOT run inference or measure Test AUC.
    tables = [pd.read_pickle(data_path + s + "_ood2.pkl")
              for s in ("train", "valid", "test")]
    cfg.model_cfg.rec_config.user_num = int(max(x.uid.max() for x in tables) + 1)
    cfg.model_cfg.rec_config.item_num = int(max(x.iid.max() for x in tables) + 1)
    del tables
    model = task.build_model(cfg)
    runner = entry.get_runner_class(cfg)(
        cfg=cfg, job_id=entry.now(), task=task, model=model, datasets=datasets
    )
    runner.set_model_mode(cfg.run_cfg.mode)
    model = runner.unwrap_dist_model(runner.model)
    if model.run_mode_ != "v2":
        raise RuntimeError("Expected Yes/No recommendation path v2")
    vg = [m for m in model.my_linears
          if getattr(m, "target_v_gate", None) is not None]
    if len(vg) != 1 or vg[0].module_tag != "v":
        raise RuntimeError(f"Expected exactly one VG1 V module, got {len(vg)}")
    v = vg[0]
    decoder = model.llama_model.model.layers[31]
    attention = decoder.self_attn
    final_norm = model.llama_model.model.norm
    lm_head = model.llama_model.lm_head
    yes_token_id = model.llama_tokenizer(
        model.pos_ans[0], add_special_tokens=False).input_ids[0]
    if not isinstance(yes_token_id, int):
        raise RuntimeError("Expected integer Yes-token ID")

    active = {
        "factor": 1.0,
        "capture": None,
        "offset": None,
    }
    batch_counter = {"n": 0}
    collected = Aggregate()
    measured = {"diagnostic_batches": 0, "examples": 0, "gate_nodes": 0}
    orig_gate = v.target_v_gate.forward
    orig_generate = model.generate_for_samples

    def gate_forward(context):
        raw = orig_gate(context)
        factor = active["factor"]
        if active["capture"] is not None:
            valid = v.local_node_ids != -1
            if len(raw) != int(valid.sum().item()):
                raise RuntimeError("VG1 gate rows misaligned with injected ID tokens")
            z = v.gcn_output[v.local_node_ids[valid]].float().detach()
            correction = z * (v.gnn_scale * v.warmup_factor) * raw.detach().float() * factor
            c = active["capture"]
            c["gate_raw"] = raw.detach().float().cpu()
            c["rank_correction"] = correction.cpu()
        return raw * factor

    def v_hook(module, input_args, output):
        if active["capture"] is None:
            return
        valid = v.local_node_ids != -1
        b = v.batch_indices[valid]
        p = v.token_positions[valid]
        value = output[b, p, :]
        active["capture"]["v_id_after_lora_b"] = value.detach().float().cpu()

    def capture_hidden(name, output):
        if active["capture"] is None:
            return
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.ndim != 3:
            raise RuntimeError(f"Unexpected L31 activation shape in {name}: {output.shape}")
        offset = active["offset"]
        if offset is None or not -output.shape[1] <= offset < 0:
            raise RuntimeError(f"Invalid Yes/No decision offset {offset} for {output.shape}")
        # First answer logit reads the final PROMPT position, not target ID.
        tensor = output[:, offset, :]
        active["capture"][name] = tensor.detach().float().cpu()
        if name == "l31_decoder_at_decision":
            active["capture"]["yes_logit_fp32_head"] = fp32_yes_from_decoder(
                tensor, final_norm, lm_head, yes_token_id)

    def attention_hook(module, args_, output):
        capture_hidden("l31_attention_at_decision", output)

    def decoder_hook(module, args_, output):
        capture_hidden("l31_decoder_at_decision", output)

    def one_forward(samples, factor):
        active["factor"] = factor
        active["capture"] = {}
        # Match generate_for_samples_v2's answer-token construction exactly.
        model.llama_tokenizer.padding_side = "right"
        pos = model.pos_ans[0]
        neg = model.neg_ans[0]
        text = [pos if int(t) == 1 else neg for t in samples["label"]]
        y = model.llama_tokenizer(
            text, return_tensors="pt", padding="longest",
            truncation=True, max_length=model.max_txt_len,
            add_special_tokens=False,
        )
        active["offset"] = -(y.input_ids.shape[-1] + 1)
        if "is_test" in samples and samples["is_test"]:
            raise RuntimeError("Test split is prohibited in VG1 precision diagnosis")
        out = orig_generate(samples)
        if not isinstance(out, dict) or not torch.is_tensor(out.get("logits")):
            raise RuntimeError("Expected Yes/No logits tensor")
        active["capture"]["yes_logit"] = out["logits"].detach().float().reshape(-1, 1).cpu()
        expected = ("gate_raw", "rank_correction", "v_id_after_lora_b",
                    "l31_attention_at_decision", "l31_decoder_at_decision",
                    "yes_logit", "yes_logit_fp32_head")
        snap = active["capture"]
        if any(k not in snap for k in expected):
            raise RuntimeError("Missing activation capture: " + str(set(expected) - set(snap)))
        active["capture"] = None
        return out, snap

    def wrapper(samples, *args_, **kwargs):
        if args_ or kwargs:
            raise RuntimeError("Expected ordinary generate_for_samples(samples) evaluation")
        n = batch_counter["n"]
        batch_counter["n"] += 1
        if n >= args.diagnostic_batches:
            active["factor"] = 1.0
            active["capture"] = None
            return orig_generate(samples)

        rng = rng_snapshot()
        initial_prompt_state = prompt_state_snapshot(model)
        snaps = {}
        actual = None
        end_rng = None
        trained_end_rng = None
        trained_prompt_state = None

        # Restore BOTH RNG and first-prompt bookkeeping before every
        # intervention. Otherwise off-A may consume an extra random.choice
        # when it prints the first example, while off-B skips the print.
        # This changes the stochastic evaluation trajectory even with RNG
        # restored, so the two supposedly matched runs become incomparable.
        for name, factor in (("off_A", 0.0), ("off_B", 0.0),
                             ("trained", 1.0), ("amplified", args.intervention)):
            rng_restore(rng)
            prompt_state_restore(model, initial_prompt_state)
            with torch.no_grad():
                output, snap = one_forward(samples, factor)
            snaps[name] = snap
            current_rng = rng_snapshot()
            if end_rng is None:
                end_rng = current_rng
            elif not rng_equal(current_rng, end_rng):
                differences = rng_differences(current_rng, end_rng)
                raise RuntimeError(
                    "RNG consumption differs despite restoring model "
                    f"prompt flags; streams={differences}, batch={n}. "
                    "Do not treat unmatched forwards as paired evidence."
                )
            if name == "trained":
                actual = output
                trained_end_rng = current_rng
                trained_prompt_state = prompt_state_snapshot(model)
        # Match the ONE trained VG1 forward that the validation runner
        # logically requested, rather than retaining off-A or amplified
        # side-effects. This preserves the following batch's prompt RNG.
        rng_restore(trained_end_rng)
        prompt_state_restore(model, trained_prompt_state)
        if actual is None:
            raise RuntimeError("Trained VG1 forward was never executed")

        for name, a, b in (
            ("noise_off_repeat", "off_B", "off_A"),
            ("trained_vs_off", "trained", "off_A"),
            ("amplified_vs_off", "amplified", "off_A"),
            ("amplified_vs_trained", "amplified", "trained"),
        ):
            for stage in ("v_id_after_lora_b", "l31_attention_at_decision",
                          "l31_decoder_at_decision", "yes_logit",
                          "yes_logit_fp32_head"):
                stat = diff_stat(snaps[a][stage], snaps[b][stage])
                collected.add(f"{name}/{stage}", stat)

        # Confirm that the FP32 reconstruction uses the same Yes-token and
        # decision position as the original GraphLoRA v2 evaluation.
        for name in ("off_A", "off_B", "trained", "amplified"):
            reference = snaps[name]["yes_logit"]
            recomputed = snaps[name]["yes_logit_fp32_head"]
            mismatch = diff_stat(recomputed, reference)
            collected.add(f"fp32_head_vs_original/{name}", mismatch)
            if mismatch["mean_abs"] > 0.05:
                raise RuntimeError(
                    f"FP32 head reconstruction differs too much from original "
                    f"Yes logit ({mismatch['mean_abs']:.6g}); check head/token alignment"
                )

        # A direct measurement of how much is being *proposed* at the
        # LoRA-A+GNN rank bottleneck, before LoRA-B and FP16 attenuation.
        for name in ("trained", "amplified"):
            corr = snaps[name]["rank_correction"]
            vals = {
                "elements": corr.numel(), "mean_abs": float(corr.abs().mean()),
                "max_abs": float(corr.abs().max()),
                "nonzero_fraction": float((corr != 0).float().mean()),
                "vectors": len(corr),
                "mean_relative_l2": 0.0,
            }
            collected.add(f"pre_lora_b_{name}/rank_correction", vals)

        measured["diagnostic_batches"] += 1
        measured["examples"] += len(snaps["trained"]["yes_logit"])
        measured["gate_nodes"] += len(snaps["trained"]["gate_raw"])
        if measured["diagnostic_batches"] in (1, 8, 32, 64, 128):
            print("[VG-PAIRED] completed", measured["diagnostic_batches"],
                  "matched batches", flush=True)

        # End state has already been restored from the single trained pass.
        active["factor"] = 1.0
        active["capture"] = None
        return actual

    v.target_v_gate.forward = gate_forward
    handles = [
        v.register_forward_hook(v_hook),
        attention.register_forward_hook(attention_hook),
        decoder.register_forward_hook(decoder_hook),
    ]
    model.generate_for_samples = wrapper

    entry.setup_seeds(cfg)
    print("[VG-PAIRED] Starting full Valid Small; first",
          args.diagnostic_batches, "batches get four paired forwards", flush=True)
    try:
        result = runner.eval_epoch(
            split_name="valid", cur_epoch=0, skip_reload=True)
    finally:
        model.generate_for_samples = orig_generate
        v.target_v_gate.forward = orig_gate
        for handle in handles:
            handle.remove()

    if measured["diagnostic_batches"] != args.diagnostic_batches:
        raise RuntimeError("Validation had fewer batches than diagnostic budget")
    auc = float(result["agg_metrics"])
    uauc = result.get("uauc")
    report = {
        "setup": {
            "method": "FP32 last RMSNorm and Yes-token output only; prior LLM stays original precision",
            "yes_token_id": yes_token_id,
            "checkpoint": str(checkpoint),
            "config": str(config_path),
            "seed": cfg.run_cfg.seed,
            "diagnostic_batches": args.diagnostic_batches,
            "intervention": args.intervention,
            "validation_samples": 5200,
            "paired_examples": measured["examples"],
            "paired_id_tokens": measured["gate_nodes"],
        },
        "full_valid": {
            "auc": auc,
            "uauc": float(uauc[0]) if isinstance(uauc, (tuple, list)) else None,
            "expected_auc": args.expected_auc,
            "abs_auc_delta": abs(auc - args.expected_auc),
        },
        "paired": collected.output(),
    }
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print("[VG-PAIRED-RESULT] " + json.dumps(
        {"full_valid": report["full_valid"], "paired": report["paired"]},
        ensure_ascii=False), flush=True)
    if abs(auc - args.expected_auc) > args.auc_tolerance:
        raise RuntimeError(
            f"Full-validation AUC changed under diagnosis: {auc:.9f} "
            f"vs {args.expected_auc:.9f}. Investigate RNG/state interference"
        )
    print("[VG-PAIRED-COMPLETE] no training, no checkpoint changes, Valid only",
          flush=True)


if __name__ == "__main__":
    main()
