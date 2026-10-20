#!/usr/bin/env python3
"""No-training paired precision/signal propagation diagnosis for VG1 L31 V.

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
                    "yes_logit")
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
        snaps = {}
        actual = None
        end_rng = None
        # Identical samples and random states for every pass:
        # the x0-vs-x0_repeat control measures real execution variability.
        for name, factor in (("off_A", 0.0), ("off_B", 0.0),
                             ("trained", 1.0), ("amplified", args.intervention)):
            rng_restore(rng)
            with torch.no_grad():
                output, snap = one_forward(samples, factor)
            snaps[name] = snap
            current_rng = rng_snapshot()
            if end_rng is None:
                end_rng = current_rng
            elif not rng_equal(current_rng, end_rng):
                raise RuntimeError(
                    "RNG consumption differs between gate factors; paired "
                    "evaluation is no longer an apples-to-apples control"
                )
            if name == "trained":
                actual = output
        if actual is None:
            raise RuntimeError("Trained VG1 forward was never executed")

        for name, a, b in (
            ("noise_off_repeat", "off_B", "off_A"),
            ("trained_vs_off", "trained", "off_A"),
            ("amplified_vs_off", "amplified", "off_A"),
            ("amplified_vs_trained", "amplified", "trained"),
        ):
            for stage in ("v_id_after_lora_b", "l31_attention_at_decision",
                          "l31_decoder_at_decision", "yes_logit"):
                stat = diff_stat(snaps[a][stage], snaps[b][stage])
                collected.add(f"{name}/{stage}", stat)

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

        # The last pass uses the amplified factor. Restore the RNG that a
        # SINGLE trained pass would have consumed, so remaining validation
        # batches retain the historical random sequence.
        #
        # The four forwards have the same RNG consumption paths, so the
        # amplified pass's final RNG equals the single trained-pass final RNG.
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
