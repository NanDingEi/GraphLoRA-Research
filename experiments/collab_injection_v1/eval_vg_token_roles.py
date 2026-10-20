#!/usr/bin/env python3
"""P0-P4: exact prompt-token role interventions on trained VG1, Valid only.

Uses the SAME checkpoint and seeded validation batches. ID placeholder roles
(user, history, target) are annotated by get_ids_order/segment positions,
NOT by item-ID equality. This preserves the distinction when an item occurs
in both the history and the candidate position.

A sixth P0_REPEAT pass must reproduce P0 exactly, including logits, FP32
logits, and L31 attention activations. No training or checkpoint changes.
"""
import argparse
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import torch
import train_collm_mf_din as entry
from experiments.collab_injection_v1.eval_vg_l31_fp32_head import fp32_yes_from_decoder
from minigpt4.common.config import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--factor", type=float, default=50.0)
    ap.add_argument("--result", required=True)
    ap.add_argument("--expected-baseline", type=float, default=0.7296752596287361)
    ap.add_argument("--auc-tolerance", type=float, default=0.00002)
    args = ap.parse_args()

    if not 0 < args.factor <= 100:
        raise ValueError("--factor must be positive and <= 100")
    conditions = (
        ("P0_OFF", None),
        ("P1_USER", 0),
        ("P2_HISTORY", 1),
        ("P3_TARGET", 2),
        ("P4_ALL", 3),
        ("P0_REPEAT", None),
    )

    path = Path(args.config).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    out = Path(args.result).expanduser().resolve()
    for p in (path, checkpoint):
        if not p.is_file():
            raise FileNotFoundError(p)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = Config(Namespace(cfg_path=str(path), options=None))
    if cfg.model_cfg.collab_mode != "target_v_gate_candidate":
        raise ValueError("Use VG1_CANDIDATE config only")
    cfg.model_cfg.ckpt = str(checkpoint)
    cfg.run_cfg.resume_ckpt_path = None
    cfg.run_cfg.distributed = False
    cfg.run_cfg.evaluate = False
    cfg.run_cfg.output_dir = str(out.parent / "eval_artifacts")

    entry.init_distributed_mode(cfg.run_cfg)
    entry.setup_seeds(cfg)
    entry.setup_logger()
    task = entry.tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    try:
        data_path = cfg.datasets_cfg.movie_ood.path
    except Exception:
        data_path = cfg.datasets_cfg.amazon_ood.path
    tables = [pd.read_pickle(data_path + s + "_ood2.pkl") for s in ("train","valid","test")]
    cfg.model_cfg.rec_config.user_num = int(max(x.uid.max() for x in tables) + 1)
    cfg.model_cfg.rec_config.item_num = int(max(x.iid.max() for x in tables) + 1)
    del tables

    model = task.build_model(cfg)
    gate_modules = [m for m in model.my_linears
                    if getattr(m, "target_v_gate", None) is not None]
    if len(gate_modules) != 1 or gate_modules[0].module_tag != "v":
        raise RuntimeError("Expected exactly one L31 V gate")
    v = gate_modules[0]
    runner = entry.get_runner_class(cfg)(
        cfg=cfg, job_id=entry.now(), task=task, model=model, datasets=datasets
    )
    runner.set_model_mode(cfg.run_cfg.mode)
    model = runner.unwrap_dist_model(runner.model)
    if model.run_mode_ != "v2":
        raise RuntimeError("Expected Yes/No v2 mode")

    active = {"role": None, "gates": [], "corrections": [],
              "scores": None, "attention": [], "fp32": [],
              "offset": None, "targeted": 0}
    original_forward = v.target_v_gate.forward
    original_gen = model.generate_for_samples
    original_gcn = model.compute_feedback_gcn
    original_attention = model.llama_model.model.layers[31].self_attn
    original_decoder = model.llama_model.model.layers[31]
    final_norm = model.llama_model.model.norm
    lm_head = model.llama_model.lm_head
    yes_token = model.llama_tokenizer(
        model.pos_ans[0], add_special_tokens=False).input_ids[0]

    def gc_with_roles(*a, **kw):
        fb = original_gcn(*a, **kw)
        if "token_roles" not in fb:
            raise RuntimeError("Sync new model code: no token_roles in GCN feedback")
        v.target_token_roles = fb["token_roles"]
        return fb

    model.compute_feedback_gcn = gc_with_roles

    def activation_hook(label, output):
        if active["scores"] is None:
            return
        x = output[0] if isinstance(output, (tuple, list)) else output
        if x.ndim != 3:
            raise RuntimeError(f"Unexpected {label} activation shape {x.shape}")
        pos = active["offset"]
        if pos is None or not -x.size(1) <= pos < 0:
            raise RuntimeError(f"Invalid decision position {pos}")
        h = x[:, pos, :]
        if label == "attention":
            active["attention"].append(h.detach().float().cpu())
        else:
            active["fp32"].append(
                fp32_yes_from_decoder(h, final_norm, lm_head, yes_token))

    attn_hook = original_attention.register_forward_hook(
        lambda m, inputs, out: activation_hook("attention", out))
    decoder_hook = original_decoder.register_forward_hook(
        lambda m, inputs, out: activation_hook("decoder", out))

    def measured_gate(context):
        delta = original_forward(context)
        valid = v.local_node_ids != -1
        if delta.size(0) != int(valid.sum().item()):
            raise RuntimeError("V gate rows do not match valid injected nodes")
        roles = v.target_token_roles[valid]
        if roles.shape[0] != delta.shape[0]:
            raise RuntimeError("VG1 role coordinates misaligned with gate rows")
        wanted = active["role"]
        selection = torch.zeros_like(roles, dtype=torch.bool) if wanted is None else (
            torch.ones_like(roles, dtype=torch.bool) if wanted == 3
            else (roles == wanted)
        )
        if wanted is not None and wanted != 3 and not bool(selection.any()):
            raise RuntimeError(f"Zero tokens of requested role {wanted} in batch")
        factor = args.factor
        selected_delta = delta * selection.to(delta.dtype).unsqueeze(-1) * factor
        z = v.gcn_output[v.local_node_ids[valid]].detach().float()
        correction = z * (v.gnn_scale * v.warmup_factor) * selected_delta.detach().float()
        active["gates"].append(delta.detach().float().cpu())
        active["corrections"].append(correction.cpu())
        active["targeted"] += int(selection.sum().item())
        return selected_delta

    def capture_gen(samples, *a, **kw):
        model.llama_tokenizer.padding_side = "right"
        texts = [model.pos_ans[0] if int(t) else model.neg_ans[0]
                 for t in samples["label"]]
        answer_tokens = model.llama_tokenizer(
            texts, return_tensors="pt", padding="longest", truncation=True,
            max_length=model.max_txt_len, add_special_tokens=False)
        active["offset"] = -(answer_tokens.input_ids.shape[-1] + 1)
        result = original_gen(samples, *a, **kw)
        scores = active["scores"]
        if scores is not None:
            score = result.get("logits")
            if not torch.is_tensor(score):
                raise RuntimeError("Expected logits tensor from Yes/No evaluation")
            scores["score"].extend(score.detach().float().flatten().cpu().tolist())
            for name, key in (("user","UserID"),("target","TargetItemID"),("label","label")):
                scores[name].extend(samples[key].detach().flatten().cpu().tolist())
        return result

    v.target_v_gate.forward = measured_gate
    model.generate_for_samples = capture_gen

    # The first evaluation prints a random prompt example. Every subsequent
    # pass must reset these one-time flags as well as the RNG so that the
    # random template trajectory matches exactly.
    first_prompt = {
        "has_print_prompt": model.has_print_prompt,
        "has_pri_decode": model.has_pri_decode,
        "prompt_list_p": model.prompt_list_p,
    }
    anchor = None
    order = None
    baseline_score = None
    baseline_fp32 = None
    baseline_attention = None
    for mode, role in conditions:
        entry.setup_seeds(cfg)
        for name, value in first_prompt.items():
            setattr(model, name, value)
        active["role"] = role
        active["gates"] = []
        active["corrections"] = []
        active["attention"] = []
        active["fp32"] = []
        active["targeted"] = 0
        active["scores"] = {"user":[],"target":[],"label":[],"score":[]}
        print(f"[TOKEN-P] BEGIN {mode} role={role} factor={args.factor}", flush=True)
        result = runner.eval_epoch(split_name="valid", cur_epoch=0, skip_reload=True)
        if result is None:
            raise RuntimeError("Evaluation returned no metrics")
        captures = active["scores"]
        if len(captures["score"]) != 5200:
            raise RuntimeError(f"Expected 5200 predictions, got {len(captures['score'])}")
        identity = tuple(tuple(captures[k]) for k in ("user","target","label"))
        if order is None:
            order = identity
        elif identity != order:
            raise RuntimeError("Sample order changed between interventions")
        score = np.asarray(captures["score"], dtype=np.float64)
        if not np.isfinite(score).all():
            raise RuntimeError("Non-finite prediction score")
        gates = torch.cat(active["gates"]).float()
        corrections = torch.cat(active["corrections"]).float()
        if not gates.numel() or not corrections.numel():
            raise RuntimeError("L31 V gate was never invoked")
        uauc = result.get("uauc")
        attention = torch.cat(active["attention"]).float()
        fp32 = torch.cat(active["fp32"]).float().flatten().numpy()
        if len(fp32) != 5200 or attention.shape[0] != 5200:
            raise RuntimeError("Missing per-sample attention or FP32 head samples")
        row = {
            "mode": mode, "factor": 0.0 if role is None else args.factor,
            "target_role": role, "n_targeted_nodes": active["targeted"],
            "auc": float(result["agg_metrics"]),
            "uauc": float(uauc[0]) if isinstance(uauc, (tuple,list)) else None,
            "loss": float(result["loss"]),
            "num_gated_nodes": int(gates.shape[0]),
            "raw_gate_mean_abs": float(gates.abs().mean()),
            "raw_gate_std": float(gates.std()),
            "raw_gate_max_abs": float(gates.abs().max()),
            "actual_correction_mean_abs": float(corrections.abs().mean()),
            "actual_correction_max_abs": float(corrections.abs().max()),
        }
        if mode == "P0_OFF":
            anchor = float(result["agg_metrics"])
            baseline_score = score.copy()
            baseline_fp32 = fp32.copy()
            baseline_attention = attention.clone()
            if abs(anchor - args.expected_baseline) > args.auc_tolerance:
                out.write_text(json.dumps([row], indent=2, ensure_ascii=False), encoding="utf-8")
                raise RuntimeError(
                    f"VG1 gate-off baseline {anchor:.9f} does not match "
                    f"S2 {args.expected_baseline:.9f}"
                )
        if baseline_score is not None:
            diff = np.abs(score - baseline_score)
            fp32_diff = np.abs(fp32 - baseline_fp32)
            attn_diff = (attention - baseline_attention).abs()
            row["mean_abs_score_delta_vs_P0"] = float(diff.mean())
            row["changed_score_frac_gt_1e-5"] = float(np.mean(diff > 1e-5))
            row["mean_abs_fp32_yes_delta_vs_P0"] = float(fp32_diff.mean())
            row["mean_abs_attention_delta_vs_P0"] = float(attn_diff.mean())
        records.append(row)
        print("[TOKEN-P-RESULT] " + json.dumps(row, ensure_ascii=False), flush=True)
        out.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        if mode == "P0_REPEAT" and (
            row["mean_abs_score_delta_vs_P0"] != 0.0 or
            row["mean_abs_fp32_yes_delta_vs_P0"] != 0.0 or
            row["mean_abs_attention_delta_vs_P0"] != 0.0
        ):
            raise RuntimeError("P0 repeated control diverges: matched prompt/RNG failed")
        active["scores"] = None
    attn_hook.remove()
    decoder_hook.remove()
    model.compute_feedback_gcn = original_gcn
    model.generate_for_samples = original_gen
    v.target_v_gate.forward = original_forward
    print("[TOKEN-P-COMPLETE] P0–P4 plus repeat control, valid only", flush=True)


if __name__ == "__main__":
    main()
