#!/usr/bin/env python3
"""Inference-only late-layer diagnostic for D4_TOPK_RES025 / C6.

Loads the existing trainable-only best checkpoint, performs one VALID pass,
and captures Yes/No token logits from decoder layers 29/30/31/32.
No optimizer, backward, training, checkpoint writing, or TEST split.
"""
import argparse
import csv
import json
import os
import runpy
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import cora_rankgran7 as base
import cora_struct10 as gran

ROOT = Path(__file__).resolve().parent
TRIAL = "D4_TOPK_RES025"
EXPECTED_PROTOCOL = "CORA_STRUCT10_20261101"
# 0-based decoder indices: the four consecutive final blocks 29, 30, 31, 32.
DIAG_LAYERS = (28, 29, 30, 31)
LAYER_NAMES = [f"L{i + 1}" for i in DIAG_LAYERS]


def _resolve_checkpoint(explicit=None):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path, EXPECTED_PROTOCOL
    summary = ROOT / "out" / "cora_all_valid_best.csv"
    if not summary.is_file():
        raise FileNotFoundError(
            f"Missing {summary}. First run: python cora_all_summary.py --sort auc")
    with summary.open("r", encoding="utf-8-sig", newline="") as fp:
        rows = list(csv.DictReader(fp))
    matches = [r for r in rows if r.get("trial") == TRIAL]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {TRIAL} row in {summary}, found {len(matches)}")
    row = matches[0]
    if row.get("status") != "COMPLETE" or row.get("ckpt") != "FOUND":
        raise RuntimeError(
            f"{TRIAL} is not a completed run with a present checkpoint: "
            f"status={row.get('status')} ckpt={row.get('ckpt')}. "
            "Inspect the global summary before proceeding.")
    protocol = row.get("protocol", "")
    if protocol != EXPECTED_PROTOCOL:
        raise RuntimeError(
            f"Expected {TRIAL} protocol {EXPECTED_PROTOCOL}, summary points to {protocol!r}. "
            "Do not silently load a different run.")
    folder = ROOT / "output" / protocol / TRIAL
    candidates = sorted(folder.rglob("checkpoint_best.pth"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one checkpoint_best.pth under {folder}, found {len(candidates)}")
    path = candidates[0]
    if not zipfile.is_zipfile(path):
        raise RuntimeError(f"Checkpoint is not a valid PyTorch ZIP archive: {path}")
    return path, protocol


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().reshape(-1)
    return np.asarray(value).reshape(-1)


def _answer_offset(model, samples):
    answers = [model.pos_ans[0] if int(y) == 1 else model.neg_ans[0]
               for y in samples["label"]]
    tokens = model.llama_tokenizer(
        answers, return_tensors="pt", padding="longest",
        truncation=True, max_length=model.max_txt_len,
        add_special_tokens=False)
    return int(tokens.input_ids.shape[-1]) + 1


def _capture_layer_pair(model, samples, fn, *args, **kwargs):
    """Capture target-position Yes/No logits using shared final norm and LM head."""
    causal_lm = model.llama_model_lora.base_model.model
    decoder = causal_lm.model
    if len(decoder.layers) != 32:
        raise RuntimeError(f"Expected 32 Vicuna layers, got {len(decoder.layers)}")
    if getattr(decoder, "gradient_checkpointing", False):
        raise RuntimeError("Layer hooks require gradient_checkpointing=False")
    offset = _answer_offset(model, samples)
    yes_id = int(model.llama_tokenizer(
        model.pos_ans[0], add_special_tokens=False).input_ids[0])
    no_id = int(model.llama_tokenizer(
        model.neg_ans[0], add_special_tokens=False).input_ids[0])
    token_ids = torch.tensor([yes_id, no_id], dtype=torch.long,
                             device=causal_lm.lm_head.weight.device)
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
        captured["final_pair"] = torch.stack(
            [logits[:, -offset, yes_id], logits[:, -offset, no_id]], dim=-1
        ).float()

    handles = [
        decoder.layers[idx].register_forward_hook(make_layer_hook(idx))
        for idx in DIAG_LAYERS
    ]
    handles.append(model.llama_model_lora.register_forward_hook(final_hook))
    try:
        result = fn(model, samples, *args, **kwargs)
    finally:
        for handle in handles:
            handle.remove()

    missing = [idx for idx in DIAG_LAYERS if idx not in captured["layers"]]
    if missing or "final_pair" not in captured:
        raise RuntimeError(
            f"Layer capture incomplete: missing={missing}, final_pair={'final_pair' in captured}")
    rows = []
    for idx in DIAG_LAYERS:
        hidden = captured["layers"][idx]
        normalized = decoder.norm(hidden)
        # Reuse the actual LM-head module so quantized/tied-weight behavior
        # exactly matches the official forward path.
        full_logits = causal_lm.lm_head(normalized)
        rows.append(full_logits.index_select(-1, token_ids).float())
    layer_pair = torch.stack(rows, dim=0)  # [4, batch, (Yes, No)]
    final_pair = captured["final_pair"]
    if not torch.isfinite(layer_pair).all() or not torch.isfinite(final_pair).all():
        raise RuntimeError("Nonfinite Yes/No logits in late-layer diagnostic")
    max_error = float((layer_pair[-1] - final_pair).detach().abs().max())
    if max_error > 0.05:
        raise RuntimeError(
            f"Layer-32 Yes/No reconstruction mismatch: max_abs={max_error:.6g}")
    return result, layer_pair, final_pair, max_error


def _rankdata(values):
    """Average ranks for ties; avoids requiring sklearn/scipy."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks_sorted = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks_sorted[start:end] = 0.5 * ((start + 1) + end)
        start = end
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def _auc(labels, scores):
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) /
                 (n_pos * n_neg))


def _spearman(a, b):
    ra, rb = _rankdata(a), _rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _compute_metrics(labels, users, scores, probabilities):
    labels = np.asarray(labels).astype(np.int64)
    users = np.asarray(users).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != 4:
        raise RuntimeError(f"Expected score shape [4,N], got {scores.shape}")
    if len(labels) != scores.shape[1] or len(users) != len(labels):
        raise RuntimeError("Prediction, label, and user counts do not match")
    if not np.isin(labels, [0, 1]).all():
        raise RuntimeError("Valid labels must be binary 0/1")
    preds = (scores >= 0).astype(np.int64)
    errors = preds != labels[None, :]
    layer_rows = []
    user_auc_counts = {}
    for i, name in enumerate(LAYER_NAMES):
        p, z = probabilities[i], scores[i]
        per_user = []
        computed, skipped = 0, 0
        for uid in np.unique(users):
            mask = users == uid
            y_u = labels[mask]
            if np.any(y_u == 1) and np.any(y_u == 0):
                per_user.append(_auc(y_u, z[mask]))
                computed += 1
            else:
                skipped += 1
        user_auc = float(np.mean(per_user)) if per_user else float("nan")
        user_auc_counts[name] = {"computed": computed, "skipped": skipped}
        q = np.quantile(p, [0.01, 0.05, 0.50, 0.95, 0.99])
        layer_rows.append({
            "layer": name,
            "decoder_layer": DIAG_LAYERS[i] + 1,
            "auc": _auc(labels, z),
            "uauc_user_both_classes": user_auc,
            "brier": float(np.mean((p - labels) ** 2)),
            "accuracy_at_zero": float(np.mean(preds[i] == labels)),
            "mean_probability": float(np.mean(p)),
            "std_probability": float(np.std(p)),
            "mean_score": float(np.mean(z)),
            "std_score": float(np.std(z)),
            "mean_probability_positive": float(np.mean(p[labels == 1])),
            "mean_probability_negative": float(np.mean(p[labels == 0])),
            "fraction_p_below_1e-4": float(np.mean(p < 1e-4)),
            "fraction_p_above_1_minus_1e-4": float(np.mean(p > 1 - 1e-4)),
            "p01": float(q[0]), "p05": float(q[1]), "p50": float(q[2]),
            "p95": float(q[3]), "p99": float(q[4]),
            "n": int(len(labels)),
        })

    pairwise = []
    for i in range(4):
        for j in range(i + 1, 4):
            both = errors[i] & errors[j]
            union = errors[i] | errors[j]
            pairwise.append({
                "layer_a": LAYER_NAMES[i],
                "layer_b": LAYER_NAMES[j],
                "spearman_score": _spearman(scores[i], scores[j]),
                "prediction_disagreement": float(np.mean(preds[i] != preds[j])),
                "both_wrong_rate": float(np.mean(both)),
                "error_jaccard": float(both.sum() / union.sum()) if union.any() else 1.0,
            })

    final_error = errors[-1]
    final_error_count = int(final_error.sum())
    rescue = []
    for i in range(3):
        corrected = final_error & ~errors[i]
        harmed = ~final_error & errors[i]
        rescue.append({
            "other_layer": LAYER_NAMES[i],
            "final_layer_error_count": final_error_count,
            "final_errors_corrected_by_other_layer": int(corrected.sum()),
            "final_error_rescue_rate": (
                float(corrected.sum() / final_error_count) if final_error_count else 0.0),
            "final_correct_but_other_wrong": int(harmed.sum()),
            "other_layer_harm_rate_on_final_correct": (
                float(harmed.sum() / max(1, int((~final_error).sum())))),
        })
    any_correct = final_error & (~errors[:3]).any(axis=0)
    return {
        "n_samples": int(len(labels)),
        "positive_count": int((labels == 1).sum()),
        "negative_count": int((labels == 0).sum()),
        "unique_users": int(len(np.unique(users))),
        "layer_metrics": layer_rows,
        "pairwise": pairwise,
        "final_layer_rescue": rescue,
        "any_earlier_layer_rescues_final_error_count": int(any_correct.sum()),
        "any_earlier_layer_rescue_rate": (
            float(any_correct.sum() / final_error_count) if final_error_count else 0.0),
        "uauc_user_counts": user_auc_counts,
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _extract_official_auc(eval_result):
    if isinstance(eval_result, dict):
        value = eval_result.get("agg_metrics")
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
    return None


def _write_outputs(capture, checkpoint, protocol, eval_result, tag):
    labels = np.concatenate(capture["labels"]).astype(np.int64)
    users = np.concatenate(capture["users"]).astype(np.int64)
    items = np.concatenate(capture["items"]).astype(np.int64)
    layer_pair = np.concatenate(capture["layer_pair"], axis=1).astype(np.float32)
    final_pair = np.concatenate(capture["final_pair"], axis=0).astype(np.float32)
    if layer_pair.shape[1] != len(labels):
        raise RuntimeError("Prediction count differs from labels")
    max_pair_error = float(np.max(np.abs(layer_pair[-1] - final_pair)))
    if max_pair_error > 0.05:
        raise RuntimeError(f"Layer-32 Yes/No mismatch after concatenation: {max_pair_error:.6g}")

    yes_scores = layer_pair[:, :, 0].astype(np.float64)
    no_scores = layer_pair[:, :, 1].astype(np.float64)
    margin_scores = yes_scores - no_scores
    yes_prob = 1.0 / (1.0 + np.exp(-np.clip(yes_scores, -80, 80)))
    margin_prob = 1.0 / (1.0 + np.exp(-np.clip(margin_scores, -80, 80)))
    by_mode = {
        "yes_raw_logit": _compute_metrics(labels, users, yes_scores, yes_prob),
        "yes_minus_no_logit": _compute_metrics(labels, users, margin_scores, margin_prob),
    }
    official_auc = _extract_official_auc(eval_result)
    deltas = {}
    for mode, result in by_mode.items():
        final_auc = result["layer_metrics"][-1]["auc"]
        deltas[mode] = abs(final_auc - official_auc) if official_auc is not None else None
    matched = [mode for mode, delta in deltas.items()
               if delta is not None and delta <= 0.005]
    selected_mode = min(matched, key=lambda mode: deltas[mode]) if matched else None

    metrics = {
        "n_samples": int(len(labels)),
        "unique_users": int(len(np.unique(users))),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": capture.get("checkpoint_epoch"),
        "protocol": protocol,
        "layers_one_based": [x + 1 for x in DIAG_LAYERS],
        "layer32_yes_no_max_abs_error": max_pair_error,
        "official_aggregate_auc": official_auc,
        "final_layer_auc_abs_delta_vs_official": deltas,
        "official_score_mode_match": selected_mode,
        "metrics_by_score_mode": by_mode,
        "official_eval_result": _json_safe(eval_result),
        "uauc_definition": (
            "Unweighted mean of per-user AUC for users with both positive and negative "
            "labels; users lacking either class are excluded."),
        "interpretation_guard": (
            "Use the score mode matching official Layer32 AUC within 0.005. "
            "If neither matches, resolve score semantics before interpreting intermediate layers."),
    }
    out_dir = ROOT / "out" / tag
    out_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        out_dir / "layer_logits_valid.npz",
        labels=labels, users=users, items=items,
        yes_logits=yes_scores, no_logits=no_scores,
        yes_minus_no_logits=margin_scores,
        final_yes_no_logits=final_pair)
    for mode, result in by_mode.items():
        with (out_dir / f"per_layer_metrics_{mode}.csv").open(
                "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(result["layer_metrics"][0].keys()))
            writer.writeheader()
            writer.writerows(result["layer_metrics"])
        with (out_dir / f"pairwise_metrics_{mode}.csv").open(
                "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(result["pairwise"][0].keys()))
            writer.writeheader()
            writer.writerows(result["pairwise"])
        with (out_dir / f"final_layer_rescue_{mode}.csv").open(
                "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(result["final_layer_rescue"][0].keys()))
            writer.writeheader()
            writer.writerows(result["final_layer_rescue"])

    with (out_dir / "sample_predictions.csv").open(
            "w", newline="", encoding="utf-8") as fp:
        fields = ["sample_index", "user_id", "item_id", "label"]
        for name in LAYER_NAMES:
            fields += [f"yes_logit_{name}", f"no_logit_{name}",
                       f"margin_{name}", f"p_yes_raw_{name}",
                       f"p_yes_pair_{name}", f"pred_margin_{name}",
                       f"error_margin_{name}"]
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        margin_preds = (margin_scores >= 0).astype(np.int64)
        for n in range(len(labels)):
            row = {"sample_index": n, "user_id": int(users[n]),
                   "item_id": int(items[n]), "label": int(labels[n])}
            for i, name in enumerate(LAYER_NAMES):
                row[f"yes_logit_{name}"] = float(yes_scores[i, n])
                row[f"no_logit_{name}"] = float(no_scores[i, n])
                row[f"margin_{name}"] = float(margin_scores[i, n])
                row[f"p_yes_raw_{name}"] = float(yes_prob[i, n])
                row[f"p_yes_pair_{name}"] = float(margin_prob[i, n])
                row[f"pred_margin_{name}"] = int(margin_preds[i, n])
                row[f"error_margin_{name}"] = int(margin_preds[i, n] != labels[n])
            writer.writerow(row)

    (out_dir / "diagnostics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8")
    print(f"[LAYERDIAG OUTPUT] {out_dir}", flush=True)
    for mode, result in by_mode.items():
        print(f"[LAYERDIAG MODE] {mode}", flush=True)
        for row in result["layer_metrics"]:
            print(
                f"{row['layer']:>4} AUC={row['auc']:.6f} "
                f"UAUC={row['uauc_user_both_classes']:.6f} "
                f"Brier={row['brier']:.6f} "
                f"mean_p={row['mean_probability']:.5f} "
                f"std_p={row['std_probability']:.5f} "
                f"sat={row['fraction_p_below_1e-4'] + row['fraction_p_above_1_minus_1e-4']:.4f}",
                flush=True)
        print("[LAYERDIAG PAIRWISE]", json.dumps(result["pairwise"]), flush=True)
        print("[LAYERDIAG RESCUE]", json.dumps(result["final_layer_rescue"]), flush=True)
        print(
            f"[LAYERDIAG ANY-RESCUE] "
            f"{result['any_earlier_layer_rescues_final_error_count']} samples; "
            f"rate among final errors={result['any_earlier_layer_rescue_rate']:.6f}",
            flush=True)
    print(f"[LAYERDIAG L32 YES_NO CHECK] max_abs_error={max_pair_error:.8g}", flush=True)
    print(f"[LAYERDIAG OFFICIAL AUC] {official_auc}; mode deltas={deltas}; "
          f"matched_mode={selected_mode}", flush=True)
    if selected_mode is None:
        print(
            "[LAYERDIAG WARNING] Neither Yes-only nor Yes-minus-No Layer32 AUC matches "
            "the official Valid AUC within 0.005. Treat all intermediate-layer AUCs as "
            "diagnostic only; inspect the official scoring expression.",
            flush=True)


def _install_capture():
    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx
    if getattr(MiniGPT4Rec_vx, "_late_layer_diag_installed", False):
        raise RuntimeError("Layer diagnostic hook already installed")
    original = MiniGPT4Rec_vx.generate_for_samples_v2
    capture = {
        "labels": [], "users": [], "items": [],
        "layer_pair": [], "final_pair": [],
    }

    def wrapped(self, samples, *args, **kwargs):
        result, layer_pair, final_pair, max_error = _capture_layer_pair(
            self, samples, original, *args, **kwargs)
        capture["labels"].append(_to_numpy(samples["label"]).astype(np.int64))
        capture["users"].append(_to_numpy(samples["UserID"]).astype(np.int64))
        capture["items"].append(_to_numpy(samples["TargetItemID"]).astype(np.int64))
        capture["layer_pair"].append(layer_pair.detach().float().cpu().numpy())
        capture["final_pair"].append(final_pair.detach().float().cpu().numpy())
        if len(capture["labels"]) == 1:
            print(
                f"[LAYERDIAG CAPTURE] first_batch={len(capture['labels'][0])} "
                f"layers={[x + 1 for x in DIAG_LAYERS]} "
                f"layer32_yes_no_max_error={max_error:.8g}", flush=True)
        return result

    MiniGPT4Rec_vx.generate_for_samples_v2 = wrapped
    MiniGPT4Rec_vx._late_layer_diag_installed = True
    return capture


def _load_compact_trainable_state(model, checkpoint):
    try:
        payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(str(checkpoint), map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise RuntimeError(
            "Expected trainable-only checkpoint payload with a 'model' state dictionary")
    state = payload["model"]
    named = dict(model.named_parameters())
    trainable = {name for name, p in named.items() if p.requires_grad}
    supplied = set(state)
    missing_trainable = trainable - supplied
    unexpected = supplied - set(model.state_dict())
    if missing_trainable:
        raise RuntimeError(
            "Checkpoint misses current trainable parameters: " +
            repr(sorted(missing_trainable)[:20]))
    if unexpected:
        raise RuntimeError(
            "Checkpoint has unexpected keys: " + repr(sorted(unexpected)[:20]))
    incompatible = model.load_state_dict(state, strict=False)
    bad_missing = set(incompatible.missing_keys) & trainable
    if bad_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Compact checkpoint load mismatch: missing_trainable={sorted(bad_missing)[:10]} "
            f"unexpected={incompatible.unexpected_keys[:10]}")
    print(
        f"[LAYERDIAG CKPT LOADED] {checkpoint} epoch={payload.get('epoch')} "
        f"loaded_tensors={len(state)} trainable_parameters={len(trainable)}",
        flush=True)
    return payload


def _install_eval_only_runner(checkpoint, protocol, capture, tag):
    from minigpt4.runners.runner_base import RunnerBase
    from minigpt4.runners.runner_base_rec import RecRunnerBase

    def eval_only(self):
        model = self.unwrap_dist_model(self.model)
        payload = _load_compact_trainable_state(model, checkpoint)
        capture["checkpoint_epoch"] = payload.get("epoch")

        # The original runner's initialization path normally selects the
        # v2 dispatcher before calling valid_step(). This inference-only
        # runner replaces train(), so that setup must be reproduced here.
        if not hasattr(model, "set_mode"):
            raise RuntimeError(
                "MiniGPT4Rec_vx has no set_mode(); inspect the local model API "
                "before running this diagnostic.")
        if not hasattr(model, "generate_for_samples_v2"):
            raise RuntimeError(
                "MiniGPT4Rec_vx has no generate_for_samples_v2(); "
                "the D4 checkpoint requires the v2 inference path.")
        if not getattr(model, "pos_ans", None) or not getattr(model, "neg_ans", None):
            raise RuntimeError("Answer tokens are not initialized; expected ans_type=v2.")
        if model.pos_ans[0] != "Yes" or model.neg_ans[0] != "No":
            raise RuntimeError(
                f"Expected D4 ans_type=v2 (Yes/No), got "
                f"pos_ans={model.pos_ans!r}, neg_ans={model.neg_ans!r}.")
        model.set_mode("v2")
        if getattr(model, "run_mode_", None) != "v2":
            raise RuntimeError(
                f"Failed to set inference dispatcher to v2: "
                f"run_mode_={getattr(model, 'run_mode_', None)!r}")
        print("[LAYERDIAG MODE] explicitly selected v2; "
              "valid_step -> generate_for_samples -> generate_for_samples_v2",
              flush=True)

        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        self.model.eval()
        print(
            f"[LAYERDIAG EVAL START] split=valid no_grad=True "
            f"layers={[x + 1 for x in DIAG_LAYERS]}",
            flush=True)
        with torch.no_grad():
            result = self.eval_epoch("valid", 0, skip_reload=True)
        if not capture["labels"]:
            raise RuntimeError(
                "No layer predictions captured. Expected valid_step() to call "
                "generate_for_samples() -> generate_for_samples_v2(); inspect "
                "the local rec_base_task.py/model dispatcher before proceeding.")
        _write_outputs(capture, checkpoint, protocol, result, tag)
        print("[LAYERDIAG DONE] inference-only; no optimizer step or checkpoint save", flush=True)
        return result

    # Patch RunnerBase and every already-imported subclass so a custom
    # registered runner cannot accidentally enter the original train loop.
    pending = [RunnerBase]
    seen = set()
    while pending:
        cls = pending.pop()
        if cls in seen:
            continue
        seen.add(cls)
        cls.train = eval_only
        pending.extend(cls.__subclasses__())
    if RecRunnerBase not in seen:
        RecRunnerBase.train = eval_only


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="",
                        help="Optional exact checkpoint path; default resolves D4 from global summary")
    parser.add_argument("--tag", default="",
                        help="Output folder name under out/; default uses current timestamp")
    parser.add_argument("--run", action="store_true",
                        help="Run one inference-only Valid evaluation")
    args = parser.parse_args()
    os.chdir(ROOT)
    if not args.run:
        parser.error("Specify --run; this script never trains")
    checkpoint, protocol = _resolve_checkpoint(args.checkpoint)
    if protocol != EXPECTED_PROTOCOL:
        raise RuntimeError(f"Unexpected protocol {protocol}")
    config_path = ROOT / "local_configs" / protocol / f"{TRIAL}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing {config_path}; run 'python cora_struct10.py --plan' first")
    cfg = yaml.safe_load(config_path.read_text())
    if (cfg["run"]["seed"] != 42 or cfg["model"]["freeze_rec"] is not False or
            cfg["run"].get("test_splits") or cfg["run"].get("resume_ckpt_path") or
            cfg["run"].get("valid_splits") != ["valid"]):
        raise RuntimeError("D4 config must remain seed42, MF-trainable, Valid-only, no resume")

    # Install exactly the same D4 runtime routing used during training.
    gran.configure()
    base._make_group_inputs = gran.group_inputs
    base._configure_initialized_model = gran.configure_initialized_model
    base._patch_runtime_classes(gran.TRIAL_BY_NAME[TRIAL])
    capture = _install_capture()

    # The existing training entry point builds the official model/task/runner.
    # Replace only runner.train(): load D4's compact best weights and run Valid.
    tag = args.tag or ("LATE_LAYER_DIAG_D4_C6_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    _install_eval_only_runner(checkpoint, protocol, capture, tag)
    sys.argv = ["train_collm_mf_din.py", "--cfg-path", str(config_path)]
    print(f"[LAYERDIAG PLAN] checkpoint={checkpoint}", flush=True)
    print(f"[LAYERDIAG PLAN] config={config_path}", flush=True)
    print(f"[LAYERDIAG PLAN] split=valid layers={[x + 1 for x in DIAG_LAYERS]}", flush=True)
    print("[LAYERDIAG PLAN] training/backward/optimizer/checkpoint-save/test all disabled", flush=True)
    runpy.run_path(str(ROOT / "train_collm_mf_din.py"), run_name="__main__")


if __name__ == "__main__":
    main()
