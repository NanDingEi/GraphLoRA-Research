#!/usr/bin/env python3
"""S2 recommendation: no-training late-layer Yes-logit voting on Valid Small.

ONE ordinary LLM forward per validation batch with hooks on decoder layers.
No checkpoint edits, no gradient steps, no Test AUC, no learned vote weights.
Use the same Yes row of the pre-trained LM head plus the FINAL RMSNorm for
each decoder layer; this is a logit-lens probe, not a separately trained exit.

Raw layer logits have different scales, so the predeclared soft votes average
unsupervised per-layer z-scores (fit without labels). The unmodified native
S2 L31 logit is also collected and verified against the historical Valid AUC.

The output .npz keeps example IDs, labels and all scores for mechanism work.
"""
import argparse
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

# Only GPU0; must be set before torch import.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import train_collm_mf_din as entry
from minigpt4.common.config import Config
from minigpt4.tasks.rec_base_task import uAUC_pandas

DEFAULT_AUC = 0.7296752596287361


def parse_layers(spec, count=32):
    layers = [int(x.strip()) for x in str(spec).split(",") if x.strip()]
    if not layers or len(layers) != len(set(layers)):
        raise ValueError("Layer indices must be nonempty and unique")
    if any(l < 0 or l >= count for l in layers):
        raise ValueError(f"Layer indices outside [0,{count-1}]: {layers}")
    if layers != sorted(layers) or layers[-1] != count - 1:
        raise ValueError(f"Use ascending layers ending at L{count-1}")
    return layers


def zscore_columns(scores):
    """Label-free calibration across the identical validation examples."""
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("Expected [samples, layer_count] score matrix")
    if not np.isfinite(x).all():
        raise ValueError("Scores contain NaN/Inf")
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    if (std < 1e-10).any():
        raise RuntimeError("At least one layer has a nearly constant score")
    return (x - mean) / std, mean.ravel(), std.ravel()


def predefined_votes(layer_scores, native_score, layers):
    """No weights optimized on labels; all formulas fixed before validation."""
    layers = list(layers)
    z, means, stds = zscore_columns(layer_scores)
    native_z, native_mu, native_sd = zscore_columns(
        np.asarray(native_score, dtype=np.float64).reshape(-1, 1))
    n = len(layers)
    # Deeper layers weighted increasingly, independent of observed AUC.
    depth = np.arange(1, n + 1, dtype=np.float64)
    depth /= depth.sum()
    result = {
        "vote_equal_all": z.mean(axis=1),
        "vote_linear_depth": z @ depth,
    }
    # Native S2 Yes logit, not L31 logit-lens, is the fixed anchor.
    # Intermediate experts exclude the last (L31) to avoid double voting.
    if n > 1:
        aux = z[:, :-1].mean(axis=1)
        baseline = native_z[:, 0]
        result["vote_s2_50_aux_50"] = 0.50 * baseline + 0.50 * aux
        result["vote_s2_75_aux_25"] = 0.75 * baseline + 0.25 * aux
    info = {
        "calibration": "unlabeled full-Valid per-layer mean/std",
        "layer_means": means.tolist(),
        "layer_stds": stds.tolist(),
        "native_mean": native_mu.item(),
        "native_std": native_sd.item(),
        "fixed_linear_weights": depth.tolist(),
        "no_label_fitting": True,
    }
    return result, info


def metric_of(y, user, score):
    y = np.asarray(y).reshape(-1).astype(np.int64)
    user = np.asarray(user).reshape(-1)
    score = np.asarray(score).reshape(-1).astype(np.float64)
    if y.shape != user.shape or y.shape != score.shape:
        raise ValueError("Metric sample alignment failure")
    if not np.isfinite(score).all():
        raise ValueError("NaN/Inf score in evaluation")
    auc = float(roc_auc_score(y, score))
    uauc, valid_users, _ = uAUC_pandas(user, score, y)
    return {"auc": auc, "uauc": float(uauc),
            "uauc_users": len(valid_users)}


def fp32_layer_yes(decision_hidden, norm, yes_head_weight):
    """Final RMSNorm and one Yes row, recomputed in fp32 only."""
    if decision_hidden.ndim != 2:
        raise RuntimeError(f"Bad decoder decision state {decision_hidden.shape}")
    eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", None))
    if eps is None:
        raise RuntimeError("LLaMA final RMSNorm epsilon unavailable")
    with torch.autocast(device_type=decision_hidden.device.type, enabled=False):
        hidden = decision_hidden.detach().float()
        hidden = hidden * torch.rsqrt(
            hidden.square().mean(dim=-1, keepdim=True) + float(eps))
        hidden = hidden * norm.weight.detach().float()
        return (hidden * yes_head_weight.unsqueeze(0)).sum(dim=-1).cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="tran_configs/collab_v1/e3/S2_L31_CONTROL.yaml")
    p.add_argument("--checkpoint", default=None,
                   help="S2 champion checkpoint; default=model.ckpt in config")
    p.add_argument("--layers", default="24,26,28,30,31")
    p.add_argument("--result", default="logs/collab_v1/S2_MULTILAYER_VOTE.json")
    p.add_argument("--expected-auc", type=float, default=DEFAULT_AUC)
    p.add_argument("--auc-tolerance", type=float, default=2e-5)
    p.add_argument("--expected-samples", type=int, default=5200)
    args = p.parse_args()

    cfg_path = Path(args.config).resolve()
    result_path = Path(args.result).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"{cfg_path}. First run python "
            "experiments/collab_injection_v1/prepare_e3_configs.py"
        )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = Config(Namespace(cfg_path=str(cfg_path), options=None))
    if str(cfg.model_cfg.collab_mode) != "legacy":
        raise RuntimeError("Use the S2_L31_CONTROL.yaml legacy injection config")
    if int(cfg.model_cfg.gcn_layer) != 31 or abs(float(cfg.model_cfg.gnn_scale) - 0.1) > 1e-10:
        raise RuntimeError("Use the historical S2 L31 gnn_scale=0.1")
    ckpt = Path(args.checkpoint or cfg.model_cfg.ckpt).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    cfg.model_cfg.ckpt = str(ckpt)
    cfg.run_cfg.distributed = False
    cfg.run_cfg.resume_ckpt_path = None
    cfg.run_cfg.evaluate = False
    cfg.run_cfg.output_dir = str(result_path.parent / "vote_eval_artifacts")

    entry.init_distributed_mode(cfg.run_cfg)
    entry.setup_seeds(cfg)
    entry.setup_logger()
    task = entry.tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    try:
        data_path = cfg.datasets_cfg.movie_ood.path
    except Exception:
        data_path = cfg.datasets_cfg.amazon_ood.path
    # Same vocabulary derivation as S2 evaluation. Test metadata is used
    # ONLY for the vocabulary upper bound; no Test predictions/metrics.
    frames = [pd.read_pickle(data_path + x + "_ood2.pkl")
              for x in ("train", "valid", "test")]
    cfg.model_cfg.rec_config.user_num = int(max(x.uid.max() for x in frames) + 1)
    cfg.model_cfg.rec_config.item_num = int(max(x.iid.max() for x in frames) + 1)
    del frames

    model = task.build_model(cfg)
    runner = entry.get_runner_class(cfg)(
        cfg=cfg, job_id=entry.now(), task=task, model=model, datasets=datasets)
    runner.set_model_mode(cfg.run_cfg.mode)
    model = runner.unwrap_dist_model(runner.model)
    model.eval()
    if model.run_mode_ != "v2":
        raise RuntimeError("Expected GraphLoRA Yes/No forward v2")
    llama = model.llama_model
    modules = llama.model.layers
    layers = parse_layers(args.layers, len(modules))
    final_norm, lm_head = llama.model.norm, llama.lm_head
    yes_id = model.llama_tokenizer(
        model.pos_ans[0], add_special_tokens=False).input_ids[0]
    yes_weight = lm_head.weight[yes_id].detach().float()

    active = {"enabled": False, "offset": None, "layers": {}}
    all_scores = {str(l): [] for l in layers}
    user_ids, item_ids, labels, native_logits = [], [], [], []
    batch_counter = 0
    hooks = []
    original_generate = model.generate_for_samples

    def layer_hook(l):
        def callback(module, inputs, output):
            if not active["enabled"]:
                return
            h = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(h) or h.ndim != 3:
                raise RuntimeError(f"L{l} decoder hook output invalid")
            offset = active["offset"]
            if offset is None or not -h.size(1) <= offset < 0:
                raise RuntimeError(f"L{l} invalid decision offset {offset}")
            if l in active["layers"]:
                raise RuntimeError(f"L{l} invoked more than once per batch")
            active["layers"][l] = fp32_layer_yes(
                h[:, offset, :], final_norm, yes_weight)
        return callback

    for layer in layers:
        hooks.append(modules[layer].register_forward_hook(layer_hook(layer)))

    def capture(samples, *extra_args, **extra_kwargs):
        nonlocal batch_counter
        if extra_args or extra_kwargs:
            raise RuntimeError("Expected standard generate_for_samples(samples) call")
        if bool(samples.get("is_test", False)):
            raise RuntimeError("No Test set evaluation in vote probe")
        if active["enabled"]:
            raise RuntimeError("Recursive capture forward")
        # Must match generate_for_samples_v2's answer-token layout exactly.
        model.llama_tokenizer.padding_side = "right"
        answers = [model.pos_ans[0] if int(x) == 1 else model.neg_ans[0]
                   for x in samples["label"]]
        tokenized = model.llama_tokenizer(
            answers, return_tensors="pt", padding="longest",
            truncation=True, max_length=model.max_txt_len,
            add_special_tokens=False)
        active["offset"] = -(tokenized.input_ids.shape[-1] + 1)
        active["layers"] = {}
        active["enabled"] = True
        try:
            with torch.no_grad():
                output = original_generate(samples)
        finally:
            active["enabled"] = False
        if set(active["layers"]) != set(layers):
            raise RuntimeError(
                f"Missing layer activations: {set(layers)-set(active['layers'])}"
            )
        score = output.get("logits")
        if not torch.is_tensor(score):
            raise RuntimeError("Expected native GraphLoRA Yes logit")
        batch_size = len(samples["label"])
        if score.numel() != batch_size:
            raise RuntimeError("Native score count differs from labels")
        for l in layers:
            values = active["layers"][l].reshape(-1).numpy()
            if len(values) != batch_size:
                raise RuntimeError(f"L{l} layer batch length changed")
            all_scores[str(l)].extend(values.tolist())
        native_logits.extend(score.detach().float().reshape(-1).cpu().tolist())
        user_ids.extend(samples["UserID"].detach().reshape(-1).cpu().tolist())
        item_ids.extend(samples["TargetItemID"].detach().reshape(-1).cpu().tolist())
        labels.extend(samples["label"].detach().reshape(-1).cpu().tolist())
        batch_counter += 1
        if batch_counter % 200 == 0:
            print(f"[VOTE-PROGRESS] batches={batch_counter} "
                  f"examples={len(labels)}", flush=True)
        return output

    model.generate_for_samples = capture
    try:
        # Uses unmodified task data ordering, validation preprocessing and
        # standard GraphLoRA metrics. Each sample has ONE full-model forward.
        entry.setup_seeds(cfg)
        reported = runner.eval_epoch(split_name="valid", cur_epoch=0,
                                     skip_reload=True)
    finally:
        model.generate_for_samples = original_generate
        for handle in hooks:
            handle.remove()

    count = len(labels)
    if count != args.expected_samples:
        raise RuntimeError(f"Expected {args.expected_samples} Valid examples; found {count}")
    native = np.asarray(native_logits, dtype=np.float64)
    layer_matrix = np.column_stack(
        [np.asarray(all_scores[str(l)], dtype=np.float64) for l in layers])
    y = np.asarray(labels, dtype=np.int64)
    u = np.asarray(user_ids, dtype=np.int64)
    items = np.asarray(item_ids, dtype=np.int64)
    if layer_matrix.shape != (count, len(layers)):
        raise RuntimeError("Layer matrix sample alignment failure")

    results = {"S2_native": metric_of(y, u, native)}
    if abs(results["S2_native"]["auc"] - args.expected_auc) > args.auc_tolerance:
        raise RuntimeError(
            "S2 native anchor mismatch; do not compare multi-layer votes. "
            f"Observed {results['S2_native']['auc']:.10f}, "
            f"historical {args.expected_auc:.10f}"
        )
    if abs(float(reported["agg_metrics"]) - results["S2_native"]["auc"]) > 1e-8:
        raise RuntimeError("Captured native AUC differs from runner AUC")
    for j, layer in enumerate(layers):
        results[f"L{layer}_fp32_lens"] = metric_of(y, u, layer_matrix[:, j])

    votes, calib = predefined_votes(layer_matrix, native, layers)
    for name, score in votes.items():
        results[name] = metric_of(y, u, score)

    for name, row in results.items():
        row["delta_auc_vs_S2"] = row["auc"] - results["S2_native"]["auc"]
        row["delta_uauc_vs_S2"] = row["uauc"] - results["S2_native"]["uauc"]
        print(f"[VOTE-RESULT] {name:24s} AUC={row['auc']:.9f} "
              f"UAUC={row['uauc']:.9f} "
              f"dAUC={row['delta_auc_vs_S2']:+.9f} "
              f"dUAUC={row['delta_uauc_vs_S2']:+.9f}", flush=True)

    # Compare native fp16 Yes score to the independently read L31 fp32 lens.
    last = layer_matrix[:, -1]
    pearson = float(np.corrcoef(last, native)[0, 1])
    report = {
        "status": "complete", "split": "valid", "n_samples": count,
        "n_batches": batch_counter, "checkpoint": str(ckpt),
        "source_config": str(cfg_path), "layers": layers,
        "native_yes_logit": "original S2 output without modifications",
        "layer_lens": "shared final RMSNorm + Yes LM-head row in fp32",
        "voting_calibration": calib,
        "l31_lens_vs_native_pearson": pearson,
        "l31_lens_vs_native_mean_abs": float(np.abs(last-native).mean()),
        "metrics": results,
    }
    npz = result_path.with_suffix(".npz")
    np.savez_compressed(npz, user=u, item=items, label=y,
                        s2_native=native, layers=np.asarray(layers),
                        layer_logits=layer_matrix, **votes)
    report["prediction_archive"] = str(npz)
    result_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"[VOTE-COMPLETE] no training; report={result_path} predictions={npz}",
          flush=True)


if __name__ == "__main__":
    main()
