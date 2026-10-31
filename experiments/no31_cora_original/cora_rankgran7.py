#!/usr/bin/env python3
"""Seven serial CoRA/MF experiments: finish C8, then variable-rank and
layer-wise multi-granularity collaborative information.

The original CoRA source files are not edited. Runtime patches preserve the
official dynamic query generator and Q/K/V/O PLora path. All trials use the
C6 loss recipe (CTR=.005 + reliability-gated KD=.01), seed=42, VALID only,
GPU0, patience=10, and trainable-parameter-only atomic best checkpoints.
"""
import argparse
import json
import math
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

import cora_combo8 as combo

core = combo.core
ROOT = Path(__file__).resolve().parent
PROTOCOL = "CORA_RANKGRAN7_20261101"
SEED = 42
CORA_LR = 3e-4
MF_LR_RATIO = 0.1
STEPS_PER_EPOCH = 400
ALPHA_OVER_R = 2.0
LAYER_GROUPS = [(0, 7), (8, 15), (16, 23), (24, 31)]

TRIALS = [
    dict(name="P1_RANK_46810", mode="rank", ranks=[4, 6, 8, 10]),
    dict(name="P2_RANK_8121620", mode="rank", ranks=[8, 12, 16, 20]),
    dict(name="G1_CLUSTER_K32", mode="cluster_k32"),
    dict(name="G2_COARSE_FINE", mode="coarse_fine"),
    dict(name="G3_CLUSTER_RESIDUAL", mode="cluster_residual"),
    dict(name="G4_MULTI_TOPK", mode="multi_topk"),
]
TRIAL_BY_NAME = {t["name"]: t for t in TRIALS}
ALL_ORDER = ["C8_MAIN_HALF"] + [t["name"] for t in TRIALS]
RANK_RE = re.compile(r"layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$")


def configure_core():
    core.RUNS = [(t["name"], False, .005, .010, 0., 0.) for t in TRIALS]
    core.PROTOCOL = PROTOCOL
    core.CONFIG_DIR = ROOT / "local_configs" / PROTOCOL
    core.LOG_DIR = ROOT / "logs" / PROTOCOL
    core.OUT_DIR = ROOT / "output" / PROTOCOL
    combo.configure_protocol(PROTOCOL)


def _kmeans_cosine(table, k, seed=42, iterations=15):
    """Deterministic spherical k-means on the pretrained MF table (CPU only)."""
    x = F.normalize(table.detach().float().cpu()[1:], dim=-1)
    n, dim = x.shape
    if k < 1 or k >= n:
        raise ValueError(f"K={k} must be in [1,{n-1}] for MF table with {n+1} rows")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    first = int(torch.randint(n, (1,), generator=gen).item())
    centers = [x[first].clone()]
    closest = (1.0 - x @ centers[0]).clamp_min(0)
    for _ in range(1, k):
        probs = closest.square()
        if not torch.isfinite(probs).all() or float(probs.sum()) <= 1e-12:
            idx = int(torch.randint(n, (1,), generator=gen).item())
        else:
            idx = int(torch.multinomial(probs, 1, generator=gen).item())
        candidate = x[idx].clone()
        centers.append(candidate)
        closest = torch.minimum(closest, (1.0 - x @ candidate).clamp_min(0))
    centers = torch.stack(centers, dim=0)

    for _ in range(iterations):
        labels = (x @ centers.T).argmax(dim=1)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, labels, x)
        counts = torch.bincount(labels, minlength=k).to(x.dtype)
        updated = sums / counts.clamp_min(1).unsqueeze(-1)
        empty = counts == 0
        updated[empty] = centers[empty]
        centers = F.normalize(updated, dim=-1)

    labels = (x @ centers.T).argmax(dim=1)
    full = torch.full((table.shape[0],), -1, dtype=torch.long)
    full[1:] = labels
    members = [torch.nonzero(full == c, as_tuple=False).flatten() for c in range(k)]
    return {"labels": full, "members": members, "k": k}


def _precompute_neighbors(table, max_k=40, chunk_size=512):
    """Fixed neighbor IDs from pretrained MF cosine similarity; excludes ID 0/self."""
    x = F.normalize(table.detach().float().cpu(), dim=-1)
    n = x.shape[0]
    if max_k > n - 2:
        raise ValueError(f"TopK={max_k} requires at least {max_k+2} embedding rows; got {n}")
    result = torch.zeros((n, max_k), dtype=torch.long)
    for start in range(1, n, chunk_size):
        end = min(n, start + chunk_size)
        sim = x[start:end] @ x.T
        sim[:, 0] = -float("inf")
        rows = torch.arange(start, end)
        sim[torch.arange(end-start), rows] = -float("inf")
        result[start:end] = sim.topk(max_k, dim=1).indices
    return result


def _cluster_repr(table, ids, meta):
    labels = meta["labels"]
    ids_cpu = ids.detach().long().cpu().view(-1).tolist()
    outputs = []
    cache = {}
    for idx in ids_cpu:
        cluster = int(labels[idx])
        if cluster < 0:
            outputs.append(table[idx])
            continue
        if cluster not in cache:
            member_ids = meta["members"][cluster].to(table.device)
            cache[cluster] = table.index_select(0, member_ids).mean(dim=0)
        outputs.append(cache[cluster])
    return torch.stack(outputs, dim=0)


def _neighbor_repr(table, ids, neighbor_table, k, temperature=0.2):
    ids = ids.to(table.device).long().view(-1)
    neighbor_ids = neighbor_table[ids.detach().cpu(), :k].to(table.device)
    query = table.index_select(0, ids)
    neighbors = table[neighbor_ids]
    scores = F.cosine_similarity(query.unsqueeze(1), neighbors, dim=-1)
    weights = torch.softmax(scores / temperature, dim=-1)
    aggregate = (neighbors * weights.unsqueeze(-1)).sum(dim=1)
    return torch.where((ids == 0).unsqueeze(-1), query, aggregate)


def _cat_ui(user, item):
    return torch.cat([user, item], dim=-1)


def _make_group_inputs(generator, ui_embedding, ctx):
    rec = ctx["rec_encoder"]
    user_ids = ctx["user_ids"].long()
    item_ids = ctx["item_ids"].long()
    user_table = rec.user_embedding.weight
    item_table = rec.item_embedding.weight
    mode = generator._rankgran_mode

    user_ind = user_table[user_ids.to(user_table.device)]
    item_ind = item_table[item_ids.to(item_table.device)]
    meta = generator._rankgran_meta

    if mode == "cluster_k32":
        u = _cluster_repr(user_table, user_ids, meta["user_k32"])
        i = _cluster_repr(item_table, item_ids, meta["item_k32"])
        return [_cat_ui(u, i)]

    if mode == "coarse_fine":
        u32 = _cluster_repr(user_table, user_ids, meta["user_k32"])
        i32 = _cluster_repr(item_table, item_ids, meta["item_k32"])
        u128 = _cluster_repr(user_table, user_ids, meta["user_k128"])
        i128 = _cluster_repr(item_table, item_ids, meta["item_k128"])
        un = _neighbor_repr(user_table, user_ids, meta["user_neighbors"], 20)
        inn = _neighbor_repr(item_table, item_ids, meta["item_neighbors"], 20)
        return [_cat_ui(u32, i32), _cat_ui(u128, i128),
                _cat_ui(un, inn), ui_embedding]

    if mode == "cluster_residual":
        uc = _cluster_repr(user_table, user_ids, meta["user_k32"])
        ic = _cluster_repr(item_table, item_ids, meta["item_k32"])
        groups = []
        for alpha in (0.0, 1.0/3.0, 2.0/3.0):
            u = uc + alpha * (user_ind - uc)
            i = ic + alpha * (item_ind - ic)
            groups.append(_cat_ui(u, i))
        groups.append(ui_embedding)
        return groups

    if mode == "multi_topk":
        groups = []
        for k in (5, 10, 20, 40):
            u = _neighbor_repr(user_table, user_ids, meta["user_neighbors"], k)
            i = _neighbor_repr(item_table, item_ids, meta["item_neighbors"], k)
            groups.append(_cat_ui(u, i))
        return groups

    raise ValueError(f"Unknown granularity mode: {mode}")


def _patch_runtime_classes(trial):
    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx
    from minigpt4.models.ppeft.plora import Linear as PLoraLinear
    from minigpt4.models.ppeft.weight_generator import PerceptualWeightsGenerator
    from minigpt4.models.pllama import LlamaDecoderLayer

    if not getattr(PLoraLinear, "_rankgran_patched", False):
        original_linear_forward = PLoraLinear.forward

        def rank_slice_forward(self, x, w=None):
            if w is not None:
                rank = int(self.r[self.active_adapter])
                if w.shape[-1] < rank:
                    raise RuntimeError(
                        f"Dynamic weight rank {w.shape[-1]} is smaller than layer rank {rank}")
                if w.shape[-1] != rank:
                    w = w[..., :rank]
                    # CoRA normalizes the generated H x r matrix to unit norm.
                    # Re-normalize each truncated prefix so lower-rank layers
                    # do not receive a smaller signal solely because of slicing.
                    w = F.normalize(w.flatten(start_dim=1), p=2, dim=1).view_as(w)
            return original_linear_forward(self, x, w)

        PLoraLinear.forward = rank_slice_forward
        PLoraLinear._rankgran_patched = True

    if not getattr(LlamaDecoderLayer, "_rankgran_patched", False):
        original_decoder_forward = LlamaDecoderLayer.forward

        def routed_decoder_forward(self, *args, **kwargs):
            w = kwargs.get("w")
            if isinstance(w, (list, tuple)):
                layer_idx = getattr(self, "_rankgran_layer_idx", None)
                if layer_idx is None or len(w) != 32:
                    raise RuntimeError("Layer-specific dynamic weights require 32 indexed decoder layers")
                kwargs["w"] = w[layer_idx]
            return original_decoder_forward(self, *args, **kwargs)

        LlamaDecoderLayer.forward = routed_decoder_forward
        LlamaDecoderLayer._rankgran_patched = True

    if not getattr(PerceptualWeightsGenerator, "_rankgran_patched", False):
        original_generator_forward = PerceptualWeightsGenerator.forward

        def grouped_generator_forward(self, ui_embedding):
            ctx = getattr(self, "_rankgran_ctx", None)
            mode = getattr(self, "_rankgran_mode", "rank")
            if ctx is None or mode == "rank":
                return original_generator_forward(self, ui_embedding)

            groups = _make_group_inputs(self, ui_embedding, ctx)
            if len(groups) == 1:
                one = original_generator_forward(self, groups[0])
                return [one] * 32
            if len(groups) != 4:
                raise RuntimeError(f"Expected four layer-group inputs, got {len(groups)}")
            batch = ui_embedding.shape[0]
            packed = torch.cat(groups, dim=0)
            packed_weights = original_generator_forward(self, packed)
            if packed_weights.shape[0] != batch * 4:
                raise RuntimeError("Grouped query generator returned an unexpected batch shape")
            grouped = packed_weights.reshape(4, batch, *packed_weights.shape[1:])
            layer_weights = []
            for group_idx in range(4):
                layer_weights.extend([grouped[group_idx]] * 8)
            return layer_weights

        PerceptualWeightsGenerator.forward = grouped_generator_forward
        PerceptualWeightsGenerator._rankgran_patched = True

    if not getattr(MiniGPT4Rec_vx, "_rankgran_init_patched", False):
        original_init = MiniGPT4Rec_vx.__init__

        def rankgran_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            _configure_initialized_model(self, trial, PLoraLinear, LlamaDecoderLayer)

        MiniGPT4Rec_vx.__init__ = rankgran_init
        MiniGPT4Rec_vx._rankgran_init_patched = True

    # combo.install_combo_model() has already patched forward_v2 with C6 losses.
    # Wrap both train and generation entry points so every LLM pass sees the
    # same current user/item IDs and the same frozen cluster/neighbor indices.
    for method_name in ("forward_v2", "generate_for_samples_v2"):
        if getattr(MiniGPT4Rec_vx, "_rankgran_ctx_" + method_name, False):
            continue
        original_method = getattr(MiniGPT4Rec_vx, method_name)

        def make_context_method(fn):
            def with_rankgran_context(self, samples, *args, **kwargs):
                generator = self.llama_model_lora.base_model.weight_generator
                generator._rankgran_ctx = {
                    "rec_encoder": self.rec_encoder,
                    "user_ids": samples["UserID"],
                    "item_ids": samples["TargetItemID"],
                }
                try:
                    return fn(self, samples, *args, **kwargs)
                finally:
                    if hasattr(generator, "_rankgran_ctx"):
                        delattr(generator, "_rankgran_ctx")
            return with_rankgran_context

        setattr(MiniGPT4Rec_vx, method_name, make_context_method(original_method))
        setattr(MiniGPT4Rec_vx, "_rankgran_ctx_" + method_name, True)


def _configure_initialized_model(model, trial, PLoraLinear, LlamaDecoderLayer):
    generator = model.llama_model_lora.base_model.weight_generator
    causal_lm = model.llama_model_lora.base_model.model
    llama_model = causal_lm.model
    layers = llama_model.layers
    if len(layers) != 32:
        raise RuntimeError(f"Expected Vicuna-7B 32 layers, found {len(layers)}")
    if getattr(llama_model, "gradient_checkpointing", False):
        raise RuntimeError(
            "Layer-wise CoRA routing requires gradient_checkpointing=False; "
            "the official checkpoint wrapper drops the custom w argument")

    for idx, layer in enumerate(layers):
        if not isinstance(layer, LlamaDecoderLayer):
            raise RuntimeError(f"Unexpected decoder layer type at index {idx}: {type(layer)}")
        layer._rankgran_layer_idx = idx

    target_modules = []
    for name, module in model.llama_model_lora.named_modules():
        match = RANK_RE.search(name)
        if match and isinstance(module, PLoraLinear):
            target_modules.append((int(match.group(1)), match.group(2), module))
    if len(target_modules) != 32 * 4:
        sample_names = [name for name, module in model.llama_model_lora.named_modules()
                        if isinstance(module, PLoraLinear)][:12]
        raise RuntimeError(
            f"Expected 128 CoRA Q/K/V/O PLora layers, found {len(target_modules)}. "
            f"Sample PLora names: {sample_names}")

    if trial["mode"] == "rank":
        ranks = trial["ranks"]
        if len(ranks) != 4 or any(r <= 0 for r in ranks):
            raise ValueError(f"Invalid rank profile: {ranks}")
        max_rank = max(ranks)
        old_final = generator.final
        old_rank = int(generator.r)
        if old_rank != max_rank:
            new_final = torch.nn.Linear(
                old_final.in_features, int(generator.hidden_size) * max_rank,
                bias=old_final.bias is not None)
            new_final = new_final.to(device=old_final.weight.device,
                                     dtype=old_final.weight.dtype)
            torch.nn.init.kaiming_uniform_(new_final.weight, a=math.sqrt(5))
            if new_final.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(new_final.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.uniform_(new_final.bias, -bound, bound)
            keep = min(old_final.out_features, new_final.out_features)
            with torch.no_grad():
                new_final.weight[:keep].copy_(old_final.weight[:keep])
                if new_final.bias is not None and old_final.bias is not None:
                    new_final.bias[:keep].copy_(old_final.bias[:keep])
            generator.final = new_final
            generator.r = max_rank

        for layer_idx, projection, module in target_modules:
            rank = int(ranks[layer_idx // 8])
            adapter = module.active_adapter
            old_p = module.lora_P[adapter]
            new_p = torch.nn.Linear(rank, module.out_features, bias=False)
            new_p = new_p.to(device=old_p.weight.device, dtype=old_p.weight.dtype)
            torch.nn.init.zeros_(new_p.weight)
            module.lora_P[adapter] = new_p
            module.r[adapter] = rank
            module.lora_alpha[adapter] = int(round(ALPHA_OVER_R * rank))
            module.scaling[adapter] = ALPHA_OVER_R

        for cfg in model.llama_model_lora.peft_config.values():
            cfg.r = max_rank
            cfg.lora_alpha = int(round(ALPHA_OVER_R * max_rank))
        generator._rankgran_mode = "rank"
        print(f"[RANKGRAN RANK] profile={ranks} max_rank={max_rank} "
              f"alpha/r={ALPHA_OVER_R} targets={len(target_modules)}", flush=True)
        return

    # Granularity trials retain the official uniform r=16, alpha=32.
    for _, _, module in target_modules:
        adapter = module.active_adapter
        if int(module.r[adapter]) != 16 or abs(float(module.scaling[adapter]) - 2.0) > 1e-8:
            raise RuntimeError("Granularity trials require original CoRA r=16, alpha=32")
    generator._rankgran_mode = trial["mode"]
    rec = model.rec_encoder
    if not hasattr(rec, "user_embedding") or not hasattr(rec, "item_embedding"):
        raise RuntimeError("Multi-granularity experiments require the official MF encoder")
    print("[RANKGRAN KMEANS] building fixed assignments from pretrained MF tables on CPU", flush=True)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(min(old_threads, 8))
    try:
        generator._rankgran_meta = {
            "user_k32": _kmeans_cosine(rec.user_embedding.weight, 32, SEED + 1),
            "item_k32": _kmeans_cosine(rec.item_embedding.weight, 32, SEED + 2),
            "user_k128": _kmeans_cosine(rec.user_embedding.weight, 128, SEED + 3),
            "item_k128": _kmeans_cosine(rec.item_embedding.weight, 128, SEED + 4),
            "user_neighbors": _precompute_neighbors(rec.user_embedding.weight, 40),
            "item_neighbors": _precompute_neighbors(rec.item_embedding.weight, 40),
        }
    finally:
        torch.set_num_threads(old_threads)
    print(f"[RANKGRAN META] mode={trial['mode']} users={rec.user_embedding.num_embeddings} "
          f"items={rec.item_embedding.num_embeddings} K=32/128 TopK=5/10/20/40 "
          "assignments/neighbors are fixed from train-only pretrained MF; "
          "current trainable MF vectors remain differentiable", flush=True)


def _cpu_test():
    torch.manual_seed(42)
    table = torch.randn(65, 16, requires_grad=True)
    k32 = _kmeans_cosine(table, 8, seed=42, iterations=4)
    assert k32["labels"].shape[0] == table.shape[0]
    assert int(k32["labels"][0]) == -1
    ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    centers = _cluster_repr(table, ids, k32)
    centers.square().mean().backward()
    assert table.grad is not None and torch.isfinite(table.grad).all()

    with torch.no_grad():
        neighbors = _precompute_neighbors(table, max_k=5, chunk_size=16)
    table2 = table.detach().clone().requires_grad_(True)
    agg = _neighbor_repr(table2, ids, neighbors, 5)
    agg.sum().backward()
    assert table2.grad is not None and torch.isfinite(table2.grad).all()

    assert [x["ranks"] for x in TRIALS[:2]] == [[4, 6, 8, 10], [8, 12, 16, 20]]
    assert all(len(x) == 4 for x in (TRIALS[0]["ranks"], TRIALS[1]["ranks"]))
    print("[RANKGRAN CPU PASS] deterministic clustering, centroid gradients, "
          "neighbor aggregation gradients, and both rank profiles", flush=True)


def prepare():
    configure_core()
    core.check_sources()
    core.prepare()
    for trial in TRIALS:
        cfg_path = core.CONFIG_DIR / (trial["name"] + ".yaml")
        cfg = yaml.safe_load(cfg_path.read_text())
        if (cfg["run"]["seed"] != SEED or cfg["model"]["freeze_rec"] is not False or
                cfg["run"]["test_splits"] or cfg["run"].get("resume_ckpt_path") or
                cfg["run"]["iters_per_epoch"] != STEPS_PER_EPOCH or
                abs(float(cfg["run"]["init_lr"]) - CORA_LR) > 1e-12):
            raise RuntimeError(f"Experiment control mismatch in {cfg_path}")
    manifest = {
        "protocol": PROTOCOL,
        "trials": TRIALS,
        "seed": SEED,
        "gpu": "0",
        "loss_baseline": {
            "main": "original CoRA Yes-logit BCE",
            "ctr": 0.005,
            "kd": 0.010,
            "kd_gate": "LLM prediction agrees with label AND abs(p-0.5)>=0.20",
        },
        "optimizer": {"cora_lr": CORA_LR, "mf_lr": CORA_LR * MF_LR_RATIO},
        "training": {
            "batch_size": 2, "accum_grad_iters": 4,
            "iters_per_epoch": STEPS_PER_EPOCH,
            "valid_only": True, "patience": 10,
        },
        "rank": {
            "layers": "four consecutive groups of eight layers",
            "profiles": [[4, 6, 8, 10], [8, 12, 16, 20]],
            "alpha_over_rank": ALPHA_OVER_R,
            "implementation": "one shared max-rank generator output; each layer slices its prefix and has its own rank-specific P projection",
        },
        "granularity": {
            "G1": "all 32 layers use K=32 cluster centroids",
            "G2": "layers 0-7 K32; 8-15 K128; 16-23 TopK20; 24-31 individual MF",
            "G3": "K32 centroid-to-individual residual interpolation alpha=[0,1/3,2/3,1]",
            "G4": "layers 0-7 TopK5; 8-15 TopK10; 16-23 TopK20; 24-31 TopK40",
            "cluster_and_neighbor_indices": "computed from pretrained train-only MF tables; fixed during training",
            "centroids_and_neighbor_vectors": "computed from current trainable MF tables with gradients",
            "generator": "one shared official CoRA generator, called once for G1 and as a 4x-batched call for G2-G4",
        },
        "checkpoint": "atomic trainable-parameters-only; no frozen Vicuna or optimizer/scaler",
        "note": "No prompt modification; no test evaluation.",
    }
    core.LOG_DIR.mkdir(parents=True, exist_ok=True)
    (core.LOG_DIR / "rankgran7_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[RANKGRAN7 READY] protocol={PROTOCOL}", flush=True)
    for t in TRIALS:
        print("[RANKGRAN7 TRIAL] " + json.dumps(t, sort_keys=True), flush=True)
    print(f"[CONFIG] {core.CONFIG_DIR} [LOG] {core.LOG_DIR} [OUTPUT] {core.OUT_DIR}", flush=True)


def worker(name, patience, screen_epochs):
    if name not in TRIAL_BY_NAME:
        raise ValueError("Unknown rank/granularity trial: " + name)
    trial = TRIAL_BY_NAME[name]
    configure_core()
    cfg_path = core.CONFIG_DIR / (name + ".yaml")
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing {cfg_path}; run python cora_rankgran7.py --plan first")
    core.patch_runner(screen_epochs=screen_epochs,
                      mf_lr_ratio=MF_LR_RATIO, patience=patience)
    core.install_compact_atomic_best_checkpoint()
    combo.install_combo_model(combo.NAMES["C6_KD_BOTH"])
    _patch_runtime_classes(trial)
    cfg = yaml.safe_load(cfg_path.read_text())
    if cfg["model"]["freeze_rec"] is not False or cfg["run"]["seed"] != SEED:
        raise RuntimeError("MF must be trainable and seed must remain 42")
    sys.argv = ["train_collm_mf_din.py", "--cfg-path", str(cfg_path)]
    runpy.run_path(str(ROOT / "train_collm_mf_din.py"), run_name="__main__")


def _refresh_global_summary():
    script = ROOT / "cora_all_summary.py"
    if script.is_file():
        return subprocess.run(
            [sys.executable, str(script), "--sort", "auc"],
            cwd=ROOT, check=False)
    return None


def run_c8(patience, screen_epochs):
    log = ROOT / "logs" / PROTOCOL / "C8_MAIN_HALF.log"
    if log.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing C8 log: {log}. "
            "If it is incomplete, inspect and archive it explicitly before retrying.")
    command = [
        sys.executable, "-u", str(ROOT / "cora_combo8.py"),
        "--run", "--only", "C8_MAIN_HALF",
        "--patience", str(patience),
        "--screen-epochs", str(screen_epochs),
        "--protocol", PROTOCOL,
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    print(f"[RANKGRAN7 C8 START] fresh isolated attempt; log={log}", flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    _refresh_global_summary()


def run_rankgran_trials(names, patience, screen_epochs):
    for name in names:
        if name not in TRIAL_BY_NAME:
            raise ValueError("Unknown trial: " + name)
    for name in names:
        log_path = core.LOG_DIR / (name + ".log")
        if log_path.exists():
            raise RuntimeError(
                f"Existing log protects prior run: {log_path}. "
                "Do not overwrite it; use a new protocol for a clean attempt.")
        command = [
            sys.executable, "-u", str(ROOT / "cora_rankgran7.py"),
            "--_worker", name, "--patience", str(patience),
            "--screen-epochs", str(screen_epochs),
        ]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
        print(f"\n[RANKGRAN7 START] {name}; serial foreground GPU0; log={log_path}", flush=True)
        with log_path.open("x", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace", bufsize=1)
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
            code = process.wait()
        if code != 0:
            _refresh_global_summary()
            raise SystemExit(f"[RANKGRAN7 STOP ON FAILURE] {name}, exit={code}; see {log_path}")
        _refresh_global_summary()
    print("[RANKGRAN7 COMPLETE] requested trials completed; VALID-only; no TEST.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true",
                        help="Prepare isolated configs and manifests; no training")
    parser.add_argument("--run", action="store_true",
                        help="Run selected trials serially in foreground on GPU0")
    parser.add_argument("--only", default="",
                        help="Comma-separated subset; include C8_MAIN_HALF to rerun C8")
    parser.add_argument("--skip-c8", action="store_true",
                        help="Skip the C8 retry and run only the selected rank/granularity trials")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--screen-epochs", type=int, default=0)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--cpu-test", action="store_true")
    parser.add_argument("--_worker", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)

    if args.cpu_test:
        _cpu_test()
        return
    if args._worker:
        worker(args._worker, args.patience, args.screen_epochs)
        return
    if args.summary:
        _refresh_global_summary()
        return
    if args.patience < 1 or args.screen_epochs < 0:
        parser.error("--patience must be >=1 and --screen-epochs must be >=0")
    if not args.plan and not args.run:
        parser.error("Choose --plan or --run")

    if args.only:
        selected = [x.strip() for x in args.only.split(",") if x.strip()]
    else:
        selected = ALL_ORDER[:]
    if args.skip_c8:
        selected = [x for x in selected if x != "C8_MAIN_HALF"]
    if len(selected) != len(set(selected)):
        parser.error("--only contains duplicate experiment IDs")
    unknown = [x for x in selected if x != "C8_MAIN_HALF" and x not in TRIAL_BY_NAME]
    if unknown:
        parser.error("Unknown experiment IDs: " + ", ".join(unknown))

    if args.plan:
        # C8 config and rank/granularity configs share one isolated protocol.
        subprocess.run(
            [sys.executable, str(ROOT / "cora_combo8.py"), "--protocol", PROTOCOL],
            cwd=ROOT, check=True)
        prepare()
        print("[RANKGRAN7 PLAN ONLY] no GPU training started.", flush=True)
        print("Run: CUDA_VISIBLE_DEVICES=0 python -u cora_rankgran7.py --run --patience 10", flush=True)
        return

    if "C8_MAIN_HALF" in selected:
        run_c8(args.patience, args.screen_epochs)
    rank_names = [x for x in selected if x != "C8_MAIN_HALF"]
    if rank_names:
        prepare()
        run_rankgran_trials(rank_names, args.patience, args.screen_epochs)
    _refresh_global_summary()


if __name__ == "__main__":
    main()
