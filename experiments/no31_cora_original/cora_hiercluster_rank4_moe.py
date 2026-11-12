#!/usr/bin/env python3
"""Hierarchical Group + Personalized CoRA-MoE (4 independent rank-4 experts).

Expert E0: ORIGINAL MF user/item vector -> shared Q-Former -> independent Rank4 head
Expert E1: train-cluster COARSE user/item centroids -> Q-Former -> independent Rank4
Expert E2: train-cluster MIDDLE user/item centroids -> Q-Former -> independent Rank4
Expert E3: train-cluster FINE user/item centroids -> Q-Former -> independent Rank4

Each expert supplies its OWN [B, 4096, 4] dynamic down matrix, followed by
its OWN [4096, 4] up projection for every Q/K/V/O layer. Exactly 4xRank4,
NO shared rank-16 generator or split of an existing rank-16 representation.

All centroids derive from train-seen entity IDs and PRETRAINED MF vectors
using nested spherical k-means, once at initialization (seed42); assignments
then stay fixed. Centroids are recomputed from CURRENT trainable MF embeddings
in forward, retaining gradients. Cold/padding entities use original MF vectors.

Controls: all-raw; flat group (one cluster level repeated for E1/E2/E3);
          nested hierarchical groups, uniform/pair/layer_pair routing.
No RR1, D4, neighbor TopK, C6 CTR/KD, data leakage from valid/test, or new
optimization method. Reuses the existing working Clean Rank4 training path:
GPU0, seed42, original CoRA LR, no epoch ceiling, Early Stop patience5.
Runtime patches are process-local; repository source files are not modified.

Cluster assignments are ALSO saved as an experiment-local metadata sidecar,
since the inherited compact trainable-only checkpoint excludes fixed buffers.
"""
import argparse
import json
import math
import os
import re
import runpy
import subprocess
import sys
import types
import weakref
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for d in (str(ROOT), str(HERE)):
    if d not in sys.path:
        sys.path.insert(0, d)
import cora_clean_rank4_moe6 as clean
import cora_joint8 as joint

SEED = 42
N_EXPERTS = 4
RANK = 4
USER_K = (8, 32, 64)
ITEM_K = (16, 64, 128)
TRIALS = (
    ("B0_RAW_UNIFORM", "raw", "uniform", 0.2),
    ("B1_FLAT_UNIFORM", "flat", "uniform", 0.2),
    ("H1_HIER_UNIFORM", "hier", "uniform", 0.2),
    ("H2_HIER_PAIR_L02", "hier", "pair", 0.2),
    ("H3_HIER_PAIR_L10", "hier", "pair", 1.0),
    ("H4_HIER_LAYERPAIR_L10", "hier", "layer_pair", 1.0),
)
KIND_CHOICES = ("raw", "flat", "hier")


def _train_only_entities(source_yaml):
    """Read ONLY train_ood2.pkl. No valid/test material is inspected."""
    import pandas as pd
    cfg = yaml.safe_load(Path(source_yaml).read_text(encoding="utf-8"))
    storage = next(iter(cfg["datasets"].values()))["build_info"]["storage"]
    p = Path(storage).expanduser()
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    source = p / "train_ood2.pkl"
    if not source.is_file():
        raise FileNotFoundError("Missing train-only cluster source: " + str(source))
    frame = pd.read_pickle(source)
    if not {"uid", "iid", "label"}.issubset(frame.columns):
        raise RuntimeError("Unexpected train_ood2.pkl columns: " + repr(list(frame.columns)))
    positive = frame.loc[frame["label"].astype(float) > 0.5]
    if len(positive) == 0:
        raise RuntimeError("Train split contains no positive interactions")
    users = {int(z) for z in positive["uid"].dropna() if int(z) > 0}
    items = {int(z) for z in positive["iid"].dropna() if int(z) > 0}
    for col in ("his", "sessionItems"):
        if col not in frame.columns:
            continue
        for row_id, seq in frame[col].items():
            if seq is None:
                continue
            try:
                if len(seq) == 0:
                    continue
            except (TypeError, ValueError):
                continue
            uid = int(frame.at[row_id, "uid"])
            if uid > 0:
                users.add(uid)
            for z in seq:
                try:
                    iid = int(z)
                except (ValueError, TypeError, OverflowError):
                    continue
                if iid > 0:
                    items.add(iid)
    return sorted(users), sorted(items), str(source)


def _spherical_kmeans(x, k, seed, iterations=18, min_size=1):
    """Deterministic spherical k-means with minimum viable child-group size."""
    x = F.normalize(x.detach().float().cpu(), dim=-1, eps=1e-12)
    n = x.shape[0]
    if k < 2 or min_size < 1 or n < k * min_size:
        raise ValueError(f"Cannot split {n} observed vectors into {k} groups "
                         f"with at least {min_size} members each")
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    first = int(torch.randint(n, (1,), generator=rng))
    centers = [x[first]]
    distance = (1 - x @ x[first]).clamp_min(0)
    for _ in range(1, k):
        prob = distance.square()
        idx = int(torch.multinomial(prob, 1, generator=rng)) if float(prob.sum()) > 1e-12 else int(torch.randint(n, (1,), generator=rng))
        centers.append(x[idx])
        distance = torch.minimum(distance, (1 - x @ x[idx]).clamp_min(0))
    centers = torch.stack(centers)
    labels = None
    for _ in range(iterations + 1):
        sim = x @ centers.T
        labels = sim.argmax(dim=1)
        counts = torch.bincount(labels, minlength=k)
        # For the NEXT hierarchy level to exist, guarantee each parent
        # has enough train-seen members. Repair underfull groups by relocating
        # a member from an oversized donor with minimal similarity penalty.
        for target in range(k):
            while int(counts[target]) < min_size:
                surplus = counts - min_size
                donor = int(surplus.argmax())
                if int(surplus[donor]) <= 0:
                    raise RuntimeError("Insufficient members to repair cluster")
                donor_rows = torch.where(labels == donor)[0]
                advantage = sim[donor_rows, target] - sim[donor_rows, donor]
                moved = donor_rows[advantage.argmax()]
                labels[moved] = target
                counts[donor] -= 1
                counts[target] += 1
        sums = torch.zeros_like(centers)
        sums.index_add_(0, labels, x)
        means = sums / counts.to(x.dtype)[:, None]
        centers = F.normalize(means, dim=-1, eps=1e-12)
    if int(torch.bincount(labels, minlength=k).min()) < min_size:
        raise RuntimeError("Spherical k-means violated minimum subgroup size")
    return labels


def _nested_hierarchy(table, train_ids, ks, seed):
    """True nested hierarchy: K0 -> 4 children each -> 2 children each."""
    ids = torch.tensor(sorted(set(train_ids)), dtype=torch.long)
    ids = ids[(ids > 0) & (ids < table.shape[0])]
    # Coarse groups need 4 middle children and each middle child needs 2
    # fine children. Enforce at least 16 observed entities per coarse group.
    if ids.numel() < ks[0] * 16:
        raise ValueError(f"Insufficient observed entities: {ids.numel()} for K={ks}")
    if ks[1] != 4 * ks[0] or ks[2] != 2 * ks[1]:
        raise ValueError("Hierarchy must be K,4K,8K")
    all_vec = table.detach().float().cpu()
    assignments = torch.full((3, table.shape[0]), -1, dtype=torch.long)
    root = _spherical_kmeans(
        all_vec.index_select(0, ids), ks[0], seed, min_size=4 * 4
    )
    assignments[0, ids] = root
    for level, children in ((1, 4), (2, 2)):
        nparents = ks[level - 1]
        for p in range(nparents):
            selected = ids[assignments[level-1, ids] == p]
            if selected.numel() < 2 * children:
                raise RuntimeError(
                    f"Unbalanced cluster tree: level={level} parent={p} "
                    f"members={selected.numel()} required={2 * children}"
                )
            local = _spherical_kmeans(
                all_vec.index_select(0, selected),
                children, seed + 1000*level + p,
                min_size=4 if level == 1 else 1,
            )
            assignments[level, selected] = p * children + local
        if (assignments[level, ids] < 0).any():
            raise RuntimeError("Incomplete nested cluster assignments")
    # Child assignments are guaranteed to nest within their parent.
    assert torch.equal(assignments[1, ids] // 4, assignments[0, ids])
    assert torch.equal(assignments[2, ids] // 2, assignments[1, ids])
    return assignments


class HierClusterBank(nn.Module):
    """Fixed train-only memberships, differentiable current-MF cluster means."""
    def __init__(self, u_labels, i_labels):
        super().__init__()
        if u_labels.shape[0] != 3 or i_labels.shape[0] != 3:
            raise ValueError("Need three hierarchy levels")
        self.register_buffer("user_labels", u_labels.clone().long())
        self.register_buffer("item_labels", i_labels.clone().long())
        self.register_buffer("user_seen", torch.where(u_labels[0] >= 0)[0].long())
        self.register_buffer("item_seen", torch.where(i_labels[0] >= 0)[0].long())
        self.user_k = tuple(int(x) for x in USER_K)
        self.item_k = tuple(int(x) for x in ITEM_K)

    @staticmethod
    def _lookup(table, labels, seen, keys, k):
        keys = keys.long().view(-1).to(table.device)
        labels = labels.to(table.device)
        seen = seen.to(table.device)
        if (keys < 0).any() or (keys >= table.shape[0]).any():
            raise RuntimeError("Invalid user/item ID")
        own = table.index_select(0, keys)
        batch_labels = labels.index_select(0, keys)
        group_labels = labels.index_select(0, seen)
        counts = torch.bincount(group_labels, minlength=k)
        if (counts == 0).any():
            raise RuntimeError("Empty group in differentiable aggregation")
        centroid = table.new_zeros((k, table.shape[-1]))
        centroid.index_add_(0, group_labels, table.index_select(0, seen))
        centroid = centroid / counts.to(dtype=table.dtype, device=table.device)[:, None]
        # Match the original MF norm scale so large groups are not unfairly
        # weakened merely because averaging reduces vector magnitude.
        scale = table.index_select(0, seen).norm(dim=-1).mean()
        centroid = F.normalize(centroid, dim=-1, eps=1e-10) * scale
        picked = centroid.index_select(0, batch_labels.clamp_min(0))
        return torch.where(batch_labels[:, None] >= 0, picked, own)

    def contexts(self, original_ui, rec, user_ids, item_ids, kind):
        if kind not in KIND_CHOICES:
            raise ValueError("Unknown cluster conditioning type")
        if original_ui.ndim != 2 or original_ui.shape[-1] % 2 != 0:
            raise RuntimeError("Require concatenated full user+item MF features")
        if kind == "raw":
            # Four independent Rank4 heads see identical raw features.
            return torch.stack([original_ui] * 4, dim=1)
        u_table = rec.user_embedding.weight
        i_table = rec.item_embedding.weight
        dim = u_table.shape[-1]
        if original_ui.shape[-1] != 2 * dim:
            raise RuntimeError("MF table / input dimension mismatch")
        ctx = [original_ui]  # E0 truly personalized; never averaged
        cache = {}
        levels = (1, 1, 1) if kind == "flat" else (0, 1, 2)
        for level in levels:
            if level not in cache:
                uv = self._lookup(u_table, self.user_labels[level], self.user_seen,
                                  user_ids, self.user_k[level])
                iv = self._lookup(i_table, self.item_labels[level], self.item_seen,
                                  item_ids, self.item_k[level])
                cache[level] = torch.cat([uv, iv], dim=-1)
            ctx.append(cache[level].to(original_ui.dtype))
        return torch.stack(ctx, dim=1)  # [B, E=4, full MF user+item]


def _generator_four_conditions(g, all_conditions):
    """Shared QFormer, four independently projected rank4 dynamic outputs."""
    if all_conditions.ndim != 3 or all_conditions.shape[1] != 4:
        raise RuntimeError("Expected [batch, four experts, complete MF condition]")
    b, e, width = all_conditions.shape
    if width != 2 * g.embedding_size:
        raise RuntimeError("Mismatched complete MF conditioning size")
    packed = all_conditions.reshape(b*e, 1, width)
    attention = torch.ones((b*e, 1), device=packed.device, dtype=torch.long)
    query = g.query_tokens.expand(b*e, -1, -1)
    states = g.qformer(
        query_embeds=query,
        encoder_hidden_states=packed,
        encoder_attention_mask=attention,
        use_cache=True,
        return_dict=True,
    ).last_hidden_state
    pooled = states.mean(dim=1).reshape(b, e, -1)
    dynamic = []
    for expert in range(e):
        # One independent [width, 4096*4] head PER expert, never rank16.
        a = F.normalize(g.final[expert](pooled[:, expert]), p=2, dim=-1, eps=1e-12)
        dynamic.append(a.reshape(b, g.hidden_size, RANK) / math.sqrt(e))
    result = torch.stack(dynamic, dim=1)
    if result.shape != (b, N_EXPERTS, g.hidden_size, RANK):
        raise RuntimeError("Four rank4 outputs malformed")
    return result


def _cluster_labels(model, args):
    rec = model.rec_encoder
    u_table = rec.user_embedding.weight
    i_table = rec.item_embedding.weight
    uids, iids, pickle_path = _train_only_entities(args.source_yaml)
    src = Path(pickle_path)
    if not all(0 < x < u_table.shape[0] for x in uids):
        raise RuntimeError("Training user IDs outside MF table")
    if not all(0 < x < i_table.shape[0] for x in iids):
        raise RuntimeError("Training item IDs outside MF table")
    cfg, _, _ = clean._paths(args.protocol)
    cache = cfg.parent / "train_only_cluster_assignments.pt"
    if cache.is_file():
        # Immutable assignments are needed to evaluate a saved trainable-only
        # best checkpoint without silently refitting to later MF parameters.
        saved = torch.load(str(cache), map_location="cpu")
        if saved.get("seed") != SEED or tuple(saved["user_k"]) != USER_K or tuple(saved["item_k"]) != ITEM_K:
            raise RuntimeError("Saved clusters do not match requested protocol")
        if saved.get("source_pickle") != str(src):
            raise RuntimeError("Saved cluster source mismatch")
        ulabels, ilabels = saved["user_labels"], saved["item_labels"]
    else:
        ulabels = _nested_hierarchy(u_table, uids, USER_K, SEED)
        ilabels = _nested_hierarchy(i_table, iids, ITEM_K, SEED + 100)
        to_save = {
            "seed": SEED,
            "source_pickle": str(src),
            "user_k": USER_K,
            "item_k": ITEM_K,
            "user_seen_count": len(uids),
            "item_seen_count": len(iids),
            "user_labels": ulabels,
            "item_labels": ilabels,
            "note": "Nested train-only fixed membership; current MF means recomputed per forward.",
        }
        # The saved sidecar is not model/optimizer weights.
        torch.save(to_save, str(cache))
        print(f"[HCLUSTER SAVED] sidecar={cache}", flush=True)
    bank = HierClusterBank(ulabels, ilabels).to(device=u_table.device)
    if bank.user_labels.shape[1] != u_table.shape[0] or bank.item_labels.shape[1] != i_table.shape[0]:
        raise RuntimeError("Saved cluster table incompatible with MF dimensions")
    print(f"[HCLUSTER FIT] kind={args.condition_kind} userK={USER_K} itemK={ITEM_K} "
          f"train_seen_users={len(uids)} train_seen_items={len(iids)} "
          "nested=True fixed_ids=True current_MF_centroids=True cold_fallback=raw "
          "no_valid_or_test_cluster_fit=True", flush=True)
    return bank


def patch_cluster_conditioning(args):
    """Call AFTER clean._patch_model(args), BEFORE model construction."""
    from minigpt4.models.minigpt4rec_vx import MiniGPT4Rec_vx
    old_init = MiniGPT4Rec_vx.__init__
    old_train = MiniGPT4Rec_vx.forward_v2
    old_eval = MiniGPT4Rec_vx.generate_for_samples_v2

    def cluster_init(self, *a, **kw):
        old_init(self, *a, **kw)
        generator = self.llama_model_lora.base_model.weight_generator
        if len(generator.final) != 4 or any(h.out_features != generator.hidden_size * RANK for h in generator.final):
            raise RuntimeError("Four Rank4 heads missing from Clean Rank4 bootstrap")
        if args.condition_kind == "raw":
            self.cora_hier_cluster_bank = None
        else:
            self.cora_hier_cluster_bank = _cluster_labels(self, args)
        generator._hcluster_model = weakref.ref(self)
        generator._hcluster_samples = None

        def grouped_forward(g, ui_embedding):
            owner = g._hcluster_model()
            if owner is None:
                raise RuntimeError("CoRA hierarchy owner was freed")
            if g._hcluster_samples is None:
                raise RuntimeError("Cluster sample IDs are missing; cannot condition experts")
            sample_ids = g._hcluster_samples
            bank = owner.cora_hier_cluster_bank
            if bank is None:
                conditions = torch.stack([ui_embedding] * N_EXPERTS, dim=1)
            else:
                conditions = bank.contexts(
                    ui_embedding, owner.rec_encoder,
                    sample_ids[0], sample_ids[1], args.condition_kind,
                )
            return _generator_four_conditions(g, conditions)

        generator.forward = types.MethodType(grouped_forward, generator)
        print(f"[HCLUSTER MODEL] condition={args.condition_kind} "
              f"E0=personal_raw E1-3={'raw_all' if args.condition_kind=='raw' else 'nested_group' if args.condition_kind=='hier' else 'flat_group'} "
              "all_rank4=True all_full_MF_feature=True shared_qformer=True "
              "base_CleanRank4_train_runner_untouched=True", flush=True)

    def wrap(fn):
        def forward(self, samples, *a, **kw):
            gen = self.llama_model_lora.base_model.weight_generator
            if gen._hcluster_samples is not None:
                raise RuntimeError("Nested hierarchical CoRA forward")
            gen._hcluster_samples = (samples["UserID"], samples["TargetItemID"])
            try:
                return fn(self, samples, *a, **kw)
            finally:
                gen._hcluster_samples = None
        return forward

    MiniGPT4Rec_vx.__init__ = cluster_init
    MiniGPT4Rec_vx.forward_v2 = wrap(old_train)
    MiniGPT4Rec_vx.generate_for_samples_v2 = wrap(old_eval)


def _paths(protocol):
    return clean._paths(protocol)


def plan(args):
    cfg, log, out = _paths(args.protocol)
    # clean.plan() creates the configuration under local_configs/, but writes
    # clean_rank4_manifest.json under logs/<protocol>/. The old code wrongly
    # looked next to the config, raising before F1 had even started training.
    # Some users already have this half-created configuration. Reuse it only
    # when nothing has trained and its exact run settings match the request.
    if log.exists() or out.exists():
        raise FileExistsError(
            "Training artifacts already exist; refusing to overwrite "
            f"log={log} output={out}"
        )
    manifest = log.parent / "clean_rank4_manifest.json"
    if cfg.exists():
        if not manifest.is_file():
            raise RuntimeError(
                "Partial config exists, but expected Clean Rank4 manifest is "
                f"missing: {manifest}. Refusing to guess the experiment settings."
            )
        existing = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        run = existing["run"]
        model = existing["model"]
        if (meta.get("protocol") != args.protocol or
                int(meta.get("seed", -1)) != SEED or
                int(run.get("seed", -1)) != SEED or
                str(meta.get("mode")) != str(args.route_mode) or
                abs(float(meta.get("strength", -99)) - args.route_strength) > 1e-12 or
                abs(float(run.get("init_lr", -99)) - args.lr) > 1e-12 or
                int(model["lora_config"].get("r", -1)) != RANK or
                model.get("ckpt") is not None or
                run.get("resume_ckpt_path") is not None or
                list(run.get("test_splits", [])) != [] or
                str(run.get("output_dir")) != str(out)):
            raise RuntimeError(
                f"Existing partial configuration does not match {args.protocol}; "
                "refusing to silently reuse or overwrite it."
            )
        if ("condition_kind" in meta and
                meta["condition_kind"] != args.condition_kind):
            raise RuntimeError(
                "Existing clustering mode differs from requested experiment."
            )
        print(
            f"[HCLUSTER RECOVER PLAN] validated partial config={cfg} "
            f"manifest={manifest}; no train log/output exists; preserving both",
            flush=True,
        )
    else:
        clean.plan(args)
        if not manifest.is_file():
            raise RuntimeError(
                f"Clean Rank4 did not write manifest to expected path: {manifest}"
            )
        meta = json.loads(manifest.read_text(encoding="utf-8"))
    meta.update({
        "experiment_family": "hierarchical_group_plus_personal_4xrank4",
        "condition_kind": args.condition_kind,
        "E0": "raw user+item personalized embedding",
        "E1_E2_E3": ("raw/raw/raw control" if args.condition_kind=="raw" else
                     "same flat cluster-level repeated" if args.condition_kind=="flat" else
                     "true nested coarse/middle/fine cluster centers"),
        "cluster_user_levels": USER_K,
        "cluster_item_levels": ITEM_K,
        "memberships": "fit once on train-seen entities + pretrained MF, seed42; fixed during joint MF training",
        "centroid_gradient": "means calculated from current trainable MF in every forward",
        "checkpoint_metadata": str(cfg.parent / "train_only_cluster_assignments.pt"),
        "qformer_cost": "four packed MF conditions for every experiment (including raw control)",
        "nested_guarantee": "middle//4=coarse, fine//2=middle",
        "no_D4_no_C6_no_RR1": True,
    })
    manifest.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[HCLUSTER PLAN] kind={args.condition_kind} "
          f"mode={args.route_mode} lambda={args.route_strength} "
          f"config={cfg}", flush=True)


def worker(args):
    if args.epochs != 0 or args.patience != 5:
        raise RuntimeError("Hierarchical CoRA experiments require uncapped/patience5 protocol")
    cfg, _, _ = _paths(args.protocol)
    if not cfg.exists():
        raise FileNotFoundError("Missing experiment configuration: " + str(cfg))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("GPU0 only")
    clean._remove_training_ceiling()
    joint.patch_runner(screen_epochs=0, mf_lr_ratio=0.1, patience=5)
    joint.install_compact_atomic_best_checkpoint()
    clean._patch_model(args)
    patch_cluster_conditioning(args)
    clean._audit_optimizer()
    print(f"[HCLUSTER WORKER] protocol={args.protocol} kind={args.condition_kind} "
          f"route={args.route_mode} lambda={args.route_strength} "
          "fresh_CoRA=True MF_trainable=True no_RR1=True "
          "four_independent_rank4=True no_D4_CTR_KD=True "
          "early_stop=5 no_max_epoch=True original_CoRA_LR=True", flush=True)
    sys.argv = ["train_collm_mf_din.py", "--cfg-path", str(cfg)]
    runpy.run_path(str(ROOT / "train_collm_mf_din.py"), run_name="__main__")


def _cpu_test_plan_recovery():
    """Regression test for config-vs-log manifest path and interrupted plan."""
    import tempfile
    with tempfile.TemporaryDirectory(prefix="hcluster_plan_test_") as work:
        work = Path(work)
        cfg = work / "local_configs" / "TRIAL" / "CLEAN_R4_MOE.yaml"
        log = work / "logs" / "TRIAL" / "CLEAN_R4_MOE.log"
        out = work / "output" / "TRIAL" / "CLEAN_R4_MOE"
        cfg.parent.mkdir(parents=True)
        log.parent.mkdir(parents=True)
        cfg.write_text(yaml.safe_dump({
            "run": {
                "seed": SEED,
                "init_lr": 3e-4,
                "test_splits": [],
                "output_dir": str(out),
                "resume_ckpt_path": None,
            },
            "model": {"lora_config": {"r": RANK}, "ckpt": None},
        }), encoding="utf-8")
        manifest = log.parent / "clean_rank4_manifest.json"
        manifest.write_text(json.dumps({
            "protocol": "TRIAL", "seed": SEED,
            "mode": "uniform", "strength": 0.2,
        }), encoding="utf-8")
        args = types.SimpleNamespace(
            protocol="TRIAL", route_mode="uniform",
            route_strength=0.2, condition_kind="hier", lr=3e-4,
        )
        old = globals()["_paths"]
        try:
            globals()["_paths"] = lambda _: (cfg, log, out)
            # clean.plan already created cfg + manifest, but hierarchy plan
            # previously crashed before adding its metadata. Repair safely.
            plan(args)
            loaded = json.loads(manifest.read_text(encoding="utf-8"))
            assert loaded["condition_kind"] == "hier"
            assert tuple(loaded["cluster_user_levels"]) == USER_K
            # Repeated planning is harmless before training has started.
            plan(args)
            # Any actual run log MUST protect against overwrite.
            log.touch()
            try:
                plan(args)
            except FileExistsError:
                pass
            else:
                raise AssertionError("Existing training log was not protected")
        finally:
            globals()["_paths"] = old
    print("[HCLUSTER PLAN CPU PASS] manifest stored in logs/, "
          "partial config repaired, existing training log protected", flush=True)


def cpu_test():
    _cpu_test_plan_recovery()
    torch.manual_seed(SEED)
    u_n, i_n, dim = 80, 160, 12
    rec = nn.Module()
    rec.user_embedding = nn.Embedding(u_n, dim)
    rec.item_embedding = nn.Embedding(i_n, dim)
    uids = list(range(1, 65))
    iids = list(range(1, 129))
    # small nested ratios, same structure as real (K,4K,8K).
    u = _nested_hierarchy(rec.user_embedding.weight, uids, (2,8,16), SEED)
    i = _nested_hierarchy(rec.item_embedding.weight, iids, (2,8,16), SEED+100)
    # The bank's production K is fixed 8/32/64, 16/64/128.
    bank = HierClusterBank(u, i)
    bank.user_k = (2,8,16)
    bank.item_k = (2,8,16)
    query_u = torch.tensor([1,4,9,75])
    query_i = torch.tensor([1,5,30,155])
    original = torch.cat((
        rec.user_embedding(query_u), rec.item_embedding(query_i)), dim=-1)
    for kind in KIND_CHOICES:
        ctx = bank.contexts(original, rec, query_u, query_i, kind)
        assert ctx.shape == (4,4,dim*2)
        assert torch.allclose(ctx[:,0], original)
        if kind == "hier":
            assert (ctx[:3,1]-ctx[:3,2]).abs().sum() > 0
            assert (ctx[:3,2]-ctx[:3,3]).abs().sum() > 0
            assert torch.allclose(ctx[-1,1],original[-1])
            assert torch.allclose(ctx[-1,2],original[-1])
            assert torch.allclose(ctx[-1,3],original[-1])
        elif kind == "flat":
            assert torch.allclose(ctx[:,1],ctx[:,2])
            assert torch.allclose(ctx[:,2],ctx[:,3])
        else:
            assert torch.allclose(ctx[:,0],ctx[:,3])
        (ctx.square().sum()).backward(retain_graph=True)
        assert rec.user_embedding.weight.grad is not None
        assert rec.user_embedding.weight.grad.abs().sum() > 0
        assert rec.item_embedding.weight.grad is not None
        assert rec.item_embedding.weight.grad.abs().sum() > 0
    # QFormer/independent rank4 generator shape and gradients.
    class FakeQFormer(nn.Module):
        def forward(self, query_embeds, encoder_hidden_states, **kw):
            return types.SimpleNamespace(
                last_hidden_state=query_embeds + encoder_hidden_states)
    gen = nn.Module()
    gen.embedding_size = dim
    gen.hidden_size = dim * 2
    gen.query_tokens = nn.Parameter(torch.randn(1, 4, dim*2))
    gen.qformer = FakeQFormer()
    gen.final = clean._heads(dim*2, dim*2)
    weights = _generator_four_conditions(
        gen, bank.contexts(original, rec, query_u, query_i, "hier"))
    assert weights.shape == (4,4,dim*2,4)
    weights.square().sum().backward()
    assert all(h.weight.grad is not None and h.weight.grad.abs().sum()>0 for h in gen.final)
    assert gen.query_tokens.grad is not None
    # Inherited CleanRank4 train-control check; no new LR monkeypatch.
    clean._remove_training_ceiling()
    print("[HCLUSTER CPU PASS] 1 personal + 3 true nested train-only "
          "cluster experts; 4xRank4; cold fallback; MF and head gradients; "
          "original CoRA optimizer and scheduler untouched", flush=True)


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpu-test", action="store_true")
    p.add_argument("--plan", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--protocol", default="CORA_HCLUSTER_R4_20261112_H1_HIER_UNIFORM")
    p.add_argument("--source-yaml", default=str(joint.BASE))
    p.add_argument("--condition-kind", choices=KIND_CHOICES, default="hier")
    p.add_argument("--route-mode", choices=("uniform","user","pair","layer_pair"), default="uniform")
    p.add_argument("--route-strength", type=float, default=0.2)
    p.add_argument("--warmup-microsteps", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-free-mib", type=int, default=22000)
    args = p.parse_args()
    if args.epochs != 0 or args.patience != 5 or args.lr <= 0 or args.lr > 0.001:
        p.error("Need no epoch ceiling, patience=5 and 0<lr<=0.001")
    if not (0 <= args.route_strength <= 1 and args.warmup_microsteps >= 0):
        p.error("Invalid router strength or warmup")
    return p, args


def main():
    p, args = _args()
    if args.cpu_test:
        cpu_test()
        return
    if args.plan:
        plan(args)
        return
    if args._worker:
        worker(args)
        return
    if not args.run:
        p.error("Choose --cpu-test, --plan or --run")
    cfg, log, out = _paths(args.protocol)
    if log.exists() or out.exists():
        raise FileExistsError("Experiment artifacts exist; do not overwrite: " + args.protocol)
    # Also repair a config generated by the pre-fix script: no manual cleanup.
    plan(args)
    clean.gpu_check(args.min_free_mib)
    cmd = [
        sys.executable, "-u", str(Path(__file__).resolve()), "--_worker",
        "--protocol", args.protocol,
        "--source-yaml", str(args.source_yaml),
        "--condition-kind", args.condition_kind,
        "--route-mode", args.route_mode,
        "--route-strength", str(args.route_strength),
        "--warmup-microsteps", str(args.warmup_microsteps),
        "--lr", str(args.lr),
        "--epochs", "0", "--patience", "5",
    ]
    print("[HCLUSTER RUN] " + " ".join(cmd), flush=True)
    with log.open("x", encoding="utf-8") as fh:
        # Child process remains FOREGROUND-bound; parent waits and streams log.
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            fh.write(line)
        rc = proc.wait()
    if rc:
        raise SystemExit(f"[HCLUSTER STOP] rc={rc} log={log}")
    print(f"[HCLUSTER DONE] {args.protocol} output={out}", flush=True)


if __name__ == "__main__":
    main()
