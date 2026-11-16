#!/usr/bin/env python3
"""Three isolated 2-personal + 2-group CoRA MoE experiments (4 x Rank4).

Exactly four independent dynamic Rank4 heads and four independent Rank4 P
projections per Q/K/V/O layer. Reuse the verified Clean Rank4/HierCluster
training, runner, QFormer, checkpoint, and LR implementation unmodified.

O1: E0 raw, E1 raw (independent head), E2 coarse group, E3 fine group;
    uniform routing. This isolates the 2P2G budget allocation.
O2: E0 raw, E1 raw - 0.5*coarse (personal-specific contrast), E2 coarse,
    E3 fine; uniform routing. This isolates personal residual conditioning.
O3: same inputs as O2, pair-aware soft routing with lambda=0.2; isolates
    whether moderate routing helps without the unstable full-strength route.

All cluster memberships are nested/fixed and derived from train_ood2.pkl
and pretrained MF at model initialization. Current trainable MF is used
to calculate differentiable group means; held-out IDs keep raw vectors.
No RR1, no D4 neighbor TopK, no C6 CTR/KD; no added trainable parameters.
All three runs start from fresh CoRA adapters and pretrained MF, seed=42.
GPU0 foreground/serial, valid only, no epoch cap, early stop patience=10.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for d in (str(ROOT), str(HERE)):
    if d not in sys.path:
        sys.path.insert(0, d)
import cora_hiercluster_rank4_moe as hier
import cora_clean_rank4_moe6 as clean
import cora_joint8 as joint

SEED = 42
BETA = 0.5
PREFIX = "CORA_2P2G_R4_20261112"
TRIALS = {
    "O1_RAW2_G2_UNIFORM": ("raw2", "uniform", 0.2),
    "O2_RESID_G2_UNIFORM": ("resid", "uniform", 0.2),
    "O3_RESID_G2_PAIR02": ("resid", "pair", 0.2),
}
_ORIGINAL_CONTEXTS = hier.HierClusterBank.contexts


def _personal_group_contexts(bank, original_ui, rec, user_ids, item_ids,
                             kind, layout):
    """Rearrange only the four expert conditions; no source model changes.

    Inherited condition order: [raw, coarse, middle, fine].
    New condition order: [raw, personal2, coarse, fine].
    Coarse and fine share the SAME nested train-only memberships as H1/H2.
    """
    if kind != "hier":
        raise RuntimeError("2P2G requires original nested hierarchical groups")
    if layout not in ("raw2", "resid"):
        raise ValueError(f"Unknown expert layout: {layout}")
    original = _ORIGINAL_CONTEXTS(
        bank, original_ui, rec, user_ids, item_ids, kind
    )
    if original.ndim != 3 or original.shape[1] != 4:
        raise RuntimeError("Expected four full-dimension conditions")
    raw, coarse, fine = original[:, 0], original[:, 1], original[:, 3]
    if layout == "raw2":
        personal2 = raw  # independent output head even though input matches
    else:
        dim = rec.user_embedding.weight.shape[-1]
        if raw.shape[-1] != 2 * dim:
            raise RuntimeError("Full concatenated user/item MF dimension changed")
        uids = user_ids.long().view(-1).to(bank.user_labels.device)
        iids = item_ids.long().view(-1).to(bank.item_labels.device)
        known_users = (bank.user_labels[0].index_select(0, uids) >= 0)
        known_items = (bank.item_labels[0].index_select(0, iids) >= 0)
        # Conservative 0.5x subtraction avoids magnifying tiny residuals and
        # keeps the original MF magnitude scale reasonably comparable.
        # For each cold/padding entity, retain its ORIGINAL vector, rather
        # than forming an artificial zero residual.
        u_raw, i_raw = raw.split(dim, dim=-1)
        u_coarse, i_coarse = coarse.split(dim, dim=-1)
        u_context = torch.where(
            known_users.to(raw.device)[:, None],
            u_raw - BETA * u_coarse, u_raw,
        )
        i_context = torch.where(
            known_items.to(raw.device)[:, None],
            i_raw - BETA * i_coarse, i_raw,
        )
        personal2 = torch.cat((u_context, i_context), dim=-1)
    mixed = torch.stack((raw, personal2, coarse, fine), dim=1)
    if mixed.shape != original.shape:
        raise RuntimeError("2P2G changed the Rank4 dynamic generator shape")
    return mixed


def _install_conditions(layout):
    """Only the current child process is patched; same optimizer & scheduler."""
    if hier.HierClusterBank.contexts is not _ORIGINAL_CONTEXTS:
        raise RuntimeError("Existing cluster patch detected; refusing double-patch")

    def rearranged(self, original_ui, rec, user_ids, item_ids, kind):
        return _personal_group_contexts(
            self, original_ui, rec, user_ids, item_ids, kind, layout
        )

    hier.HierClusterBank.contexts = rearranged
    print(f"[2P2G PATCH PASS] layout={layout}; E0=raw E1={'raw' if layout=='raw2' else 'raw-minus-half-coarse'} "
          "E2=coarse E3=fine; 4 independent Rank4 heads; no training-code changes",
          flush=True)


def _build_args(raw):
    if raw.trial not in TRIALS:
        raise ValueError("Unknown trial " + raw.trial)
    layout, route, strength = TRIALS[raw.trial]
    raw.protocol = f"{PREFIX}_{raw.trial}"
    raw.expert_layout = layout
    raw.condition_kind = "hier"
    raw.route_mode = route
    raw.route_strength = strength
    raw.epochs = 0
    raw.patience = 10
    raw.lr = 3e-4
    raw.warmup_microsteps = 200
    return raw


def plan(args):
    cfg, log, out = clean._paths(args.protocol)
    if log.exists() or out.exists():
        raise FileExistsError(f"Existing experiment data must never be overwritten: {args.protocol}")
    # HIER plan already protects/reuses a partially written config, including
    # the correct log/manifest path. No new Runner / LR changes here.
    hier.plan(args)
    manifest = log.parent / "clean_rank4_manifest.json"
    if not manifest.is_file():
        raise RuntimeError("Hierarchical experiment plan did not write log manifest")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    existing = data.get("expert_layout")
    if existing is not None and existing != args.expert_layout:
        raise RuntimeError("Conflicting expert layout in an existing experiment plan")
    data.update({
        "experiment_family": "CORA_2_personal_2_cluster_group_rank4",
        "expert_layout": args.expert_layout,
        "E0": "original user-item MF",
        "E1": ("original user-item MF with independent Rank4 head"
               if args.expert_layout == "raw2" else
               "original MF minus 0.5x coarse group centroid, cold falls back to original"),
        "E2": "coarse nested cluster centroid: U K8 / I K16",
        "E3": "fine nested cluster centroid: U K64 / I K128",
        "personal_contrast_strength": BETA,
        "group_cluster_membership": "train_only, fixed; centroids use current MF weights",
        "QFormer": "shared, four inputs packed into a single forward",
        "total_rank_budget": 16,
        "additional_trainable_params": 0,
        "patience": 10,
        "prior_experiments_untouched": True,
    })
    manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[2P2G PLAN PASS] trial={args.trial} layout={args.expert_layout} "
          f"routing={args.route_mode} lambda={args.route_strength} "
          f"patience=10 config={cfg} manifest={manifest}", flush=True)


def worker(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("GPU0 only")
    cfg, log, out = clean._paths(args.protocol)
    if not cfg.is_file():
        raise FileNotFoundError("Missing experiment plan: " + str(cfg))
    manifest = log.parent / "clean_rank4_manifest.json"
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if (metadata.get("expert_layout") != args.expert_layout
            or metadata.get("early_stopping_patience") != 10):
        raise RuntimeError("Manifest does not match selected trial / patience10")
    _install_conditions(args.expert_layout)
    print(f"[2P2G WORKER] trial={args.trial} seed=42 4xRank4 "
          "new_condition_only=True no_RR1_D4_C6=True "
          "MF_joint=True original_LR=True uncapped=True patience=10 valid_only=True",
          flush=True)
    hier.worker(args)


def cpu_test():
    # Recheck the upstream experiment planning regression before launching
    # three GPU runs (manifest belongs under logs/, not local_configs/).
    hier._cpu_test_plan_recovery()
    torch.manual_seed(SEED)
    dim = 12
    rec = nn.Module()
    rec.user_embedding = nn.Embedding(80, dim)
    rec.item_embedding = nn.Embedding(160, dim)
    u_seen = list(range(1, 65))
    i_seen = list(range(1, 129))
    u = hier._nested_hierarchy(
        rec.user_embedding.weight, u_seen, (2, 8, 16), SEED
    )
    i = hier._nested_hierarchy(
        rec.item_embedding.weight, i_seen, (2, 8, 16), SEED + 100
    )
    bank = hier.HierClusterBank(u, i)
    bank.user_k, bank.item_k = (2, 8, 16), (2, 8, 16)
    uids = torch.tensor([1, 75, 3, 75], dtype=torch.long)
    iids = torch.tensor([1, 5, 155, 155], dtype=torch.long)
    raw = torch.cat(
        [rec.user_embedding(uids), rec.item_embedding(iids)], dim=-1
    )
    original = _ORIGINAL_CONTEXTS(bank, raw, rec, uids, iids, "hier")
    for layout in ("raw2", "resid"):
        out = _personal_group_contexts(
            bank, raw, rec, uids, iids, "hier", layout
        )
        assert out.shape == (len(uids), 4, dim*2)
        assert torch.allclose(out[:,0], raw)
        assert torch.allclose(out[:,2], original[:,1])
        assert torch.allclose(out[:,3], original[:,3])
        if layout == "raw2":
            assert torch.allclose(out[:,1], raw)
        else:
            # Seen user/item: exact 0.5 coarse subtraction.
            assert torch.allclose(
                out[0,1], raw[0] - BETA*original[0,1], atol=1e-6
            )
            # Unseen user: raw user half, even for a seen item.
            assert torch.allclose(out[1,1,:dim], raw[1,:dim])
            # Unseen item: raw item half, even for a seen user.
            assert torch.allclose(out[2,1,dim:], raw[2,dim:])
            # Both unseen: the entire private condition is raw.
            assert torch.allclose(out[3,1], raw[3])
        # Gradient path must reach both current MF table and the centroids.
        rec.zero_grad(set_to_none=True)
        test_loss = (out * torch.randn_like(out)).sum()
        # Same synthetic raw MF embedding is shared by both ablation checks.
        # Keep the graph for the second check; production forwards build a new
        # graph per batch and do not need retain_graph.
        test_loss.backward(retain_graph=True)
        assert rec.user_embedding.weight.grad is not None
        assert rec.item_embedding.weight.grad is not None
        assert float(rec.user_embedding.weight.grad.abs().sum()) > 0
        assert float(rec.item_embedding.weight.grad.abs().sum()) > 0
    # Same actual packed QFormer and 4 distinct Rank4 generator-head geometry.
    class FakeQFormer(nn.Module):
        def forward(self, query_embeds, encoder_hidden_states, **kw):
            return type("Hidden", (), {
                "last_hidden_state": query_embeds + encoder_hidden_states
            })()
    gen = nn.Module()
    gen.embedding_size = dim
    gen.hidden_size = 2*dim
    gen.query_tokens = nn.Parameter(torch.randn(1, 4, 2*dim))
    gen.qformer = FakeQFormer()
    gen.final = clean._heads(2*dim, 2*dim)
    new_ctx = _personal_group_contexts(
        bank, raw.detach(), rec, uids, iids, "hier", "resid"
    )
    weight = hier._generator_four_conditions(gen, new_ctx)
    assert weight.shape == (len(uids), 4, 2*dim, 4)
    (weight * torch.randn_like(weight)).sum().backward()
    assert all(h.weight.grad is not None for h in gen.final)
    # Patch installed in-process only and keeps original condition API intact.
    _install_conditions("resid")
    patched = bank.contexts(raw.detach(), rec, uids, iids, "hier")
    assert torch.allclose(patched, new_ctx.detach(), atol=1e-6)
    hier.HierClusterBank.contexts = _ORIGINAL_CONTEXTS
    print("[2P2G CPU PASS] 2P2G expert ordering, nested coarse/fine, "
          "warm raw2/residual comparison, cold fallback, MF gradients, "
          "four independent Rank4 heads", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpu-test", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--trial", choices=tuple(TRIALS), default="O1_RAW2_G2_UNIFORM")
    p.add_argument("--source-yaml", default=str(joint.BASE))
    p.add_argument("--min-free-mib", type=int, default=22000)
    args = _build_args(p.parse_args())
    if args.cpu_test:
        cpu_test()
        return
    if args._worker:
        worker(args)
        return
    if not args.run:
        p.error("Choose --cpu-test or --run")
    cfg, log, out = clean._paths(args.protocol)
    if log.exists() or out.exists():
        raise FileExistsError(f"Experiment already started: {args.protocol}")
    plan(args)
    clean.gpu_check(args.min_free_mib)
    cmd = [
        sys.executable, "-u", str(Path(__file__).resolve()), "--_worker",
        "--trial", args.trial, "--source-yaml", args.source_yaml,
    ]
    print(f"[2P2G START] trial={args.trial} cmd={' '.join(cmd)}", flush=True)
    with log.open("x", encoding="utf-8") as fh:
        # Parent remains attached to terminal; no nohup / background launch.
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", bufsize=1
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            fh.write(line)
        code = proc.wait()
    if code != 0:
        raise SystemExit(f"[2P2G ERROR] trial={args.trial} status={code} log={log}")
    print(f"[2P2G DONE] trial={args.trial} valid-only output={out}", flush=True)


if __name__ == "__main__":
    main()
