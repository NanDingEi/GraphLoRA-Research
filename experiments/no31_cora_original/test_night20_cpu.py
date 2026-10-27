#!/usr/bin/env python3
"""CPU-only synthetic NIGHT20: prevent label leakage and enforce Q-Former parity."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn

from experiments.no31_cora_original.night20_features import (
    install_frozen_history, make_collaborative_features,
)
from experiments.no31_cora_original.weight_generator import PerceptualWeightsGenerator
from experiments.no31_cora_original.night20 import EXPERIMENTS


def synthetic_history_probe(tmpdir):
    torch.manual_seed(42)
    n_users, n_items, dim = 5, 33, 256
    users = torch.randn(n_users, dim)
    items = torch.randn(n_items, dim)
    items[0] = 0
    groups = np.full(n_items, -1, dtype=np.int64)
    groups[1:] = (np.arange(1, n_items) - 1) // 2  # 16 x 2-item groups
    proto = np.stack(
        [items[torch.tensor(groups == k)].numpy().mean(axis=0)
         for k in range(16)], axis=0
    ).astype(np.float32)
    residual = items.numpy().copy()
    residual[1:] -= proto[groups[1:]]
    residual[0] = 0
    np.savez_compressed(
        tmpdir / "prototypes_k16.npz",
        item_to_cluster=groups, prototypes=proto,
        mf_residual=residual,
    )
    # User1 likes both items of community0, user2 has only item1 with
    # duplicate (positive, negative) interactions: target exclusion removes ALL.
    pd.DataFrame(
        [(1, 1, 1), (1, 2, 1), (1, 3, 0),
         (2, 1, 1), (2, 1, 0), (3, 3, 1)],
        columns=["uid", "iid", "label"],
    ).to_pickle(tmpdir / "train_ood2.pkl")
    model = nn.Module()
    model.register_buffer("cora_fixed_user", users)
    model.register_buffer("cora_fixed_item", items)
    install_frozen_history(
        model, str(tmpdir / "prototypes_k16.npz"),
        str(tmpdir / "train_ood2.pkl"),
    )
    coarse, fine = make_collaborative_features(
        model, torch.tensor([1, 2]), torch.tensor([1, 1])
    )
    assert coarse.shape == (2, 514) and fine.shape == (2, 513)
    # User 1 target item 1 is excluded, but item 2 remains in the user's
    # positive history, so history residual equals item2 residual.
    assert torch.allclose(
        fine[0, 256:512], model.n20_item_residual[2], atol=1e-6,
    )
    # For user 2, all training interactions are for target item 1:
    # leave-target-out history must be empty and preference equals prior.
    assert torch.equal(fine[1, 256:512], torch.zeros(256))
    global_prior = model.n20_group_prior[0]
    assert torch.allclose(
        coarse[1, -1], torch.logit(global_prior), atol=1e-6,
    )
    assert torch.isfinite(coarse).all() and torch.isfinite(fine).all()
    # Check source files contribute no target-positive history for singleton u.
    print("[NIGHT20-CPU] train-only prototype reconstruction + leave-target-out "
          "pair exclusion PASS", flush=True)


def generator_routing_probe():
    torch.manual_seed(42)
    g = PerceptualWeightsGenerator(
        num_queries=4,
        user_embedding_size=16,
        hidden_size=32,
        r=16,
        input_mode="concat",
        injection_mode="shared",
        group_projector_mode="group8",
    ).eval()
    # Exactly the same two-head architecture is instantiated in all four arms.
    g.n20_coarse_head = nn.Sequential(
        nn.Linear(34, 64), nn.GELU(), nn.Linear(64, 32),
    )
    g.n20_fine_head = nn.Sequential(
        nn.Linear(33, 64), nn.GELU(), nn.Linear(64, 32),
    )
    for head in (g.n20_coarse_head, g.n20_fine_head):
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
    g.n20_query_cap = 0.1
    pair = torch.randn(2, 32)
    zc = torch.randn(2, 34)
    zf = torch.randn(2, 33)
    original = g(pair)
    schedules = {
        "coarse_only": [(1., 0.)] * 8,
        "fine_only": [(0., 1.)] * 8,
        "dual_uniform": [(0.5, 0.5)] * 8,
        "dual_c2f": [(v, 1-v) for v in
                     (1., 1., .75, .75, .25, .25, 0., 0.)],
    }
    assert set(schedules) == {row["collab_mode"] for row in EXPERIMENTS}
    for mode, schedule in schedules.items():
        g.n20_schedule = schedule
        g.zero_grad(set_to_none=True)
        conditioned = g(pair, collab_features=(zc, zf))
        for idx in range(8):
            assert torch.equal(original[idx], conditioned[idx]), (
                mode, idx, "Zero-init must yield exact E0 dynamic-A parity"
            )
        # Non-zero derivatives through active branch must be available from
        # the very first optimizer step. Zero-init earlier head layers receive
        # gradients only after the final zero layer begins to update.
        probe = torch.randn_like(conditioned[0])
        loss = sum((conditioned[idx] * probe).sum() for idx in range(8))
        loss.backward()
        for name, head, active in (
            ("coarse", g.n20_coarse_head, any(a for a,b in schedule)),
            ("fine", g.n20_fine_head, any(b for a,b in schedule)),
        ):
            grad = head[-1].weight.grad
            if active:
                assert grad is not None and grad.abs().sum().item() > 0, (
                    mode, name, "Active branch must have nonzero gradient"
                )
            else:
                assert grad is None or grad.abs().sum().item() == 0, (
                    mode, name, "Inactive branch must not train"
                )
        print(f"[NIGHT20-CPU] mode={mode} E0 dynamic-A exact parity "
              "and active/inactive gradients PASS", flush=True)


def main():
    with tempfile.TemporaryDirectory(prefix="n20_cpu_") as directory:
        synthetic_history_probe(Path(directory))
    generator_routing_probe()
    print("[NIGHT20-CPU-PASS] 4 arms, no leakage, parity, gradients", flush=True)


if __name__ == "__main__":
    main()
