#!/usr/bin/env python3
"""R1-R4 CPU smoke: exact identity, target exclusion, gradients, depth/projection routes.

No Vicuna, GPU, graph artifact, network, or private data necessary.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != ROOT]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F
from experiments.no31_cora_original.preference_router import PreferenceResidualRouter
from experiments.no31_cora_original.night26_preference_four import ARMS, COMMON
from experiments.no31_cora_original import night8 as core


def main():
    assert len(ARMS) == 4 and len({r["id"] for r in ARMS}) == 4
    assert [r["preference_router"] for r in ARMS] == list(PreferenceResidualRouter.MODES)
    assert COMMON["seed"] == 42 and COMMON["group_projectors"] == "group8"
    assert COMMON["graph_joint"] is False and COMMON["graph_mode"] == "off"
    assert COMMON["collab_mode"] == "off"
    for row in ARMS:
        env = core.build_environment(dict(COMMON, **row))
        assert env["NO31_CORA_PREFERENCE_ROUTER"] == row["preference_router"]
        assert env["NO31_CORA_GRAPH_JOINT"] == "0"
        assert env["NO31_CORA_GRAPH_MODE"] == "off"
        assert env["NO31_CORA_GROUP_PROJECTORS"] == "group8"
        assert env["S2_SMALL_KD"] == ".05" or float(env["S2_SMALL_KD"]) == .05
    torch.manual_seed(42)
    mf_users = torch.randn(4, 256).float() * 0.05
    mf_items = torch.randn(11, 256).float() * 0.05
    users = torch.tensor([0, 1, 2])
    targets = torch.tensor([1, 5, 8])
    # Target included deliberately, to verify that target item is excluded.
    history = torch.tensor([
        [1, 2, 3, 0],
        [2, 5, 7, 0],
        [8, 8, 9, 0],
    ])
    history_without_target = history.clone()
    history_without_target[history == targets.unsqueeze(1)] = 0
    feats = None
    for mode in PreferenceResidualRouter.MODES:
        rng = torch.get_rng_state()
        model = PreferenceResidualRouter(mode, epsilon=.1)
        assert model.head[-1].weight.detach().count_nonzero() == 0
        assert model.head[-1].bias.detach().count_nonzero() == 0
        observed = model.preference_features(
            users, targets, history, mf_users, mf_items, padding_index=0)
        excluded = model.preference_features(
            users, targets, history_without_target, mf_users, mf_items, padding_index=0)
        assert torch.equal(observed, excluded), "Target item leaked into history features"
        assert observed.shape == (3, 515) and torch.isfinite(observed).all()
        if feats is None:
            feats = observed
        gains = model(users, targets, history, mf_users, mf_items, padding_index=0)
        assert gains.shape == (3, 8, 4)
        assert torch.equal(gains, torch.ones_like(gains))
        # All 128 original Group8 residual terms: identity multipliers leave
        # forward and P/W gradients EXACTLY unchanged at initialization.
        x = torch.randn(3, 5, 16)
        A = torch.randn(3, 16, 4, requires_grad=True)
        P = torch.randn(12, 4, requires_grad=True)
        base = torch.randn(3, 5, 12)
        delta = F.linear(torch.bmm(x, A), P)
        plain = base + delta
        gw = torch.autograd.grad(plain.square().sum(), (A, P), retain_graph=True)
        routed = base + gains[:, 0, 0].view(3, 1, 1) * delta
        rgw = torch.autograd.grad(routed.square().sum(), (A, P), retain_graph=True)
        assert torch.equal(plain, routed)
        assert all(torch.equal(u, v) for u, v in zip(gw, rgw))

        # Use a nonzero residual and asymmetric synthetic per-(group,proj)
        # weights to ensure route head receives meaningful gradients.
        weights = torch.arange(1, 33).float().view(1, 8, 4)
        with torch.no_grad():
            pre = model.head[-1].weight.detach().clone()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        opt.zero_grad()
        loss = (model(users, targets, history, mf_users, mf_items, 0) * weights).square().mean()
        loss.backward()
        assert model.head[-1].weight.grad is not None
        assert model.head[-1].weight.grad.abs().max() > 0
        assert model._positive_grad_events > 0
        opt.step()
        assert not torch.equal(pre, model.head[-1].weight.detach())
        newg = model(users, targets, history, mf_users, mf_items, 0)
        assert newg.min() >= .9 - 1e-6 and newg.max() <= 1.1 + 1e-6
        assert not torch.equal(newg, torch.ones_like(newg))

        if mode == "global":
            assert torch.equal(newg[:, 0, 0], newg[:, 7, 3])
        elif mode == "depth8":
            assert torch.equal(newg[:, :, 0], newg[:, :, 3])
        elif mode == "projection2":
            assert torch.equal(newg[:, :, 0], newg[:, :, 1])
            assert torch.equal(newg[:, :, 2], newg[:, :, 3])
            assert torch.equal(newg[:, 0, :], newg[:, 7, :])
        else:
            assert tuple(model.head[-1].weight.shape) == (32, 64)
        print(f"[N29-CPU-ROUTE-PASS] mode={mode} identity=True "
              "target_exclusion=True P_gradient_parity=True "
              "router_updated=True bounded=True", flush=True)
    no_history = PreferenceResidualRouter("global").preference_features(
        users, targets, None, mf_users, mf_items, padding_index=0)
    assert torch.equal(no_history[:, -2:], torch.zeros_like(no_history[:, -2:]))
    print("[N29-CPU-PASS] four routers, target-excluded history, zero-identity "
          "CoRA, per-group/projection routing and gradient path", flush=True)


if __name__ == "__main__":
    main()
