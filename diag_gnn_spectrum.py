"""
GNN representation spectrum diagnostic (zero-training-cost analysis).

Loads one or more GraphLoRA checkpoints, extracts the NGCF weights + MF
embedding table, runs the GNN over the FULL train graph, and reports the
singular spectrum of the resulting collaborative representation Z_G
(256-d penult, the exact view the semantic MSE aligns). Also reports the
spectrum of the raw MF embedding table as the pre-GNN reference.

Usage:
  python diag_gnn_spectrum.py --ckpt output/.../checkpoint_best.pth [more.pth ...] \
      [--mf-baseline checkpoints/0912_..._best_model_...pth]

No LLM is loaded; this runs on CPU or any free GPU in seconds.
"""
import argparse

import numpy as np
import scipy.sparse as sp
import torch

from minigpt4.models.minigpt4rec_v2 import NGCF


def spectrum_summary(Z, tag):
    Zc = Z - Z.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(Zc)
    s = s[s > 1e-10]
    e = s / s.sum()
    e2 = s ** 2 / (s ** 2).sum()
    pr = (s ** 2).sum() ** 2 / (s ** 4).sum()          # participation ratio
    eff_rank = float(torch.exp(-(e2 * torch.log(e2.clamp_min(1e-12))).sum()))
    cum = torch.cumsum(e, dim=0)
    print(f"\n[{tag}]  nodes={Z.shape[0]} dim={Z.shape[1]}  rank={len(s)}")
    print(f"  sigma1/sum   = {e[0].item():.4f}")
    print(f"  energy@k     = " + "  ".join(
        f"k={k}:{cum[k - 1].item():.3f}" for k in (1, 2, 5, 10, 20, 50) if k <= len(s)))
    print(f"  part. ratio  = {pr.item():.2f}   eff. rank = {eff_rank:.2f}")
    top = ", ".join(f"{v.item():.3f}" for v in s[:10])
    print(f"  top-10 sigma = {top}")
    return s.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--mf-baseline", default=None,
                   help="pre-GNN MF encoder state dict (embedding-table reference)")
    p.add_argument("--graph", default="dataset/ml-1m/movie_graph_train.npz")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    adj = sp.load_npz(args.graph).tocoo()
    indices = torch.tensor(np.vstack([adj.row, adj.col]), dtype=torch.long)
    values = torch.tensor(adj.data, dtype=torch.float32)
    full_adj = torch.sparse_coo_tensor(indices, values, adj.shape).coalesce().to(args.device)

    if args.mf_baseline:
        sd = torch.load(args.mf_baseline, map_location="cpu")
        if "model" in sd:
            sd = sd["model"]
        eu = sd["user_embedding.weight"]
        ei = sd["item_embedding.weight"]
        E0 = torch.cat([eu, ei], dim=0).float()
        spectrum_summary(E0, f"MF table (pre-GNN baseline) {args.mf_baseline.split('/')[-1]}")

    for path in args.ckpt:
        sd = torch.load(path, map_location="cpu")
        if "model" in sd:
            sd = sd["model"]
        eu = sd["rec_encoder.user_embedding.weight"]
        ei = sd["rec_encoder.item_embedding.weight"]
        E = torch.cat([eu, ei], dim=0).float().to(args.device)
        spectrum_summary(E, f"MF table (ckpt) {path.split('/')[-1]}")

        gcn_sd = {k[len("gcn."):]: v for k, v in sd.items() if k.startswith("gcn.")}
        first = gcn_sd["layers.0.W1.weight"]
        in_dim = first.shape[1]
        hidden = first.shape[0]
        out_dim = gcn_sd["layers.-1.W1.weight"].shape[0] if "layers.-1.W1.weight" in gcn_sd else None
        # infer num layers & out_dim from the last present layer index
        layer_ids = sorted({int(k.split(".")[1]) for k in gcn_sd if k.startswith("layers.")})
        n_layers = len(layer_ids)
        out_dim = gcn_sd[f"layers.{layer_ids[-1]}.W1.weight"].shape[0]
        gcn = NGCF(in_dim=in_dim, hidden_dim=hidden, out_dim=out_dim, num_layers=n_layers, dropout=0.1)
        missing, unexpected = gcn.load_state_dict(gcn_sd, strict=False)
        assert not missing, f"missing GNN keys: {missing}"
        gcn = gcn.to(args.device).eval()
        with torch.no_grad():
            out, penult, _ = gcn(full_adj, E, None, return_penult=True)
        spectrum_summary(penult.float().cpu(), f"Z_G penult (GNN) {path.split('/')[-1]}")
        # note: penult is row-normalized inside NGCF; spectrum of the
        # unnormalized trunk would additionally carry node-degree magnitude


if __name__ == "__main__":
    main()
