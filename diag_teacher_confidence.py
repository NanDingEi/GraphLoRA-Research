"""
Teacher-confidence diagnostic (zero-training-cost analysis).

Question: which LLM semantic relations are worth trusting? For every item
pair in a batch we measure how stable the teacher's relation estimate is
ACROSS DEPTHS (layers 6/8/10 of the LLM):

    var_ij = Var_l( S^(l)_ij ),   S^(l)_ij = cos(h_i^(l), h_j^(l))

and then test whether high-uncertainty pairs carry larger semantic
alignment error |S_G - S_L8| against the GNN view. If they do, a
confidence-weighted L_sem is justified; if not, the direction is closed.

Usage:
  python diag_teacher_confidence.py --cfg-path tran_configs/collm_pretrain_mf_ood_rlm_eval_base.yaml \
      [--max-batches 400] [--out diag_conf_out]

The config must point at the checkpoint to analyse (model.ckpt). Data and
model build follow train_collm_mf_din.py exactly.
"""
import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.models import *
from minigpt4.tasks import *


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg-path", default="tran_configs/collm_pretrain_mf_ood_rlm_eval_base.yaml")
    p.add_argument("--max-batches", type=int, default=400)
    p.add_argument("--out", default="diag_confidence.npz")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(argparse.Namespace(cfg_path=args.cfg_path, options=None))

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    try:  # movie
        data_dir = cfg.datasets_cfg.movie_ood.path
    except Exception:  # amazon
        data_dir = cfg.datasets_cfg.amazon_ood.path
    import pandas as pd
    tr = pd.read_pickle(data_dir + "train_ood2.pkl")
    va = pd.read_pickle(data_dir + "valid_ood2.pkl")
    te = pd.read_pickle(data_dir + "test_ood2.pkl")
    cfg.model_cfg.rec_config.user_num = int(max(tr.uid.max(), va.uid.max(), te.uid.max()) + 1)
    cfg.model_cfg.rec_config.item_num = int(max(tr.iid.max(), va.iid.max(), te.iid.max()) + 1)

    model = task.build_model(cfg)
    model = model.to(cfg.run_cfg.device)
    model.eval()
    model.set_mode("v2")
    num_users = model.num_users
    device = model.device
    print(f"[conf-diag] checkpoint loaded, num_users={num_users}, num_items={model.num_items}")

    valid_ds = None
    for name, split in datasets.items():
        if "valid" in split:
            valid_ds = split["valid"]
    assert valid_ds is not None
    loader = DataLoader(
        valid_ds,
        batch_size=int(cfg.run_cfg.batch_size_eval),
        shuffle=False,
        collate_fn=getattr(valid_ds, "collater", None),
        num_workers=0,
    )

    teacher_layers = (6, 8, 10)
    rec_s = []   # S_G per pair
    rec_vars = []  # Var_l(S_L) per pair
    rec_s8 = []
    spread = []  # max_l S_l - min_l S_l (sanity)
    gap12 = []   # |S6 - S10| raw depth disagreement

    n_batches = 0
    with torch.no_grad():
        for samples in loader:
            if n_batches >= args.max_batches:
                break
            samples = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in samples.items()
            }
            prompt = random.choice(model.prompt_with_p([5, 5, 5, 1]))
            sample_embeds, atts, samples_encode = model.prompt_based_encode_v2(prompt, samples)

            # ---- GNN view (mirrors compute_feedback_gcn eval path) ----
            node_ids = samples_encode["node_ids"].to(device)
            sub_adj, sub_node_ids = model.get_subgraph(node_ids)
            node_map = {old.item(): new for new, old in enumerate(sub_node_ids)}
            batch_mask = samples_encode["replaced_idx"][:, 0] < samples["UserID"].shape[0]
            batch_indices = samples_encode["replaced_idx"][batch_mask, 0]
            token_positions = samples_encode["replaced_idx"][batch_mask, 1]
            batch_node_ids = node_ids[batch_mask]
            local_node_ids = torch.tensor(
                [node_map.get(nid.item(), -1) for nid in batch_node_ids], device=device
            )
            mf_embeddings = torch.cat(
                [model.rec_encoder.user_embedding.weight, model.rec_encoder.item_embedding.weight],
                dim=0,
            )
            model.gcn.eval()
            with torch.cuda.amp.autocast(enabled=False):
                gcn_out, gcn_penult, _ = model.gcn(
                    sub_adj.to(dtype=torch.float32, device=device),
                    mf_embeddings[sub_node_ids].to(dtype=torch.float32, device=device),
                    sub_node_ids,
                    return_penult=True,
                )
            valid = local_node_ids >= 0
            token_gnn = torch.zeros(
                (local_node_ids.shape[0], gcn_penult.shape[1]),
                dtype=gcn_penult.dtype, device=device,
            )
            token_gnn[valid] = gcn_penult[local_node_ids[valid]]

            # ---- teacher pass: one forward to layer 10, take 6/8/10 ----
            with model.maybe_autocast():
                t_out = model.llama_model_lora(
                    inputs_embeds=sample_embeds,
                    attention_mask=atts,
                    return_dict=True,
                    use_cache=False,
                    exit_layer=max(teacher_layers),
                    output_hidden_states=True,
                )
            hs = t_out.hidden_states
            hs_items = {
                l: hs[l][batch_indices, token_positions].float() for l in teacher_layers
            }

            item_rows = ((batch_node_ids >= num_users) & (local_node_ids >= 0)).nonzero(as_tuple=True)[0]
            if item_rows.numel() < 2:
                n_batches += 1
                continue
            item_ids = batch_node_ids[item_rows]
            z = torch.nn.functional.normalize(token_gnn[item_rows].float(), dim=-1)
            h_norm = {
                l: torch.nn.functional.normalize(hs_items[l][item_rows], dim=-1) for l in teacher_layers
            }
            pair_mask = torch.triu(
                torch.ones(item_rows.numel(), item_rows.numel(), dtype=torch.bool, device=device),
                diagonal=1,
            )
            pair_mask &= item_ids.unsqueeze(1) != item_ids.unsqueeze(0)
            if not bool(pair_mask.any()):
                n_batches += 1
                continue

            s_g = (z @ z.t())[pair_mask]
            s_layers = torch.stack([(h_norm[l] @ h_norm[l].t())[pair_mask] for l in teacher_layers])
            s_var = s_layers.var(dim=0)
            s8 = s_layers[1]
            s_spread = s_layers.max(dim=0).values - s_layers.min(dim=0).values
            s_gap = (s_layers[0] - s_layers[2]).abs()

            rec_s.append(s_g.cpu().numpy())
            rec_vars.append(s_var.cpu().numpy())
            rec_s8.append(s8.cpu().numpy())
            spread.append(s_spread.cpu().numpy())
            gap12.append(s_gap.cpu().numpy())
            n_batches += 1
            if n_batches % 50 == 0:
                print(f"[conf-diag] {n_batches} batches", flush=True)

    s_g = np.concatenate(rec_s)
    s_var = np.concatenate(rec_vars)
    s8 = np.concatenate(rec_s8)
    s_spread = np.concatenate(spread)
    s_gap12 = np.concatenate(gap12)
    err = np.abs(s8 - s_g)

    np.savez(args.out, s_g=s_g, s_var=s_var, s8=s8, spread=s_spread, gap12=s_gap12)
    print(f"\n[conf-diag] pairs={len(s_g)} saved -> {args.out}")

    def pearson(a, b):
        if a.std() < 1e-12 or b.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return pearson(ra, rb)

    print(f"corr( var_L, |S8-SG| )  pearson={pearson(s_var, err):.4f}  spearman={spearman(s_var, err):.4f}")
    print(f"corr( spread, |S8-SG| ) pearson={pearson(s_spread, err):.4f} spearman={spearman(s_spread, err):.4f}")
    print(f"corr( var_L, |S6-S10| ) pearson={pearson(s_var, s_gap12):.4f}")

    q1, q2 = np.quantile(s_var, [1 / 3, 2 / 3])
    buckets = [("low", s_var <= q1), ("mid", (s_var > q1) & (s_var <= q2)), ("high", s_var > q2)]
    print(f"var quantiles: q33={q1:.6f} q66={q2:.6f}")
    print(f"{'bucket':6s} {'n':>9s} {'mean|S8-SG|':>12s} {'p90|S8-SG|':>11s} {'mean var':>10s}")
    for name, m in buckets:
        print(f"{name:6s} {m.sum():9d} {err[m].mean():12.5f} {np.quantile(err[m], 0.9):11.5f} {s_var[m].mean():10.6f}")


if __name__ == "__main__":
    main()
