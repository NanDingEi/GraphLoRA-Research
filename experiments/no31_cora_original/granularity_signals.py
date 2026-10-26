"""Explicit coarse/fine collaborative signal extraction for original CoRA.

Coarse: the original entity-level MF pair [user; target].
Fine: target-conditioned user-target and target-history interaction features.
The coarse path is exactly the original CoRA input; the fine path is norm-matched
to it so the comparison is not dominated by input magnitude.
"""
import torch
import torch.nn.functional as F


def build_coarse_fine_signals(
    user_ids,
    target_item_ids,
    history_item_ids,
    user_table,
    item_table,
    padding_idx=0,
):
    """Return (coarse, fine), each [batch, 2 * embedding_dim].

    Args:
        user_ids: user indices, shape [B].
        target_item_ids: candidate item indices, shape [B].
        history_item_ids: padded history indices, shape [B, H]. Padding is
            ignored when constructing the target-conditioned history summary.
        user_table/item_table: frozen MF embedding tables.
        padding_idx: item-table index used for history padding.
    """
    if user_table.ndim != 2 or item_table.ndim != 2:
        raise ValueError("MF embedding tables must be rank-2 tensors")
    if user_table.shape[1] != item_table.shape[1]:
        raise ValueError("User and item MF embedding dimensions must match")

    device = user_table.device
    user_ids = torch.as_tensor(user_ids, device=device, dtype=torch.long).reshape(-1)
    target_item_ids = torch.as_tensor(
        target_item_ids, device=device, dtype=torch.long
    ).reshape(-1)
    history_item_ids = torch.as_tensor(
        history_item_ids, device=device, dtype=torch.long
    )
    if history_item_ids.ndim == 1 and user_ids.numel() == 1:
        history_item_ids = history_item_ids.unsqueeze(0)
    if history_item_ids.ndim != 2:
        raise ValueError("history_item_ids must have shape [batch, history_length]")
    if user_ids.numel() != target_item_ids.numel():
        raise ValueError("User and target batch sizes differ")
    if history_item_ids.shape[0] != user_ids.numel():
        raise ValueError("History and user batch sizes differ")
    if user_ids.numel() == 0:
        raise ValueError("Empty batch is not supported")
    if user_ids.min().item() < 0 or user_ids.max().item() >= user_table.shape[0]:
        raise IndexError("User ID is outside the MF user table")
    if (
        target_item_ids.min().item() < 0
        or target_item_ids.max().item() >= item_table.shape[0]
    ):
        raise IndexError("Target item ID is outside the MF item table")
    if history_item_ids.numel() and (
        history_item_ids.min().item() < 0
        or history_item_ids.max().item() >= item_table.shape[0]
    ):
        raise IndexError("History item ID is outside the MF item table")
    if not 0 <= int(padding_idx) < item_table.shape[0]:
        raise IndexError("Padding index is outside the MF item table")

    user = user_table[user_ids].float()
    target = item_table[target_item_ids].float()

    # Coarse/entity-level signal deliberately preserves the original CoRA input
    # equation and scale: [MF user embedding ; MF target embedding].
    coarse = torch.cat((user, target), dim=-1)

    # Fine/target-conditioned signal: attend to history items by their MF
    # similarity to the current candidate, then represent explicit interactions.
    user_n = F.normalize(user, dim=-1)
    target_n = F.normalize(target, dim=-1)
    history = item_table[history_item_ids].float()
    history_valid = history_item_ids.ne(int(padding_idx))

    if history.shape[1] == 0:
        history_context = torch.zeros_like(target)
    else:
        history_n = F.normalize(history, dim=-1)
        scores = (history_n * target_n.unsqueeze(1)).sum(dim=-1)
        scores = scores.masked_fill(~history_valid, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * history_valid.to(attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        history_context = (attention.unsqueeze(-1) * history).sum(dim=1)

    history_n = F.normalize(history_context, dim=-1)
    user_target_interaction = F.normalize(user_n * target_n, dim=-1)
    target_history_interaction = F.normalize(target_n * history_n, dim=-1)
    fine_unit = F.normalize(
        torch.cat((user_target_interaction, target_history_interaction), dim=-1),
        dim=-1,
    )
    # Match each fine vector's norm to the original [u;i] pair. This keeps
    # coarse-only exactly on the old input path and avoids scale-only effects.
    fine = fine_unit * coarse.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    if not torch.isfinite(coarse).all() or not torch.isfinite(fine).all():
        raise FloatingPointError("Non-finite coarse/fine collaborative signal")
    return coarse, fine
