"""D1: role-aware collaborative injection without updating MF/NGCF.

All helpers are pure torch, for fast CPU tests without loading Vicuna.
"""
import torch
import torch.nn.functional as F


def decision_position(prompt_length, answer_length, full_length):
    """Index of the prompt-final logit used by logits[:, -(A+1), :]."""
    p, a, t = int(prompt_length), int(answer_length), int(full_length)
    if p < 1 or a < 1 or t != p + a:
        raise ValueError("D1 invalid prompt/answer split for Yes-logit decision")
    position = t - (a + 1)
    if position != p - 1:
        raise AssertionError("D1 decision index not aligned with BCE logit")
    return position


def build_decision_pair_features(
    batch_indices, local_node_ids, token_node_ids,
    user_ids, target_item_ids, num_users, gcn_output
):
    """[B,3*r] frozen GNN user/candidate features; strict ID checks.

    Different prompt layouts may permute token positions. Select exact graph
    IDs, not 'first item' or 'last item', so the target can never silently
    become a historical item. Missing IDs raise instead of padding with zero.
    """
    batch_size = user_ids.shape[0]
    if any(t.ndim != 1 for t in (batch_indices, local_node_ids, token_node_ids)):
        raise ValueError("D1 expected 1D token maps")
    if not (batch_indices.numel() == local_node_ids.numel() ==
            token_node_ids.numel()):
        raise ValueError("D1 token metadata lengths do not match")
    if target_item_ids.shape[0] != batch_size:
        raise ValueError("D1 user/item batch size mismatch")
    z_user = []
    z_item = []
    for b in range(batch_size):
        match_u = ((batch_indices == b) &
                   (token_node_ids == user_ids[b]) & (local_node_ids >= 0))
        match_i = ((batch_indices == b) &
                   (token_node_ids == target_item_ids[b] + num_users) &
                   (local_node_ids >= 0))
        if match_u.sum().item() != 1 or match_i.sum().item() != 1:
            raise ValueError(
                "D1 missing/ambiguous user or candidate ID tokens for row {}".format(b)
            )
        z_user.append(gcn_output[local_node_ids[match_u][0]])
        z_item.append(gcn_output[local_node_ids[match_i][0]])
    zu = F.normalize(torch.stack(z_user).float(), dim=-1)
    zi = F.normalize(torch.stack(z_item).float(), dim=-1)
    return torch.cat((zu, zi, zu * zi), dim=-1).detach()


def inject_decision_query(bottleneck, pair, position, scale):
    """Update ONLY the final prompt token's Q bottleneck, all other Q rows untouched."""
    if bottleneck.ndim != 3 or pair.ndim != 2:
        raise ValueError("D1 expected bottleneck [B,T,r] and pair [B,r]")
    b, t, r = bottleneck.shape
    if pair.shape != (b, r) or position < 0 or position >= t:
        raise ValueError("D1 decision-pair projection shape or position invalid")
    # Clone prevents in-place alias complications with PEFT/autograd.
    result = bottleneck.clone()
    result[:, position, :] = result[:, position, :] + pair.to(
        dtype=result.dtype, device=result.device
    ) * scale
    return result
