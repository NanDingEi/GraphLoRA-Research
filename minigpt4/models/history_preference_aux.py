"""Training-only history-conditioned collaborative preference supervision.

No new modules, inference changes, teacher scores, candidate pairing, or
candidate-history gate. For each CTR row, compare GNN penultimate embedding
of the candidate to embeddings of positively rated history items. The
sample's binary label supervises this structural signal with BCE.

Auxiliary gradients reach the GNN/MF representations that S2 already
injects at layer 31. The LLM remains updated by its original BCE loss.
"""
import torch
import torch.nn.functional as F


def history_target_preference_loss(
    token_gnn, token_node_ids, batch_indices, target_item_ids,
    labels, num_users, similarity_scale=4.0,
):
    if token_gnn.ndim != 2:
        raise ValueError("token_gnn must be [number_of_id_tokens, embedding_dim]")
    if target_item_ids.numel() != labels.numel():
        raise ValueError("candidate IDs and binary CTR labels mismatch")
    if token_gnn.shape[0] != token_node_ids.numel() or (
        token_node_ids.numel() != batch_indices.numel()
    ):
        raise ValueError("ID-token metadata and embeddings mismatch")
    if similarity_scale <= 0:
        raise ValueError("similarity_scale must be positive")

    # Cosine in FP32 for AMP stability and non-trivial auxiliary gradients.
    z = F.normalize(token_gnn.float(), dim=-1)
    per_row_scores, per_row_labels = [], []
    total = int(target_item_ids.numel())
    for row in range(total):
        mask = batch_indices == row
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue
        ids = token_node_ids[idx]
        target_global = target_item_ids[row].long() + int(num_users)
        target_rows = idx[ids == target_global]
        # Only item IDs from the same user's history (not user, candidate,
        # or padding). The graph neighborhood is not in token_node_ids.
        history_rows = idx[(ids >= num_users) & (ids != target_global)]
        if target_rows.numel() == 0 or history_rows.numel() == 0:
            continue

        # Duplicate candidate IDs in history are excluded above. Duplicate
        # history items within the prompt are not artificially reweighted.
        tgt_z = z[target_rows[0]]
        hist_z = z[history_rows]
        mean_similarity = (hist_z * tgt_z).sum(dim=-1).mean()
        per_row_scores.append(mean_similarity * similarity_scale)
        per_row_labels.append(labels[row].float())

    if not per_row_scores:
        return token_gnn.float().sum() * 0.0, 0, total

    scores = torch.stack(per_row_scores)
    targets = torch.stack(per_row_labels).to(scores.device)
    return F.binary_cross_entropy_with_logits(scores, targets), len(per_row_scores), total
