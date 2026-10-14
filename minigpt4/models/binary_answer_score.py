"""Binary CTR score from LLM answer-token logits.

The original GraphLoRA implementation uses logit(Yes) as the BCE score.
The optional margin mode instead uses logit(Yes)-logit(No), which is
equivalent to the conditional two-answer log-odds. Both still train with
standard binary cross entropy; no ranking objective or extra parameters.
"""
import torch


def binary_answer_score(logits, answer_position, yes_token_id,
                        no_token_id=None, mode="yes"):
    if mode not in ("yes", "margin"):
        raise ValueError("answer_score_mode must be 'yes' or 'margin'")
    if logits.ndim != 3:
        raise ValueError("Expected output logits [batch, tokens, vocabulary]")
    selected = logits[:, answer_position, :]
    yes_logit = selected[:, yes_token_id]
    if mode == "yes":
        return yes_logit
    if no_token_id is None:
        raise ValueError("margin score requires No token id")
    if int(yes_token_id) == int(no_token_id):
        raise ValueError("Yes and No must correspond to different token IDs")
    no_logit = selected[:, no_token_id]
    # Compute BCE log-odds in FP32 when the Vicuna backbone emits FP16
    # logits; keep the original "yes" branch dtype/path untouched.
    return yes_logit.float() - no_logit.float()
