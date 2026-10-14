"""Loss helpers for per-example weighted supervised fine-tuning."""

from __future__ import annotations

import torch


IGNORE_INDEX = -100


def compute_sample_weighted_sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weight: torch.Tensor,
    sample_weight_mean: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return weighted loss, per-example token-mean CE, and normalized weights."""
    if sample_weight_mean <= 0:
        raise ValueError("sample_weight_mean must be positive")
    shift_logits = logits[..., :-1, :].float().contiguous()
    shift_labels = labels[..., 1:].to(shift_logits.device).contiguous()
    token_loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view_as(shift_labels)
    valid = shift_labels.ne(IGNORE_INDEX)
    valid_count = valid.sum(dim=-1)
    if torch.any(valid_count == 0):
        raise ValueError("sample-weighted SFT received an example without supervised tokens")
    per_example = (token_loss * valid).sum(dim=-1) / valid_count
    weights = sample_weight.to(device=per_example.device, dtype=per_example.dtype).view(-1)
    if weights.numel() != per_example.numel() or torch.any(~torch.isfinite(weights)) or torch.any(weights <= 0):
        raise ValueError("sample_weight must contain one positive finite value per example")
    normalized = weights / float(sample_weight_mean)
    return (per_example * normalized).mean(), per_example, normalized
