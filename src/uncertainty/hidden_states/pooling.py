"""Pooling helpers for hidden-state tensors."""

import numpy as np
import torch


def mean_pool_hidden_states(
    hidden_states: torch.Tensor, positions: list[int]
) -> np.ndarray:
    """Return a float32 mean over selected sequence positions."""
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape (sequence, hidden_size)")
    if not positions:
        raise ValueError("At least one position is required")
    return (
        hidden_states[positions].float().mean(dim=0).cpu().numpy().astype(np.float32)
    )
