"""Hidden-state extraction for saved answers (causal predictor-state, ``p-1``).

``torch`` is imported lazily so that pure helpers such as
:func:`answer_positions` work without it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from ..utils.alignment import answer_token_positions, build_prompt

if TYPE_CHECKING:  # pragma: no cover
    import torch

POOLING_STRATEGIES = ("first", "last", "mean", "max")

answer_positions = answer_token_positions

__all__ = [
    "POOLING_STRATEGIES",
    "CausalModelProtocol",
    "TokenizerProtocol",
    "answer_positions",
    "build_prompt",
    "extract_record",
    "pool_hidden_states",
]


class CausalModelProtocol(Protocol):
    config: Any

    def eval(self) -> Any: ...

    def parameters(self) -> Any: ...

    def __call__(self, **kwargs: Any) -> Any: ...


class TokenizerProtocol(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


def pool_hidden_states(
    hidden_states: "tuple[torch.Tensor, ...]",
    predictor_positions: list[int],
) -> np.ndarray:
    """Pool every returned layer over causal predictor positions.

    Returns float16 of shape ``(layers, 4, hidden)`` with pooling order
    ``first, last, mean, max``.
    """
    import torch

    pooled_layers: list[np.ndarray] = []
    for layer in hidden_states:
        selected = layer[0, predictor_positions, :].float()
        pooled = torch.stack(
            (
                selected[0],
                selected[-1],
                selected.mean(dim=0),
                selected.max(dim=0).values,
            )
        )
        pooled_layers.append(pooled.cpu().numpy())
    return np.stack(pooled_layers, axis=0).astype(np.float16, copy=False)


def extract_record(
    model: CausalModelProtocol,
    tokenizer: TokenizerProtocol,
    prompt: str,
    answer: str,
) -> tuple[np.ndarray, list[int], int]:
    """Return pooled features, answer token positions, and input length."""
    import torch

    full_text = prompt + answer
    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"][0]
    offsets = encoded["offset_mapping"][0].tolist()
    positions = answer_positions(offsets, len(prompt), len(answer), input_ids.tolist())
    predictor_positions = [position - 1 for position in positions]
    if any(position < 0 for position in predictor_positions):
        raise ValueError("Causal predictor position cannot be negative")
    device = next(model.parameters()).device
    model_inputs = {
        "input_ids": input_ids.unsqueeze(0).to(device),
        "attention_mask": torch.ones_like(input_ids).unsqueeze(0).to(device),
    }
    with torch.inference_mode():
        output = model(**model_inputs, output_hidden_states=True, use_cache=False)
    hidden_states = output.hidden_states
    if not isinstance(hidden_states, tuple) or not hidden_states:
        raise ValueError("Model did not return hidden states")
    features = pool_hidden_states(hidden_states, predictor_positions)
    del output, hidden_states, model_inputs
    if not np.isfinite(features).all():
        raise ValueError("Hidden-state features contain non-finite values")
    return features, positions, int(input_ids.numel())
