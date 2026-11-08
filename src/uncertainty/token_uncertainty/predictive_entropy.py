"""Teacher-forced next-token predictive entropy (raw full-vocabulary softmax).

``torch`` is imported lazily inside each function.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from ..utils.alignment import answer_token_positions, build_prompt

if TYPE_CHECKING:  # pragma: no cover
    import torch

validate_offsets = answer_token_positions

__all__ = [
    "CausalModelProtocol",
    "TokenizerProtocol",
    "build_prompt",
    "entropy_from_logits",
    "entropy_from_logits_logsumexp",
    "score_entropy",
    "validate_offsets",
]


class TokenizerProtocol(Protocol):
    eos_token_id: int | None

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    def decode(self, *args: Any, **kwargs: Any) -> str: ...


class CausalModelProtocol(Protocol):
    def eval(self) -> Any: ...

    def parameters(self) -> Any: ...

    def __call__(self, **kwargs: Any) -> Any: ...


def entropy_from_logits(logits: "torch.Tensor") -> "torch.Tensor":
    """Shannon entropy (nats) of the softmax of one vocabulary logit row."""
    import torch

    row = logits.float()
    log_probs = torch.log_softmax(row, dim=-1)
    probabilities = log_probs.exp()
    return -(probabilities * log_probs).sum()


def entropy_from_logits_logsumexp(logits: "torch.Tensor") -> "torch.Tensor":
    """Independent entropy via log-sum-exp: ``logZ - E_p[z]``."""
    import torch

    row = logits.float()
    log_partition = torch.logsumexp(row, dim=-1)
    probabilities = torch.exp(row - log_partition)
    return log_partition - (probabilities * row).sum()


def score_entropy(
    model: CausalModelProtocol,
    tokenizer: TokenizerProtocol,
    prompt: str,
    answer: str,
    expected_positions: list[int] | None = None,
) -> dict[str, Any]:
    """Calculate entropy for each answer token using teacher forcing."""
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
    positions = validate_offsets(offsets, len(prompt), len(answer), input_ids.tolist())
    if expected_positions is not None and positions != expected_positions:
        raise ValueError("Token positions differ from saved teacher-forced scores")

    device = next(model.parameters()).device
    model_inputs = {
        "input_ids": input_ids.unsqueeze(0).to(device),
        "attention_mask": torch.ones_like(input_ids).unsqueeze(0).to(device),
    }
    if any(position <= 0 or position >= input_ids.numel() for position in positions):
        raise ValueError("Answer token position cannot be causally aligned")

    predictor_positions = [position - 1 for position in positions]
    keep_indices = torch.tensor(predictor_positions, dtype=torch.long, device=device)
    model_inputs["logits_to_keep"] = keep_indices
    with torch.inference_mode():
        selected_logits = model(**model_inputs).logits[0].float()
    if selected_logits.shape[0] != len(positions):
        raise ValueError("Selected model logits do not match answer positions")

    entropy_values: list[float] = []
    verification: list[dict[str, Any]] = []
    for sample_index, position in enumerate(positions):
        prediction_row = selected_logits[sample_index]
        entropy = entropy_from_logits(prediction_row)
        if not torch.isfinite(entropy) or entropy < 0:
            raise ValueError("Token entropy is not finite and nonnegative")
        value = float(entropy.detach().cpu())
        entropy_values.append(value)
        if sample_index < 3:
            independent = entropy_from_logits_logsumexp(prediction_row)
            difference = abs(value - float(independent.detach().cpu()))
            if difference > 1e-5:
                raise ValueError("Independent entropy verification failed")
            verification.append(
                {
                    "answer_token_position": position,
                    "predictor_logit_position": position - 1,
                    "target_token_id": int(input_ids[position].cpu()),
                    "entropy_nats": value,
                    "independent_entropy_nats": float(independent.cpu()),
                    "absolute_difference": difference,
                }
            )

    if len(entropy_values) != len(positions):
        raise ValueError("Entropy count does not match answer-token count")
    return {
        "answer_token_count": len(entropy_values),
        "answer_token_positions": positions,
        "token_entropy_nats": entropy_values,
        "mean_token_entropy_nats": float(sum(entropy_values) / len(entropy_values)),
        "std_token_entropy_nats": float(
            torch.tensor(entropy_values, dtype=torch.float64).std(unbiased=False)
        ),
        "maximum_token_entropy_nats": max(entropy_values),
        "causal_shift_verified": True,
        "independent_entropy_verification": verification,
        "eos_included": False,
    }
