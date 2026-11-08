"""Teacher-forced sequence log-likelihood scoring and cluster aggregation.

``torch`` is imported lazily inside the functions that need it.
"""
from __future__ import annotations

import math
from typing import Any

from ..utils.alignment import answer_token_positions, build_prompt
from .predictive_entropy import CausalModelProtocol, TokenizerProtocol

EOS_TOKEN_ID = 151645
validate_offsets = answer_token_positions

__all__ = [
    "EOS_TOKEN_ID",
    "build_prompt",
    "cluster_probabilities",
    "entropy",
    "log_normalize",
    "score_text",
    "validate_offsets",
]


def score_text(
    model: CausalModelProtocol,
    tokenizer: TokenizerProtocol,
    prompt: str,
    answer: str,
    include_eos: bool,
) -> dict[str, Any]:
    """Score answer tokens under the raw model distribution."""
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
    answer_positions = validate_offsets(
        offsets, len(prompt), len(answer), input_ids.tolist()
    )

    if include_eos:
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("Tokenizer has no EOS token")
        input_ids = torch.cat([input_ids, torch.tensor([eos_id], dtype=input_ids.dtype)])
        attention_mask = torch.ones_like(input_ids)
        score_positions = answer_positions + [len(input_ids) - 1]
    else:
        attention_mask = torch.ones_like(input_ids)
        score_positions = answer_positions

    device = next(model.parameters()).device
    model_inputs = {
        "input_ids": input_ids.unsqueeze(0).to(device),
        "attention_mask": attention_mask.unsqueeze(0).to(device),
    }
    with torch.inference_mode():
        logits = model(**model_inputs).logits[0].float()

    labels = model_inputs["input_ids"][0]
    log_probs = torch.log_softmax(logits[:-1], dim=-1)
    selected_log_probs = torch.stack(
        [log_probs[position - 1, labels[position]] for position in score_positions]
    )
    if not torch.isfinite(selected_log_probs).all():
        raise ValueError("Non-finite answer token log probability")

    values = selected_log_probs.detach().cpu().tolist()
    return {
        "token_count": len(values),
        "token_log_probabilities": values,
        "raw_sequence_log_likelihood": float(sum(values)),
        "length_normalized_log_likelihood": float(sum(values) / len(values)),
        "answer_token_positions": score_positions,
        "input_token_count": int(input_ids.numel()),
        "eos_included": include_eos,
    }


def log_normalize(scores: list[float]) -> list[float]:
    """Normalize log scores into weights with a stable log-sum-exp."""
    max_score = max(scores)
    shifted = [math.exp(score - max_score) for score in scores]
    total = math.fsum(shifted)
    values = [value / total for value in shifted]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite normalized answer weight")
    if not math.isclose(math.fsum(values), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Normalized answer weights do not sum to one")
    return values


def entropy(probabilities: list[float]) -> float:
    """Shannon entropy (nats) from normalized cluster probabilities."""
    if not probabilities or not math.isclose(
        sum(probabilities), 1.0, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("Cluster probabilities must sum to one")
    value = -sum(p * math.log(p) for p in probabilities if p > 0)
    if not math.isfinite(value) or value < -1e-12:
        raise ValueError("Entropy must be finite and non-negative")
    return max(0.0, value)


def cluster_probabilities(
    weights: list[float],
    cluster_record: dict[str, Any],
) -> list[float]:
    """Aggregate normalized answer weights into saved semantic clusters."""
    probabilities = []
    assigned: list[int] = []
    for cluster in cluster_record["clusters"]:
        indices = cluster["indices"]
        probabilities.append(sum(weights[index] for index in indices))
        assigned.extend(indices)
    if sorted(assigned) != list(range(len(weights))):
        raise ValueError("Saved clusters do not cover each answer exactly once")
    total = sum(probabilities)
    if not math.isclose(total, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Cluster probabilities do not sum to one")
    return probabilities
