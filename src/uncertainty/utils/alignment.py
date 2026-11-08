"""Token/character alignment helpers shared by token- and hidden-state scoring."""
from __future__ import annotations

from typing import Sequence


def answer_token_positions(
    offsets: Sequence[Sequence[int]],
    prompt_length: int,
    answer_length: int,
    token_ids: Sequence[int | None],
) -> list[int]:
    """Return token indices overlapping the answer character span.

    ``offsets`` are (start, end) character offsets for the jointly tokenized
    ``prompt + answer`` text. Tokens that straddle the prompt/answer boundary
    are included. Empty (special) tokens are skipped.

    Raises ``ValueError`` when no answer token is found, when the first
    answer token is at index 0 (no preceding causal state/logit), or when a
    selected token id is missing.
    """
    answer_end = prompt_length + answer_length
    positions = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and start < answer_end and end > prompt_length
    ]
    if not positions:
        raise ValueError("No answer tokens found from offset mapping")
    if positions[0] == 0:
        raise ValueError("First answer token has no preceding causal state")
    if any(token_ids[index] is None for index in positions):
        raise ValueError("Answer token ID is missing")
    return positions


def build_prompt(few_shot_prompt: str, question: str) -> str:
    """Reconstruct the raw-text prompt used by the teacher-forced scripts."""
    return f"{few_shot_prompt}Question: {question}\nAnswer:"
