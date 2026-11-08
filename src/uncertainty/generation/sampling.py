"""Sampling-argument resolution for ``model.generate`` (no torch needed)."""
from __future__ import annotations

from typing import Any


def resolve_sampling_kwargs(
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool = True,
) -> dict[str, Any]:
    """Return the sampling kwargs passed to ``model.generate``.

    Sampling is used only when ``do_sample`` is true *and* ``temperature > 0``;
    otherwise decoding is greedy and the sampling parameters are omitted
    (``None``). With the experiment defaults (``do_sample=True``, T = 0.1 and
    T = 1.0) both generations are sampled, exactly as in the original runs.
    """
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    sample = bool(do_sample) and temperature > 0.0
    return {
        "do_sample": sample,
        "temperature": temperature if sample else None,
        "top_p": top_p if sample else None,
        "top_k": top_k if sample else None,
    }
