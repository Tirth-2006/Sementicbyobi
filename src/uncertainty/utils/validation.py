"""Shared validation helpers."""

import numpy as np


def require_finite(values: np.ndarray, name: str) -> None:
    """Raise when an array contains non-finite values."""
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
