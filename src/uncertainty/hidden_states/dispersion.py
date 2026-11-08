"""Question-level hidden-state dispersion."""

import numpy as np


def mean_euclidean_dispersion(vectors: np.ndarray) -> float:
    """Calculate mean distance from the float64 centroid."""
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("vectors must be a non-empty 2D array")
    centroid = values.mean(axis=0, dtype=np.float64)
    return float(np.linalg.norm(values - centroid, axis=1).mean(dtype=np.float64))
