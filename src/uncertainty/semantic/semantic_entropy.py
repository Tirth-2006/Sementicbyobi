"""Empirical (frequency-based) semantic-cluster entropy."""
from __future__ import annotations

import math
from typing import List


def calculate_entropy(clusters: List[List[int]], total_answers: int) -> float:
    """Calculate Shannon entropy (nats) over semantic-cluster frequencies.

    This is *empirical* semantic entropy: cluster probabilities are observed
    sample frequencies, not summed model sequence probabilities.
    """
    if total_answers <= 0:
        raise ValueError("total_answers must be positive")
    if sum(len(cluster) for cluster in clusters) != total_answers:
        raise ValueError("Clusters must cover each answer exactly once")
    entropy = 0.0
    for cluster in clusters:
        probability = len(cluster) / total_answers
        entropy -= probability * math.log(probability)
    return entropy
