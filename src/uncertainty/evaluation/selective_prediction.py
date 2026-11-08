"""Selective prediction: rejection-accuracy curve and AURAC.

Convention (matches the README): reject questions in descending uncertainty,
break exact ties by ascending ``question_id``, and integrate retained accuracy
over retained coverage with the trapezoidal rule. Displayed accuracy is
undefined at zero coverage; for integration the zero-coverage endpoint takes
the accuracy at the smallest nonzero coverage (one retained question).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def rejection_order(
    uncertainty: np.ndarray, question_ids: Sequence[str]
) -> np.ndarray:
    """Indices from first rejected to last rejected (ties: ascending ID)."""
    uncertainty = np.asarray(uncertainty, dtype=float)
    if len(uncertainty) != len(question_ids):
        raise ValueError("uncertainty and question_ids must have equal length")
    return np.asarray(
        sorted(
            range(len(uncertainty)),
            key=lambda index: (-float(uncertainty[index]), question_ids[index]),
        ),
        dtype=int,
    )


def rejection_curve(
    correct: np.ndarray,
    rejection_order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return coverage, displayed accuracy, integration accuracy, and AURAC.

    ``rejection_order`` lists question indices from first rejected to last
    rejected.
    """
    correct = np.asarray(correct)
    total_questions = len(correct)
    if total_questions == 0:
        raise ValueError("correct must be non-empty")
    if sorted(np.asarray(rejection_order).tolist()) != list(range(total_questions)):
        raise ValueError("rejection_order must be a permutation of all questions")

    rejected_counts = np.arange(total_questions + 1)
    retained_counts = total_questions - rejected_counts
    coverage = retained_counts / total_questions

    rejected_correct = np.concatenate(
        ([0], np.cumsum(correct[rejection_order], dtype=int))
    )
    retained_correct = int(correct.sum()) - rejected_correct

    displayed_accuracy = np.full(total_questions + 1, np.nan, dtype=float)
    displayed_accuracy[:-1] = retained_correct[:-1] / retained_counts[:-1]
    integration_accuracy = displayed_accuracy.copy()
    integration_accuracy[-1] = displayed_accuracy[-2]

    # Reverse so coverage increases from 0 to 1, as required by trapezoids.
    aurac = float(np.trapezoid(integration_accuracy[::-1], coverage[::-1]))
    return coverage, displayed_accuracy, integration_accuracy, aurac


def aurac_from_scores(
    correct: np.ndarray,
    uncertainty: np.ndarray,
    question_ids: Sequence[str],
) -> float:
    """AURAC for an uncertainty score using the documented tie-break."""
    order = rejection_order(uncertainty, question_ids)
    return rejection_curve(np.asarray(correct), order)[3]
