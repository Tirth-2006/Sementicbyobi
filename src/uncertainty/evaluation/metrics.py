"""Question-level ranking metrics for uncertainty scores."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .selective_prediction import aurac_from_scores


def aurac(
    correct: np.ndarray, uncertainty: np.ndarray, question_ids: Sequence[str]
) -> float:
    """Area under the retained-accuracy / coverage curve."""
    return aurac_from_scores(correct, uncertainty, question_ids)


def metric_values(
    y: np.ndarray,
    uncertainty: np.ndarray,
    prediction: np.ndarray,
    question_ids: Sequence[str],
) -> dict[str, float]:
    """Ranking and thresholded metrics.

    AUROC treats *incorrectness* as the positive class; Spearman is computed
    against correctness (1 = correct), so useful uncertainty is negative.
    """
    y = np.asarray(y)
    uncertainty = np.asarray(uncertainty, dtype=float)
    incorrect = 1 - y
    if np.unique(incorrect).size < 2:
        raise ValueError("Both correctness classes are required for AUROC")
    spearman = spearmanr(uncertainty, y).statistic
    return {
        "auroc_incorrect": float(roc_auc_score(incorrect, uncertainty)),
        "aurac": aurac(y, uncertainty, question_ids),
        "spearman_uncertainty_vs_correct": float(spearman),
        "accuracy_at_0_5": float(accuracy_score(y, prediction)),
        "precision_at_0_5": float(precision_score(y, prediction, zero_division=0)),
        "recall_at_0_5": float(recall_score(y, prediction, zero_division=0)),
        "f1_at_0_5": float(f1_score(y, prediction, zero_division=0)),
    }
