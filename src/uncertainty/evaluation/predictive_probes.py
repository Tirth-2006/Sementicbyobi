"""Nested cross-validated logistic-regression probes."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_SEED = 20261106
DEFAULT_OUTER_FOLDS = 2
DEFAULT_INNER_FOLDS = 2
DEFAULT_REGULARIZATION_GRID = (0.1, 1.0)
DEFAULT_THRESHOLD = 0.5


def make_model(
    features: np.ndarray | None = None, seed: int = DEFAULT_SEED
) -> Pipeline:
    """StandardScaler + lbfgs logistic regression (scaler fit per training fold)."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(solver="lbfgs", max_iter=1000, random_state=seed),
            ),
        ]
    )


def nested_oof(
    features: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = DEFAULT_SEED,
    outer_folds: int = DEFAULT_OUTER_FOLDS,
    inner_folds: int = DEFAULT_INNER_FOLDS,
    regularization_grid: Sequence[float] = DEFAULT_REGULARIZATION_GRID,
    threshold: float = DEFAULT_THRESHOLD,
    n_jobs: int | None = -1,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Out-of-fold P(correct), thresholded predictions, and best C per fold.

    The inner loop selects ``C`` by AUROC on the training fold only.
    """
    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    scores = np.full(len(y), np.nan, dtype=float)
    best_c_by_fold: list[float] = []
    for fold, (train, test) in enumerate(outer.split(features, y)):
        inner = StratifiedKFold(
            n_splits=inner_folds, shuffle=True, random_state=seed + fold + 1
        )
        search = GridSearchCV(
            make_model(features, seed),
            {"classifier__C": list(regularization_grid)},
            scoring="roc_auc",
            cv=inner,
            refit=True,
            n_jobs=n_jobs,
        )
        search.fit(features[train], y[train])
        scores[test] = search.predict_proba(features[test])[:, 1]
        best_c_by_fold.append(float(search.best_params_["classifier__C"]))
    if not np.isfinite(scores).all():
        raise ValueError("Nested evaluation did not produce all OOF scores")
    predictions = (scores >= threshold).astype(int)
    return scores, predictions, best_c_by_fold
