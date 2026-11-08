import numpy as np
import pytest

from uncertainty.evaluation.metrics import aurac, metric_values
from uncertainty.evaluation.selective_prediction import (
    aurac_from_scores,
    rejection_curve,
    rejection_order,
)


def test_rejection_order_descending_uncertainty_with_id_tiebreak():
    uncertainty = np.array([0.2, 0.9, 0.9, 0.1])
    ids = ["d", "c", "b", "a"]
    order = rejection_order(uncertainty, ids)
    # ties at 0.9 -> ascending question_id: "b" (index 2) before "c" (index 1)
    assert order.tolist() == [2, 1, 0, 3]


def test_rejection_curve_perfect_ranking():
    correct = np.array([1, 1, 0, 0])
    uncertainty = np.array([0.1, 0.2, 0.8, 0.9])
    order = rejection_order(uncertainty, list("abcd"))
    coverage, shown, integ, area = rejection_curve(correct, order)
    assert coverage.tolist() == [1.0, 0.75, 0.5, 0.25, 0.0]
    assert shown[:-1].tolist() == [0.5, 2 / 3, 1.0, 1.0]
    assert np.isnan(shown[-1])  # accuracy undefined at zero coverage
    assert integ[-1] == integ[-2] == 1.0  # documented zero-coverage handling
    assert 0.5 < area <= 1.0


def test_perfect_ranking_beats_reversed_ranking():
    correct = np.array([1, 1, 1, 0, 0, 0])
    ids = [f"q{i}" for i in range(6)]
    good = aurac_from_scores(correct, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]), ids)
    bad = aurac_from_scores(correct, np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1]), ids)
    assert good > bad


def test_all_correct_gives_aurac_one():
    correct = np.ones(5, dtype=int)
    assert aurac_from_scores(correct, np.arange(5.0), list("abcde")) == pytest.approx(1.0)


def test_aurac_metric_wrapper_matches_curve():
    rng = np.random.default_rng(0)
    correct = rng.integers(0, 2, 30)
    uncertainty = rng.random(30)
    ids = [f"q{i:02d}" for i in range(30)]
    order = rejection_order(uncertainty, ids)
    assert aurac(correct, uncertainty, ids) == rejection_curve(correct, order)[3]


def test_rejection_curve_validates_order():
    with pytest.raises(ValueError):
        rejection_curve(np.array([1, 0, 1]), np.array([0, 0, 1]))
    with pytest.raises(ValueError):
        rejection_curve(np.array([]), np.array([], dtype=int))


def test_metric_values_perfect_separation():
    y = np.array([1, 1, 1, 0, 0, 0])
    uncertainty = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    prediction = (uncertainty < 0.5).astype(int)
    metrics = metric_values(y, uncertainty, prediction, list("abcdef"))
    assert metrics["auroc_incorrect"] == 1.0
    assert metrics["spearman_uncertainty_vs_correct"] < -0.8
    assert metrics["accuracy_at_0_5"] == 1.0
    assert metrics["f1_at_0_5"] == 1.0


def test_metric_values_requires_both_classes():
    with pytest.raises(ValueError, match="Both correctness classes"):
        metric_values(np.ones(4, dtype=int), np.arange(4.0), np.ones(4, dtype=int), list("abcd"))


def test_metric_values_zero_division_is_zero_not_error():
    y = np.array([1, 0, 1, 0])
    metrics = metric_values(y, np.array([0.1, 0.9, 0.2, 0.8]), np.zeros(4, dtype=int), list("abcd"))
    assert metrics["precision_at_0_5"] == 0.0
    assert metrics["f1_at_0_5"] == 0.0
