import numpy as np

from uncertainty.evaluation.predictive_probes import make_model, nested_oof


def synthetic(n=60, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    signal = (y * 2 - 1)[:, None] * 1.5
    features = rng.normal(size=(n, 3)) + signal
    return features, y


def test_make_model_has_scaler_then_logistic_regression():
    model = make_model()
    assert [name for name, _ in model.steps] == ["scaler", "classifier"]


def test_nested_oof_shapes_and_range():
    features, y = synthetic()
    scores, predictions, best_c = nested_oof(features, y, n_jobs=1)
    assert scores.shape == predictions.shape == (len(y),)
    assert np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()
    assert set(predictions.tolist()) <= {0, 1}
    assert len(best_c) == 2 and all(c in (0.1, 1.0) for c in best_c)


def test_nested_oof_learns_a_real_signal():
    features, y = synthetic(n=80, seed=1)
    scores, _, _ = nested_oof(features, y, n_jobs=1)
    positives, negatives = scores[y == 1], scores[y == 0]
    assert positives.mean() > negatives.mean() + 0.2


def test_nested_oof_is_deterministic():
    features, y = synthetic(seed=2)
    first = nested_oof(features, y, n_jobs=1)[0]
    second = nested_oof(features, y, n_jobs=1)[0]
    np.testing.assert_allclose(first, second)


def test_nested_oof_threshold_changes_predictions_not_scores():
    features, y = synthetic(seed=3)
    s1, p1, _ = nested_oof(features, y, n_jobs=1, threshold=0.5)
    s2, p2, _ = nested_oof(features, y, n_jobs=1, threshold=0.9)
    np.testing.assert_allclose(s1, s2)
    assert p2.sum() <= p1.sum()
