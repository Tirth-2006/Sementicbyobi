import numpy as np
import pytest

from uncertainty.hidden_states.dispersion import mean_euclidean_dispersion
from uncertainty.hidden_states.extraction import POOLING_STRATEGIES, answer_positions
from uncertainty.utils.validation import require_finite


def test_dispersion_zero_for_identical_vectors():
    vectors = np.tile(np.array([1.0, -2.0, 3.0]), (10, 1))
    assert mean_euclidean_dispersion(vectors) == 0.0


def test_dispersion_known_value():
    # centroid (0, 0); distances all 1 -> mean 1
    vectors = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    assert mean_euclidean_dispersion(vectors) == pytest.approx(1.0)


def test_dispersion_is_mean_distance_not_variance():
    vectors = np.array([[2.0], [-2.0]])  # distances 2 and 2; variance would be 4
    assert mean_euclidean_dispersion(vectors) == pytest.approx(2.0)


def test_dispersion_uses_float64_for_float16_input():
    vectors = np.array([[1000.0, 0.0], [1002.0, 0.0]], dtype=np.float16)
    assert mean_euclidean_dispersion(vectors) == pytest.approx(1.0, rel=1e-2)


@pytest.mark.parametrize("bad", [np.zeros((0, 3)), np.zeros(3), np.zeros((2, 2, 2))])
def test_dispersion_validates_shape(bad):
    with pytest.raises(ValueError):
        mean_euclidean_dispersion(bad)


def test_require_finite():
    require_finite(np.array([1.0, 2.0]), "ok")
    with pytest.raises(ValueError, match="bad"):
        require_finite(np.array([1.0, np.nan]), "bad")
    with pytest.raises(ValueError):
        require_finite(np.array([np.inf]), "bad")


def test_pooling_strategy_order_is_documented_order():
    assert POOLING_STRATEGIES == ("first", "last", "mean", "max")


def test_answer_positions_includes_boundary_crossing_token():
    # prompt "Q:\nAnswer:" (10 chars) + answer " Paris" (6 chars)
    offsets = [[0, 0], [0, 2], [2, 10], [9, 12], [12, 16]]  # token 3 straddles boundary
    ids = [1, 2, 3, 4, 5]
    assert answer_positions(offsets, 10, 6, ids) == [3, 4]


def test_answer_positions_skips_empty_special_tokens():
    offsets = [[0, 0], [0, 5], [5, 5], [5, 9]]
    assert answer_positions(offsets, 5, 4, [1, 2, 3, 4]) == [3]


def test_answer_positions_errors():
    with pytest.raises(ValueError, match="No answer tokens"):
        answer_positions([[0, 2], [2, 4]], 10, 5, [1, 2])
    with pytest.raises(ValueError, match="no preceding"):
        answer_positions([[0, 3], [3, 6]], 0, 6, [1, 2])  # answer starts at token 0
    with pytest.raises(ValueError, match="missing"):
        answer_positions([[0, 2], [2, 6]], 2, 4, [1, None])


def test_pool_hidden_states_shapes_and_values():
    torch = pytest.importorskip("torch")
    from uncertainty.hidden_states.extraction import pool_hidden_states

    layer0 = torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(1, 5, 4)
    layer1 = -layer0
    pooled = pool_hidden_states((layer0, layer1), [1, 2, 3])
    assert pooled.shape == (2, 4, 4)  # layers, pooling, hidden
    assert pooled.dtype == np.float16
    np.testing.assert_allclose(pooled[0, 0], layer0[0, 1].numpy())  # first
    np.testing.assert_allclose(pooled[0, 1], layer0[0, 3].numpy())  # last
    np.testing.assert_allclose(pooled[0, 2], layer0[0, 1:4].mean(0).numpy())  # mean
    np.testing.assert_allclose(pooled[0, 3], layer0[0, 3].numpy())  # max (monotone data)


def test_mean_pool_helper_validation():
    torch = pytest.importorskip("torch")
    from uncertainty.hidden_states.pooling import mean_pool_hidden_states

    hidden = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    out = mean_pool_hidden_states(hidden, [0, 2])
    np.testing.assert_allclose(out, np.array([3.0, 4.0, 5.0], dtype=np.float32))
    assert out.dtype == np.float32
    with pytest.raises(ValueError):
        mean_pool_hidden_states(hidden, [])
    with pytest.raises(ValueError):
        mean_pool_hidden_states(hidden.reshape(1, 4, 3), [0])
