import math

import pytest

from uncertainty.token_uncertainty.teacher_forcing import (
    cluster_probabilities,
    entropy,
    log_normalize,
)
from uncertainty.utils.alignment import answer_token_positions, build_prompt


def test_build_prompt_format():
    assert build_prompt("FEW\n", "Who?") == "FEW\nQuestion: Who?\nAnswer:"


def test_alignment_matches_validate_offsets_alias():
    from uncertainty.token_uncertainty.predictive_entropy import validate_offsets

    assert validate_offsets is answer_token_positions


def test_log_normalize_sums_to_one_and_is_stable():
    weights = log_normalize([-1000.0, -1001.0, -1002.0])  # would underflow naively
    assert math.isclose(sum(weights), 1.0, rel_tol=1e-12)
    assert weights[0] > weights[1] > weights[2]
    assert weights[0] / weights[1] == pytest.approx(math.e)


def test_log_normalize_uniform():
    assert log_normalize([-3.0, -3.0, -3.0, -3.0]) == pytest.approx([0.25] * 4)


def test_entropy_known_values():
    assert entropy([1.0]) == 0.0
    assert entropy([0.5, 0.5]) == pytest.approx(math.log(2))
    assert entropy([0.5, 0.5, 0.0]) == pytest.approx(math.log(2))  # 0*ln0 := 0


def test_entropy_validates_normalization():
    with pytest.raises(ValueError):
        entropy([0.5, 0.4])
    with pytest.raises(ValueError):
        entropy([])


def test_cluster_probabilities_aggregates_weights():
    record = {"clusters": [{"indices": [0, 2]}, {"indices": [1]}]}
    probabilities = cluster_probabilities([0.2, 0.5, 0.3], record)
    assert probabilities == pytest.approx([0.5, 0.5])


def test_cluster_probabilities_requires_exact_cover():
    with pytest.raises(ValueError, match="exactly once"):
        cluster_probabilities([0.5, 0.5], {"clusters": [{"indices": [0]}]})
    with pytest.raises(ValueError, match="exactly once"):
        cluster_probabilities([0.5, 0.5], {"clusters": [{"indices": [0, 0, 1]}]})


def test_entropy_from_logits_uniform_and_peaked():
    torch = pytest.importorskip("torch")
    from uncertainty.token_uncertainty.predictive_entropy import (
        entropy_from_logits,
        entropy_from_logits_logsumexp,
    )

    uniform = torch.zeros(8)
    assert float(entropy_from_logits(uniform)) == pytest.approx(math.log(8), rel=1e-6)
    peaked = torch.tensor([50.0] + [0.0] * 7)
    assert float(entropy_from_logits(peaked)) == pytest.approx(0.0, abs=1e-6)

    row = torch.tensor([1.5, -0.3, 2.2, 0.0, 0.7])
    assert float(entropy_from_logits(row)) == pytest.approx(
        float(entropy_from_logits_logsumexp(row)), abs=1e-6
    )
    assert float(entropy_from_logits(row)) >= 0.0
    # shift invariance of softmax
    assert float(entropy_from_logits(row + 100.0)) == pytest.approx(
        float(entropy_from_logits(row)), abs=1e-4
    )
