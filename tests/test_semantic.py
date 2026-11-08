import math

import pytest

from uncertainty.semantic.clustering import (
    ENTAILMENT_THRESHOLD,
    cluster_answers,
    compare_answers,
)
from uncertainty.semantic.semantic_entropy import calculate_entropy


def make_nli(equivalent_pairs, score=0.99, label_for_other="contradiction"):
    """Mock NLI: answers are 'equivalent' when they share a group id.

    ``equivalent_pairs`` maps answer text -> group id. The mock inspects the
    ``Answer: ...`` line of premise and hypothesis.
    """

    def answer_of(text):
        return text.rsplit("Answer: ", 1)[1]

    def nli(premise, text_pair):
        same = equivalent_pairs[answer_of(premise)] == equivalent_pairs[answer_of(text_pair)]
        label = "entailment" if same else label_for_other
        return [{"label": label, "score": score}]

    return nli


def test_calculate_entropy_known_values():
    assert calculate_entropy([[0, 1, 2, 3]], 4) == 0.0
    assert calculate_entropy([[0, 1], [2, 3]], 4) == pytest.approx(math.log(2))
    assert calculate_entropy([[i] for i in range(10)], 10) == pytest.approx(math.log(10))
    # frequencies 0.5, 0.25, 0.25
    expected = -(0.5 * math.log(0.5) + 2 * 0.25 * math.log(0.25))
    assert calculate_entropy([[0, 1], [2], [3]], 4) == pytest.approx(expected)


def test_calculate_entropy_validates_coverage():
    with pytest.raises(ValueError):
        calculate_entropy([[0, 1]], 3)
    with pytest.raises(ValueError):
        calculate_entropy([], 0)


def test_compare_answers_requires_both_directions_and_threshold():
    entail = lambda p, text_pair: [{"label": "entailment", "score": 0.9}]
    ok, ab, ba = compare_answers(entail, "q", "a", "b")
    assert ok and ab["label"] == "entailment" and ba["label"] == "entailment"

    low = lambda p, text_pair: [{"label": "entailment", "score": ENTAILMENT_THRESHOLD - 0.01}]
    assert compare_answers(low, "q", "a", "b")[0] is False

    boundary = lambda p, text_pair: [{"label": "ENTAILMENT", "score": ENTAILMENT_THRESHOLD}]
    assert compare_answers(boundary, "q", "a", "b")[0] is True  # >= and case-insensitive

    def one_way(premise, text_pair):
        label = "entailment" if premise.endswith("Answer: a") else "neutral"
        return [{"label": label, "score": 0.99}]

    assert compare_answers(one_way, "q", "a", "b")[0] is False


def test_cluster_answers_groups_equivalent_answers():
    groups = {"Paris": 0, "paris": 0, "Lyon": 1, "Marseille": 2, "Lyon city": 1}
    answers = ["Paris", "Lyon", "paris", "Marseille", "Lyon city"]
    clusters, comparisons = cluster_answers(make_nli(groups), "Capital?", answers)
    assert clusters == [[0, 2], [1, 4], [3]]
    assert len(comparisons) == 10  # C(5, 2) unordered pairs
    assert calculate_entropy(clusters, len(answers)) > 0


def test_cluster_answers_all_equivalent_gives_zero_entropy():
    groups = {"a": 0, "b": 0, "c": 0}
    clusters, _ = cluster_answers(make_nli(groups), "q", ["a", "b", "c"])
    assert clusters == [[0, 1, 2]]
    assert calculate_entropy(clusters, 3) == 0.0


def test_cluster_answers_all_distinct_gives_max_entropy():
    groups = {"a": 0, "b": 1, "c": 2, "d": 3}
    clusters, _ = cluster_answers(make_nli(groups), "q", list("abcd"))
    assert clusters == [[0], [1], [2], [3]]
    assert calculate_entropy(clusters, 4) == pytest.approx(math.log(4))


def test_complete_linkage_blocks_non_transitive_merge():
    """b~a and b~c but a!~c: complete linkage must not put all three together."""
    relation = {
        frozenset("ab"): True,
        frozenset("bc"): True,
        frozenset("ac"): False,
    }

    def nli(premise, text_pair):
        x = premise.rsplit("Answer: ", 1)[1]
        y = text_pair.rsplit("Answer: ", 1)[1]
        entail = x == y or relation[frozenset((x, y))]
        return [{"label": "entailment" if entail else "contradiction", "score": 0.9}]

    clusters, _ = cluster_answers(nli, "q", ["a", "b", "c"])
    # greedy first-compatible: a -> [a]; b joins [a]; c fails vs a -> new cluster
    assert clusters == [[0, 1], [2]]


def test_cluster_answers_is_order_dependent():
    relation = {
        frozenset("ab"): True,
        frozenset("bc"): True,
        frozenset("ac"): False,
    }

    def nli(premise, text_pair):
        x = premise.rsplit("Answer: ", 1)[1]
        y = text_pair.rsplit("Answer: ", 1)[1]
        entail = x == y or relation[frozenset((x, y))]
        return [{"label": "entailment" if entail else "contradiction", "score": 0.9}]

    first, _ = cluster_answers(nli, "q", ["a", "b", "c"])
    second, _ = cluster_answers(nli, "q", ["a", "c", "b"])
    assert first != second  # documented limitation of greedy clustering
