import pytest

from uncertainty.evaluation.correctness import (
    evaluate_low_temp_dataset,
    evaluate_sampled_dataset,
    label_sampled_response,
    match_references,
    normalize_answer,
    score_low_temp_response,
    token_f1,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("The Eiffel Tower!", "eiffel tower"),
        ("  an   APPLE  ", "apple"),
        ("A, B; C.", "b c"),  # leading article removed, punctuation stripped
        ("", ""),
    ],
)
def test_normalize_answer(raw, expected):
    assert normalize_answer(raw) == expected


def test_normalize_does_not_strip_articles_inside_words():
    assert normalize_answer("another theory") == "another theory"


def test_token_f1_exact_partial_and_disjoint():
    assert token_f1("Paris", "paris") == 1.0
    assert token_f1("new york city", "new york") == pytest.approx(0.8)
    assert token_f1("london", "paris") == 0.0


def test_token_f1_empty_edge_cases():
    assert token_f1("", "") == 1.0
    assert token_f1("the", "a") == 1.0  # both normalize to empty
    assert token_f1("", "paris") == 0.0


def test_match_references_exact_match_after_normalization():
    result = match_references("The Beatles.", ["Beatles", "Fab Four"])
    assert result["exact_match"] is True
    assert result["matched_reference"] == "Beatles"
    assert result["token_f1"] == 1.0


def test_match_references_no_references():
    result = match_references("anything", [])
    assert result["exact_match"] is False
    assert result["token_f1"] == 0.0
    assert result["best_f1_reference"] is None


def test_low_temp_score_is_binary_int():
    hit = score_low_temp_response("paris", ["Paris"])
    miss = score_low_temp_response("lyon", ["Paris"])
    assert hit["correct"] == 1 and miss["correct"] == 0
    assert isinstance(hit["correct"], int)


def test_sampled_label_keeps_response_text():
    record = label_sampled_response("Paris", ["paris"])
    assert record["response"] == "Paris"
    assert record["correct"] is True


def test_evaluate_low_temp_dataset_scores_only_the_target_answer():
    dataset = {
        "generations": {
            "q1": {
                "question": "Capital of France?",
                "reference_answers": ["Paris"],
                "low_temp_generation": {"response": "paris"},
                "high_temp_generations": {"responses": [{"response": "Lyon"}]},
            },
            "q2": {
                "question": "Capital of Spain?",
                "reference_answers": ["Madrid"],
                "low_temp_generation": {"response": "Barcelona"},
            },
        }
    }
    result = evaluate_low_temp_dataset(dataset, "dummy.json")
    labels = {q["question_id"]: q["correct"] for q in result["questions"]}
    assert labels == {"q1": 1, "q2": 0}
    assert result["metadata"]["generation"] == "low_temperature"
    assert result["metadata"]["num_questions"] == 2


def test_evaluate_sampled_dataset_scores_every_sample():
    dataset = {
        "questions": {
            "q1": {
                "question": "Capital of France?",
                "reference_answers": ["Paris"],
                "responses": [
                    {"response": "Paris", "index": 0},
                    {"response": "Lyon", "index": 1},
                    {"response": "paris", "index": 2},
                    {"response": "Nice", "index": 3},
                ],
            }
        }
    }
    result = evaluate_sampled_dataset(dataset)
    question = result["questions"]["q1"]
    assert question["exact_match_count"] == 2
    assert question["accuracy"] == 0.5
    assert result["metadata"]["generation"] == "high_temperature_samples"


def test_evaluators_reject_malformed_input():
    with pytest.raises(ValueError):
        evaluate_low_temp_dataset({}, "x")
    with pytest.raises(ValueError):
        evaluate_sampled_dataset({})
    with pytest.raises(ValueError):
        evaluate_low_temp_dataset(
            {"generations": {"q": {"question": "?", "reference_answers": [1]}}}, "x"
        )
