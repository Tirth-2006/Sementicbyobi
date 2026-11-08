"""Independently validate hidden-state feature artifacts."""
import json
from pathlib import Path

import numpy as np


GENERATION_PATH = Path("data/generated/trivia_qa_qwen3_20261024_162539.json")
CORRECTNESS_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_low_temp_correctness.json"
)
FEATURES_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_hidden_state_features.npz"
)
QUESTION_FEATURES_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_hidden_state_question_features.npz"
)
METADATA_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_hidden_state_metadata.json"
)
EXPECTED_QUESTIONS = 100
EXPECTED_ANSWERS = 1000
EXPECTED_POOLING = 4
AGGREGATION_TOLERANCE = 1e-6
EXPECTED_POOLING_LABELS = ("first", "last", "mean", "max")
EXPECTED_FEATURE_AXES = (
    "answer",
    "hidden_state_index",
    "pooling_strategy",
    "hidden_dimension",
)
EXPECTED_LAYER_CONVENTION = (
    "index 0 is the token embedding output; indices 1 through 27 are "
    "outputs of decoder blocks 1 through 27; index 28 is the final "
    "RMSNorm-normalized output after decoder block 28; there are 28 "
    "decoder blocks in total"
)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def main() -> None:
    generation = load_json(GENERATION_PATH)
    correctness = load_json(CORRECTNESS_PATH)
    metadata = load_json(METADATA_PATH)
    generation_questions = generation.get("generations")
    correctness_questions = correctness.get("questions")
    if not isinstance(generation_questions, dict) or len(generation_questions) != EXPECTED_QUESTIONS:
        raise ValueError("Generation artifact must contain 100 questions")
    if not isinstance(correctness_questions, list) or len(correctness_questions) != EXPECTED_QUESTIONS:
        raise ValueError("Correctness artifact must contain 100 questions")
    if metadata.get("num_questions") != EXPECTED_QUESTIONS:
        raise ValueError("Metadata question count is invalid")
    if metadata.get("num_answers") != EXPECTED_ANSWERS:
        raise ValueError("Metadata answer count is invalid")
    if (
        metadata.get("num_hidden_state_indices") != 29
        or metadata.get("hidden_size") != 2048
    ):
        raise ValueError("Metadata hidden-state dimensions are invalid")
    if tuple(metadata.get("pooling_strategies", ())) != EXPECTED_POOLING_LABELS:
        raise ValueError("Metadata pooling labels are invalid or misordered")
    if tuple(metadata.get("feature_axes", ())) != EXPECTED_FEATURE_AXES:
        raise ValueError("Metadata feature axes are invalid")
    if metadata.get("layer_index_convention") != EXPECTED_LAYER_CONVENTION:
        raise ValueError("Metadata layer-index convention is invalid")
    correctness_by_id: dict[str, int] = {}
    for item in correctness_questions:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("question_id"), str)
            or type(item.get("correct")) is not int
            or item["correct"] not in (0, 1)
            or item["question_id"] in correctness_by_id
        ):
            raise ValueError("Correctness records must have unique binary labels")
        correctness_by_id[item["question_id"]] = item["correct"]
    if len(correctness_by_id) != EXPECTED_QUESTIONS:
        raise ValueError("Correctness IDs are duplicated or missing")

    answer_data = np.load(FEATURES_PATH, allow_pickle=True)
    question_data = np.load(QUESTION_FEATURES_PATH, allow_pickle=False)
    answer_features = answer_data["features"]
    answer_ids = answer_data["question_id"].astype(str)
    answer_indices = answer_data["answer_index"]
    answer_positions = answer_data["answer_token_positions"]
    input_token_counts = answer_data["input_token_count"]
    question_features = question_data["features"]
    question_ids = question_data["question_id"].astype(str)
    saved_correct = question_data["correct"]
    if answer_ids.ndim != 1 or answer_indices.ndim != 1:
        raise ValueError("Answer mapping arrays must be one-dimensional")
    if answer_positions.ndim != 1 or input_token_counts.ndim != 1:
        raise ValueError(
            "Answer token positions and input token counts must be one-dimensional"
        )
    if question_ids.ndim != 1 or saved_correct.ndim != 1:
        raise ValueError(
            "Question IDs and correctness arrays must be one-dimensional"
        )

    if answer_features.ndim != 4 or answer_features.shape[0] != EXPECTED_ANSWERS:
        raise ValueError(f"Unexpected per-answer feature shape: {answer_features.shape}")
    if answer_features.shape != (EXPECTED_ANSWERS, 29, EXPECTED_POOLING, 2048):
        raise ValueError(f"Per-answer dimensions disagree with metadata: {answer_features.shape}")
    if answer_features.shape[2] != EXPECTED_POOLING:
        raise ValueError("Per-answer feature artifact lacks four pooling strategies")
    if question_features.shape != (EXPECTED_QUESTIONS, 29, EXPECTED_POOLING, 2048):
        raise ValueError(f"Unexpected question feature shape: {question_features.shape}")
    if metadata.get("answer_feature_shape") != list(answer_features.shape):
        raise ValueError("Metadata answer_feature_shape does not match the array")
    if metadata.get("question_feature_shape") != list(question_features.shape):
        raise ValueError("Metadata question_feature_shape does not match the array")
    if metadata.get("answer_feature_dtype") != str(answer_features.dtype):
        raise ValueError("Metadata answer_feature_dtype does not match the array")
    if metadata.get("question_feature_dtype") != str(question_features.dtype):
        raise ValueError("Metadata question_feature_dtype does not match the array")
    if metadata.get("num_hidden_state_indices") != answer_features.shape[1]:
        raise ValueError(
            "Metadata num_hidden_state_indices does not match the arrays"
        )
    if metadata.get("hidden_size") != answer_features.shape[3]:
        raise ValueError("Metadata hidden_size does not match the arrays")
    if metadata.get("num_questions") != question_features.shape[0]:
        raise ValueError("Metadata num_questions does not match the array")
    if metadata.get("num_answers") != answer_features.shape[0]:
        raise ValueError("Metadata num_answers does not match the array")
    if answer_features.shape[1:] != question_features.shape[1:]:
        raise ValueError("Per-answer and question feature dimensions disagree")
    if len(set(answer_ids)) != EXPECTED_QUESTIONS:
        counts = {qid: int(np.sum(answer_ids == qid)) for qid in set(answer_ids)}
        if any(count != 10 for count in counts.values()):
            raise ValueError("Per-answer question IDs do not occur exactly 10 times")
    if len(set(question_ids)) != EXPECTED_QUESTIONS:
        raise ValueError("Question feature IDs are duplicated")
    if set(question_ids) != set(generation_questions):
        raise ValueError("Question feature IDs do not match generation IDs")
    if set(answer_ids) != set(generation_questions):
        raise ValueError("Answer feature IDs do not match generation IDs")
    for question_id, entry in generation_questions.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{question_id} generation entry is invalid")
        high_temp = entry.get("high_temp_generations")
        responses = (
            high_temp.get("responses")
            if isinstance(high_temp, dict)
            else None
        )
        if not isinstance(responses, list) or len(responses) != 10:
            raise ValueError(
                f"{question_id} generation source must contain exactly 10 answers"
            )
    expected_answer_keys = {
        (question_id, answer_index)
        for question_id, entry in generation_questions.items()
        for answer_index, _ in enumerate(
            entry["high_temp_generations"]["responses"]
        )
    }
    actual_answer_keys = {
        (str(answer_ids[index]), int(answer_indices[index]))
        for index in range(len(answer_ids))
    }
    if actual_answer_keys != expected_answer_keys:
        raise ValueError(
            "Answer-feature rows do not map exactly to generation "
            "(question_id, answer_index) records"
        )
    if (
        len(answer_positions) != EXPECTED_ANSWERS
        or len(answer_indices) != EXPECTED_ANSWERS
        or len(input_token_counts) != EXPECTED_ANSWERS
    ):
        raise ValueError("Per-answer mapping arrays have invalid lengths")
    if len(question_ids) != len(saved_correct):
        raise ValueError("Question IDs and correctness arrays have different lengths")
    expected_correct = np.asarray(
        [correctness_by_id[question_id] for question_id in question_ids],
        dtype=np.int8,
    )
    if not np.array_equal(saved_correct, expected_correct):
        raise ValueError(
            "Question feature correctness labels do not match source labels by "
            "question ID"
        )
    if not np.isfinite(answer_features).all() or not np.isfinite(question_features).all():
        raise ValueError("Feature arrays contain non-finite values")

    for question_id in question_ids:
        indices = np.flatnonzero(answer_ids == question_id)
        if len(indices) != 10:
            raise ValueError(f"{question_id} does not have exactly 10 answers")
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in answer_indices[indices]
        ):
            raise ValueError(f"{question_id} has non-integer answer indices")
        observed_indices = sorted(int(value) for value in answer_indices[indices])
        if observed_indices != list(range(10)):
            raise ValueError(f"{question_id} answer indices are not exactly 0 through 9")
        expected_answers = generation_questions[question_id].get(
            "high_temp_generations", {}
        ).get("responses")
        if not isinstance(expected_answers, list) or len(expected_answers) != 10:
            raise ValueError(f"{question_id} generation answers are invalid")
        observed_records = {
            int(answer_indices[index]): answer_ids[index] for index in indices
        }
        if set(observed_records) != set(range(10)):
            raise ValueError(f"{question_id} answer mapping is incomplete")
        question_index = int(np.flatnonzero(question_ids == question_id)[0])
        rebuilt = answer_features[indices].astype(np.float32).mean(axis=0)
        difference = np.max(np.abs(rebuilt - question_features[question_index]))
        if difference > AGGREGATION_TOLERANCE:
            raise ValueError(
                f"{question_id} aggregation differs by {difference:.3e}, "
                f"above {AGGREGATION_TOLERANCE:.1e}"
            )
        for index in indices:
            positions_raw = answer_positions[index]
            if not isinstance(positions_raw, (list, np.ndarray)):
                raise ValueError(f"{question_id} has invalid answer positions")
            if any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                for value in positions_raw
            ):
                raise ValueError(f"{question_id} has non-integer answer positions")
            positions = [int(value) for value in positions_raw]
            input_count = input_token_counts[index]
            if (
                isinstance(input_count, (bool, np.bool_))
                or not isinstance(input_count, (int, np.integer))
                or input_count <= 0
            ):
                raise ValueError(f"{question_id} has invalid input token count")
            if not positions or positions[0] <= 0 or any(
                current <= previous or current >= input_count
                for previous, current in zip(positions, positions[1:])
            ) or positions[-1] >= input_count:
                raise ValueError(f"{question_id} has invalid causal token positions")
            predictor_positions = [position - 1 for position in positions]
            if any(
                predictor < 0 or predictor >= input_count
                for predictor in predictor_positions
            ):
                raise ValueError(
                    f"{question_id} has invalid causal predictor positions"
                )

    print("Validation passed")
    print(f"Questions: {len(question_ids)}")
    print(f"Answers: {len(answer_ids)}")
    print(f"Per-answer shape: {answer_features.shape}")
    print(f"Question shape: {question_features.shape}")
    print(f"Aggregation tolerance: {AGGREGATION_TOLERANCE:.1e}")
    print("Answer indices: complete 0-9 for every question")
    print("Finite feature values: yes")
    print("Causal predictor positions: valid")
    print(
        "Tokenizer/model recomputation: not performed; this validator checks "
        "stored positions and counts but does not reload the model or tokenizer"
    )


if __name__ == "__main__":
    main()
