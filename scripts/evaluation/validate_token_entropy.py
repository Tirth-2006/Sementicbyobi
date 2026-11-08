"""Independently validate saved Phase 3 token entropy artifacts."""
import csv
import json
import math
from pathlib import Path
from typing import Any


OUTPUT_JSON = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_token_entropy.json"
)
OUTPUT_CSV = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_token_entropy_question.csv"
)
LOW_TEMP_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_low_temp_correctness.json"
)
BOUNDARY_PATH = Path("data/generated/token_boundary_audit.json")
EXPECTED_QUESTIONS = 100
EXPECTED_ANSWERS = 1000
TOLERANCE = 1e-6


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def finite_nonnegative(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=TOLERANCE, abs_tol=TOLERANCE)


def main() -> None:
    output = load_json(OUTPUT_JSON)
    low_temp = load_json(LOW_TEMP_PATH)
    boundary = load_json(BOUNDARY_PATH)
    answers = output.get("answers")
    if not isinstance(answers, list) or len(answers) != EXPECTED_ANSWERS:
        raise ValueError("JSON must contain exactly 1,000 answer records")

    low_temp_questions = low_temp.get("questions")
    if not isinstance(low_temp_questions, list) or len(low_temp_questions) != EXPECTED_QUESTIONS:
        raise ValueError("Low-temperature file must contain 100 questions")
    correctness: dict[str, int] = {}
    for question in low_temp_questions:
        if not isinstance(question, dict):
            raise ValueError("Invalid low-temperature question record")
        question_id = question.get("question_id")
        correct = question.get("correct")
        if not isinstance(question_id, str) or correct not in {0, 1}:
            raise ValueError("Invalid low-temperature correctness record")
        if question_id in correctness:
            raise ValueError(f"Duplicate low-temperature question ID: {question_id}")
        correctness[question_id] = int(correct)

    answer_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    question_means: dict[str, list[float]] = {}
    checks = {
        "answer_vectors": True,
        "answer_statistics": True,
        "question_statistics": True,
        "correctness_labels": True,
        "entropy_values": True,
    }
    for answer in answers:
        if not isinstance(answer, dict):
            raise ValueError("Invalid answer record")
        question_id = answer.get("question_id")
        answer_index = answer.get("answer_index")
        if not isinstance(question_id, str) or type(answer_index) is not int:
            raise ValueError("Answer record lacks valid question ID or index")
        key = (question_id, answer_index)
        if key in answer_by_key:
            raise ValueError(f"Duplicate answer record: {key}")
        if not 0 <= answer_index < 10:
            raise ValueError(f"Answer index out of range: {key}")
        answer_by_key[key] = answer

        vector = answer.get("token_entropy_nats")
        if not isinstance(vector, list) or not vector:
            checks["answer_vectors"] = False
            raise ValueError(f"Empty token entropy vector: {key}")
        positions = answer.get("answer_token_positions")
        if (
            not isinstance(positions, list)
            or len(positions) != len(vector)
            or answer.get("answer_token_count") != len(vector)
            or any(type(position) is not int or position <= 0 for position in positions)
        ):
            checks["answer_vectors"] = False
            raise ValueError(f"Answer token count/positions mismatch: {key}")
        if not all(finite_nonnegative(value) for value in vector):
            checks["entropy_values"] = False
            raise ValueError(f"Invalid entropy value: {key}")
        mean = sum(vector) / len(vector)
        variance = sum((value - mean) ** 2 for value in vector) / len(vector)
        standard_deviation = math.sqrt(variance)
        maximum = max(vector)
        if not (
            close(float(answer.get("mean_token_entropy_nats")), mean)
            and close(float(answer.get("std_token_entropy_nats")), standard_deviation)
            and close(float(answer.get("maximum_token_entropy_nats")), maximum)
        ):
            checks["answer_statistics"] = False
            raise ValueError(f"Answer statistics mismatch: {key}")
        question_means.setdefault(question_id, []).append(mean)
        if answer.get("eos_included") is not False:
            raise ValueError(f"EOS handling mismatch: {key}")

    if len({question_id for question_id, _ in answer_by_key}) != EXPECTED_QUESTIONS:
        raise ValueError("JSON does not contain exactly 100 question IDs")
    for question_id in correctness:
        if len(question_means.get(question_id, [])) != 10:
            raise ValueError(f"Question does not contain exactly 10 answers: {question_id}")
        if not all(
            (question_id, index) in answer_by_key for index in range(10)
        ):
            raise ValueError(f"Missing answer index for {question_id}")
    if set(question_means) != set(correctness):
        checks["correctness_labels"] = False
        raise ValueError("Question IDs do not match low-temperature labels")

    rows_by_question: dict[str, dict[str, str]] = {}
    with OUTPUT_CSV.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required_fields = {
            "question_id",
            "mean_token_entropy",
            "std_token_entropy",
            "correct",
        }
        if reader.fieldnames is None or set(reader.fieldnames) != required_fields:
            raise ValueError("CSV columns do not match the required schema")
        csv_rows = list(reader)
    if len(csv_rows) != EXPECTED_QUESTIONS:
        raise ValueError("CSV must contain exactly 100 rows")
    for row in csv_rows:
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or question_id in rows_by_question:
            raise ValueError("CSV has duplicate or invalid question IDs")
        rows_by_question[question_id] = row
        means = question_means.get(question_id)
        if means is None:
            raise ValueError(f"CSV has unknown question ID: {question_id}")
        expected_mean = sum(means) / 10
        expected_std = math.sqrt(
            sum((value - expected_mean) ** 2 for value in means) / 10
        )
        if not close(float(row["mean_token_entropy"]), expected_mean):
            checks["question_statistics"] = False
            raise ValueError(f"Question mean mismatch: {question_id}")
        if not close(float(row["std_token_entropy"]), expected_std):
            checks["question_statistics"] = False
            raise ValueError(f"Question standard deviation mismatch: {question_id}")
        if row["correct"] != str(correctness[question_id]):
            checks["correctness_labels"] = False
            raise ValueError(f"Correctness mismatch: {question_id}")
    if set(rows_by_question) != set(question_means):
        raise ValueError("CSV question IDs do not match JSON question IDs")

    boundary_summary = boundary.get("summary")
    cases = boundary.get("boundary_crossing_cases")
    if not isinstance(boundary_summary, dict) or not isinstance(cases, list):
        raise ValueError("Boundary audit schema is invalid")
    if len(cases) != 347:
        raise ValueError("Boundary audit must contain exactly 347 cases")
    audit_keys = {(case["question_id"], case["answer_index"]) for case in cases}
    if len(audit_keys) != len(cases):
        raise ValueError("Boundary audit contains duplicate cases")
    flagged_keys = {
        key
        for key, answer in answer_by_key.items()
        if answer.get("boundary_crossing_token") is True
    }
    if (
        boundary_summary.get("answers_audited") != EXPECTED_ANSWERS
        or boundary_summary.get("answers_with_boundary_crossing_token") != 347
        or len(audit_keys) != 347
        or flagged_keys != audit_keys
    ):
        raise ValueError("Boundary flags or counts do not match the audit")

    print("Validation passed")
    print(f"Questions: {len(question_means)}")
    print(f"Answers: {len(answer_by_key)}")
    print(f"CSV rows: {len(csv_rows)}")
    print("Answer indices: complete 0-9 for every question")
    print("Correctness labels: match low-temperature file")
    print("Boundary flags: match 347 audit cases")
    print(f"Tolerance: {TOLERANCE}")
    print(json.dumps(checks, sort_keys=True))


if __name__ == "__main__":
    main()
