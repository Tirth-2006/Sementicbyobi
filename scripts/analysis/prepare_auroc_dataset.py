"""Join question-level entropy and low-temperature correctness for AUROC work."""
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List


CLUSTERS_INPUT = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_clusters.json"
)
CORRECTNESS_INPUT = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_low_temp_correctness.json"
)
OUTPUT_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_auroc_dataset.csv"
)
EXPECTED_QUESTIONS = 100
CSV_FIELDS = [
    "question_id",
    "entropy_nats",
    "correct",
    "low_temp_response",
    "question",
]


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON object from disk."""
    with path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    if not isinstance(data, dict):
        raise ValueError(f"Input JSON must contain an object: {path}")
    return data


def prepare_rows(
    clusters_data: Dict[str, Any],
    correctness_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Validate both inputs and join records by question ID."""
    cluster_questions = clusters_data.get("questions")
    if not isinstance(cluster_questions, dict):
        raise ValueError("Clustering JSON must contain a 'questions' object")

    correctness_questions = correctness_data.get("questions")
    if not isinstance(correctness_questions, list):
        raise ValueError("Low-temperature correctness JSON must contain a 'questions' list")

    cluster_ids = list(cluster_questions)
    correctness_ids = [
        item.get("question_id") if isinstance(item, dict) else None
        for item in correctness_questions
    ]

    if len(cluster_ids) != EXPECTED_QUESTIONS:
        raise ValueError(
            f"Expected {EXPECTED_QUESTIONS} clustered questions, found {len(cluster_ids)}"
        )
    if len(correctness_questions) != EXPECTED_QUESTIONS:
        raise ValueError(
            "Expected "
            f"{EXPECTED_QUESTIONS} low-temperature correctness records, "
            f"found {len(correctness_questions)}"
        )

    if len(cluster_ids) != len(set(cluster_ids)):
        raise ValueError("Question IDs in the clustering JSON must be unique")
    if any(not isinstance(question_id, str) or not question_id for question_id in correctness_ids):
        raise ValueError("Every correctness record must have a non-empty question_id")
    if len(correctness_ids) != len(set(correctness_ids)):
        raise ValueError("Question IDs in the correctness JSON must be unique")

    cluster_id_set = set(cluster_ids)
    correctness_id_set = set(correctness_ids)
    if cluster_id_set != correctness_id_set:
        missing_correctness = sorted(cluster_id_set - correctness_id_set)
        missing_clusters = sorted(correctness_id_set - cluster_id_set)
        raise ValueError(
            "Question IDs do not match. "
            f"Missing correctness records: {missing_correctness}; "
            f"missing cluster records: {missing_clusters}"
        )

    correctness_by_id = {
        item["question_id"]: item for item in correctness_questions
    }
    rows: List[Dict[str, Any]] = []

    for question_id in cluster_ids:
        cluster_record = cluster_questions[question_id]
        correctness_record = correctness_by_id[question_id]
        if not isinstance(cluster_record, dict):
            raise ValueError(f"Cluster record for {question_id!r} must be an object")

        entropy = cluster_record.get("entropy_nats")
        if entropy is None:
            raise ValueError(f"Missing entropy for question {question_id!r}")
        if isinstance(entropy, bool) or not isinstance(entropy, (int, float)):
            raise ValueError(f"Entropy for question {question_id!r} must be numeric")
        if not math.isfinite(entropy) or entropy < 0:
            raise ValueError(
                f"Entropy for question {question_id!r} must be finite and non-negative"
            )

        correct = correctness_record.get("correct")
        if correct is None:
            raise ValueError(f"Missing correctness value for question {question_id!r}")
        if isinstance(correct, bool) or not isinstance(correct, (int, float)):
            raise ValueError(f"Correctness for question {question_id!r} must be 0 or 1")
        if correct not in (0, 1):
            raise ValueError(f"Correctness for question {question_id!r} must be 0 or 1")

        low_temp_response = correctness_record.get("low_temp_response")
        question = correctness_record.get("question")
        if not isinstance(low_temp_response, str):
            raise ValueError(
                f"Low-temperature response for {question_id!r} must be a string"
            )
        if not isinstance(question, str):
            raise ValueError(f"Question text for {question_id!r} must be a string")

        rows.append(
            {
                "question_id": question_id,
                "entropy_nats": entropy,
                "correct": int(correct),
                "low_temp_response": low_temp_response,
                "question": question,
            }
        )

    return rows


def main() -> None:
    clusters_data = load_json(CLUSTERS_INPUT)
    correctness_data = load_json(CORRECTNESS_INPUT)
    rows = prepare_rows(clusters_data, correctness_data)

    # All validation and joining is complete before creating the output file.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    correct_count = sum(row["correct"] for row in rows)
    incorrect_count = len(rows) - correct_count
    mean_entropy = sum(row["entropy_nats"] for row in rows) / len(rows)

    print(f"Total questions: {len(rows)}")
    print(f"Correct answers: {correct_count}")
    print(f"Incorrect answers: {incorrect_count}")
    print(f"Mean entropy: {mean_entropy:.6f}")
    print(f"Output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
