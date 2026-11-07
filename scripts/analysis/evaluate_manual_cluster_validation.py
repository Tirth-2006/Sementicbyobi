"""Evaluate human semantic-equivalence labels against saved NLI decisions."""
import csv
import json
from pathlib import Path
from typing import Any


INPUT_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_manual_cluster_validation.csv"
)
OUTPUT_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_manual_cluster_validation_results.json"
)
REQUIRED_COLUMNS = {
    "review_id",
    "question_id",
    "answer_a",
    "answer_b",
    "nli_mutually_entail",
    "manual_semantically_equivalent",
}


def parse_binary(value: str, field: str, review_id: str) -> int:
    """Parse a manually or automatically stored binary label."""
    normalized = value.strip().lower()
    if normalized in {"1", "true"}:
        return 1
    if normalized in {"0", "false"}:
        return 0
    raise ValueError(f"{review_id}: {field} must be 0/1 or true/false")


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load and validate completed annotation rows."""
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                f"CSV must contain columns: {sorted(REQUIRED_COLUMNS)}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError("CSV contains no annotation rows")
    results: list[dict[str, Any]] = []
    for row in rows:
        review_id = row.get("review_id", "").strip()
        if not review_id:
            raise ValueError("Every row requires a review_id")
        nli = parse_binary(row["nli_mutually_entail"], "nli_mutually_entail", review_id)
        manual = parse_binary(
            row["manual_semantically_equivalent"],
            "manual_semantically_equivalent",
            review_id,
        )
        results.append(
            {
                "review_id": review_id,
                "question_id": row["question_id"],
                "answer_a": row["answer_a"],
                "answer_b": row["answer_b"],
                "nli": nli,
                "manual": manual,
                "notes": row.get("review_notes", ""),
            }
        )
    return results


def pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    """Canonicalize a question and unordered answer pair."""
    answers = tuple(sorted((row["answer_a"].strip().casefold(), row["answer_b"].strip().casefold())))
    return row["question_id"], answers[0], answers[1]


def calculate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate confusion-matrix and agreement metrics."""
    true_positive = sum(row["nli"] == 1 and row["manual"] == 1 for row in rows)
    true_negative = sum(row["nli"] == 0 and row["manual"] == 0 for row in rows)
    false_positive = sum(row["nli"] == 1 and row["manual"] == 0 for row in rows)
    false_negative = sum(row["nli"] == 0 and row["manual"] == 1 for row in rows)
    total = len(rows)
    accuracy = (true_positive + true_negative) / total
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "rows": total,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "nli_predicted_equivalent": sum(row["nli"] for row in rows),
        "manual_equivalent": sum(row["manual"] for row in rows),
    }


def main() -> None:
    rows = load_rows(INPUT_PATH)
    duplicate_groups: dict[tuple[str, str, str], list[str]] = {}
    for row in rows:
        duplicate_groups.setdefault(pair_key(row), []).append(row["review_id"])
    duplicate_groups = {
        key: ids for key, ids in duplicate_groups.items() if len(ids) > 1
    }

    metrics_all = calculate_metrics(rows)
    unique_rows = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = pair_key(row)
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
    metrics_unique = calculate_metrics(unique_rows)

    disagreements = [
        {
            "review_id": row["review_id"],
            "question_id": row["question_id"],
            "answer_a": row["answer_a"],
            "answer_b": row["answer_b"],
            "nli_prediction": row["nli"],
            "manual_label": row["manual"],
            "review_notes": row["notes"],
        }
        for row in rows
        if row["nli"] != row["manual"]
    ]
    output = {
        "input_file": str(INPUT_PATH),
        "metrics_all_rows": metrics_all,
        "metrics_unique_unordered_pairs": metrics_unique,
        "duplicate_groups": [
            {
                "question_id": key[0],
                "answer_a": key[1],
                "answer_b": key[2],
                "review_ids": ids,
            }
            for key, ids in duplicate_groups.items()
        ],
        "disagreements": disagreements,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print("All annotated rows:")
    print(json.dumps(metrics_all, indent=2))
    print("Unique unordered answer pairs:")
    print(json.dumps(metrics_unique, indent=2))
    print(f"Duplicate groups: {len(duplicate_groups)}")
    print(f"Disagreements: {len(disagreements)}")
    for disagreement in disagreements:
        print(
            f"{disagreement['review_id']}: "
            f"NLI={disagreement['nli_prediction']} "
            f"manual={disagreement['manual_label']} "
            f"({disagreement['answer_a']} vs {disagreement['answer_b']})"
        )
    print(f"Saved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
