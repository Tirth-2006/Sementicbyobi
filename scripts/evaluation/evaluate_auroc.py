"""Evaluate semantic entropy as a predictor of low-temperature correctness."""
import csv
import json
import math
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


INPUT_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_auroc_dataset.csv"
)
OUTPUT_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_auroc_results.json"
)
EXPECTED_ROWS = 100
BOOTSTRAP_RESAMPLES = 20_000
RANDOM_SEED = 20261031
REQUIRED_COLUMNS = {"question_id", "entropy_nats", "correct"}


def load_observations(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load and validate entropy and correctness at the question level."""
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(
                f"CSV must contain columns: {sorted(REQUIRED_COLUMNS)}"
            )
        rows = list(reader)

    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(rows)}")

    question_ids: List[str] = []
    entropy_values: List[float] = []
    correct_values: List[int] = []

    for row_number, row in enumerate(rows, start=2):
        question_id = row.get("question_id")
        entropy_text = row.get("entropy_nats")
        correct_text = row.get("correct")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"Row {row_number} is missing question_id")
        if not isinstance(entropy_text, str) or not entropy_text.strip():
            raise ValueError(f"Row {row_number} is missing entropy_nats")
        if not isinstance(correct_text, str) or not correct_text.strip():
            raise ValueError(f"Row {row_number} is missing correct")

        try:
            entropy = float(entropy_text)
        except ValueError as error:
            raise ValueError(f"Row {row_number} has a non-numeric entropy value") from error
        if not math.isfinite(entropy) or entropy < 0:
            raise ValueError(
                f"Row {row_number} has entropy that is not finite and non-negative"
            )

        if correct_text.strip() not in {"0", "1"}:
            raise ValueError(f"Row {row_number} has correctness outside {{0, 1}}")

        question_ids.append(question_id)
        entropy_values.append(entropy)
        correct_values.append(int(correct_text))

    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Question IDs must be unique")

    targets = np.asarray(correct_values, dtype=int)
    if np.unique(targets).size != 2:
        raise ValueError("Both correctness classes must be present")

    return np.asarray(entropy_values, dtype=float), targets


def bootstrap_auroc(
    targets: np.ndarray,
    scores: np.ndarray,
) -> Tuple[List[float], int, int]:
    """Calculate a percentile bootstrap interval from question-level resamples."""
    rng = np.random.default_rng(RANDOM_SEED)
    estimates: List[float] = []
    invalid_resamples = 0

    for _ in range(BOOTSTRAP_RESAMPLES):
        indices = rng.integers(0, len(targets), size=len(targets))
        bootstrap_targets = targets[indices]
        if np.unique(bootstrap_targets).size != 2:
            invalid_resamples += 1
            continue
        estimates.append(
            float(roc_auc_score(bootstrap_targets, scores[indices]))
        )

    valid_resamples = len(estimates)
    if valid_resamples == 0:
        raise RuntimeError("No bootstrap samples contained both correctness classes")

    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return [float(lower), float(upper)], valid_resamples, invalid_resamples


def main() -> None:
    entropy, correct = load_observations(INPUT_PATH)
    scores = -entropy
    auroc = float(roc_auc_score(correct, scores))
    confidence_interval, valid_resamples, invalid_resamples = bootstrap_auroc(
        correct, scores
    )

    correct_count = int(correct.sum())
    results: dict[str, Any] = {
        "input_csv": str(INPUT_PATH),
        "n_questions": len(correct),
        "correct_answers": correct_count,
        "incorrect_answers": len(correct) - correct_count,
        "predictor": "negative_entropy_nats",
        "target": "correct",
        "auroc": auroc,
        "bootstrap": {
            "method": "question_level_percentile_bootstrap",
            "resamples_requested": BOOTSTRAP_RESAMPLES,
            "random_seed": RANDOM_SEED,
            "confidence_interval_95": confidence_interval,
            "valid_resamples": valid_resamples,
            "invalid_resamples": invalid_resamples,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)
        output_file.write("\n")

    print(f"Total questions: {len(correct)}")
    print(f"Correct answers: {correct_count}")
    print(f"Incorrect answers: {len(correct) - correct_count}")
    print(f"AUROC: {auroc:.6f}")
    print(
        "95% bootstrap confidence interval: "
        f"[{confidence_interval[0]:.6f}, {confidence_interval[1]:.6f}]"
    )
    print(f"Valid bootstrap resamples: {valid_resamples}")
    print(f"Invalid bootstrap resamples: {invalid_resamples}")
    print(f"Saved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
