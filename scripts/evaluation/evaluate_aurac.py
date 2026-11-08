"""Evaluate entropy-based selective answering with a rejection-accuracy curve."""
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from uncertainty.evaluation.selective_prediction import rejection_curve


INPUT_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_auroc_dataset.csv"
)
RESULTS_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_aurac_results.json"
)
CURVE_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_aurac_curve.csv"
)
EXPECTED_ROWS = 100
RANDOM_SIMULATIONS = 20_000
RANDOM_SEED = 20261031
REQUIRED_COLUMNS = {"question_id", "entropy_nats", "correct"}
CURVE_FIELDS = [
    "rejection_rate_percent",
    "coverage",
    "retained_count",
    "rejected_count",
    "retained_accuracy",
    "accuracy_for_aurac",
]


def load_observations(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load and validate one entropy and correctness observation per question."""
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                f"CSV must contain columns: {sorted(REQUIRED_COLUMNS)}"
            )
        rows = list(reader)

    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(rows)}")

    question_ids: list[str] = []
    entropy_values: list[float] = []
    correct_values: list[int] = []

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
            raise ValueError(f"Row {row_number} has a non-numeric entropy") from error
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

    correct = np.asarray(correct_values, dtype=int)
    if np.unique(correct).size != 2:
        raise ValueError("Both correctness classes must be present")

    return question_ids, np.asarray(entropy_values, dtype=float), correct


def make_curve_rows(
    coverage: np.ndarray,
    displayed_accuracy: np.ndarray,
    integration_accuracy: np.ndarray,
) -> list[dict[str, Any]]:
    """Format the 0%-through-100% rejection curve for CSV output."""
    total_questions = len(coverage) - 1
    rows: list[dict[str, Any]] = []
    for rejected_count, (coverage_value, accuracy, integration_value) in enumerate(
        zip(coverage, displayed_accuracy, integration_accuracy)
    ):
        retained_count = total_questions - rejected_count
        rows.append(
            {
                "rejection_rate_percent": rejected_count,
                "coverage": float(coverage_value),
                "retained_count": retained_count,
                "rejected_count": rejected_count,
                "retained_accuracy": (
                    "" if np.isnan(accuracy) else float(accuracy)
                ),
                "accuracy_for_aurac": float(integration_value),
            }
        )
    return rows


def random_baseline_auracs(correct: np.ndarray) -> np.ndarray:
    """Simulate random rejection orders using the same curve convention."""
    rng = np.random.default_rng(RANDOM_SEED)
    auracs = np.empty(RANDOM_SIMULATIONS, dtype=float)
    for simulation_index in range(RANDOM_SIMULATIONS):
        rejection_order = rng.permutation(len(correct))
        _, _, _, auracs[simulation_index] = rejection_curve(
            correct, rejection_order
        )
    return auracs


def main() -> None:
    question_ids, entropy, correct = load_observations(INPUT_PATH)

    # Highest entropy is rejected first. Exact ties use question ID ascending.
    entropy_rejection_order = np.asarray(
        sorted(
            range(len(question_ids)),
            key=lambda index: (-entropy[index], question_ids[index]),
        ),
        dtype=int,
    )
    coverage, displayed_accuracy, integration_accuracy, aurac = rejection_curve(
        correct, entropy_rejection_order
    )
    curve_rows = make_curve_rows(
        coverage, displayed_accuracy, integration_accuracy
    )

    random_auracs = random_baseline_auracs(correct)
    random_interval = np.percentile(random_auracs, [2.5, 97.5])
    random_mean = float(random_auracs.mean())
    initial_accuracy = float(correct.mean())

    results = {
        "input_csv": str(INPUT_PATH),
        "curve_csv": str(CURVE_PATH),
        "n_questions": len(correct),
        "correct_answers": int(correct.sum()),
        "incorrect_answers": int(len(correct) - correct.sum()),
        "initial_accuracy": initial_accuracy,
        "entropy_rejection_order": {
            "direction": "descending_entropy",
            "ties": "question_id_ascending",
        },
        "curve": {
            "rejection_rates_percent": {"start": 0, "stop": 100, "step": 1},
            "x_axis": "coverage_fraction_retained",
            "y_axis": "accuracy_among_retained_answers",
            "zero_coverage": {
                "retained_accuracy": None,
                "integration_convention": (
                    "right_continuous_extension_using_accuracy_at_one_percent_coverage"
                ),
            },
        },
        "aurac": {
            "value": aurac,
            "method": "trapezoidal_integral_of_accuracy_over_coverage",
            "normalization": (
                "none; coverage is a fraction from 0 to 1, so the integral is "
                "already on the AURAC scale"
            ),
        },
        "random_rejection_baseline": {
            "simulations": RANDOM_SIMULATIONS,
            "random_seed": RANDOM_SEED,
            "mean_aurac": random_mean,
            "confidence_interval_95_percentile": [
                float(random_interval[0]),
                float(random_interval[1]),
            ],
            "entropy_minus_random_mean": aurac - random_mean,
        },
    }

    # Every validation step occurs before either output is written.
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CURVE_PATH.open("w", encoding="utf-8", newline="") as curve_file:
        writer = csv.DictWriter(curve_file, fieldnames=CURVE_FIELDS)
        writer.writeheader()
        writer.writerows(curve_rows)
    with RESULTS_PATH.open("w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=2)
        results_file.write("\n")

    print(f"Total questions: {len(correct)}")
    print(f"Initial accuracy: {initial_accuracy:.6f}")
    print(f"AURAC: {aurac:.6f}")
    print(f"Mean random-baseline AURAC: {random_mean:.6f}")
    print(
        "95% random-baseline interval: "
        f"[{random_interval[0]:.6f}, {random_interval[1]:.6f}]"
    )
    print(f"Entropy minus random-baseline AURAC: {aurac - random_mean:.6f}")
    print(f"Results file: {RESULTS_PATH}")
    print(f"Curve file: {CURVE_PATH}")


if __name__ == "__main__":
    main()
