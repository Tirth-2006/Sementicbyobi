"""Analyze question-level Spearman associations for semantic entropy."""

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


INPUT_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_entropy_accuracy_f1.csv"
)
OUTPUT_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_spearman_results.json"
)
BOOTSTRAP_RESAMPLES = 20_000
PERMUTATION_RESAMPLES = 100_000
RANDOM_SEED = 20261031


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return 1-based average ranks, assigning tied values their mean rank."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_values) != 0) + 1, len(values)]
    sorted_ranks = np.empty(len(values), dtype=float)

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        sorted_ranks[start:end] = ((start + 1) + end) / 2.0

    ranks = np.empty(len(values), dtype=float)
    ranks[order] = sorted_ranks
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rho using average ranks for ties."""
    rank_x = average_ranks(x)
    rank_y = average_ranks(y)
    centered_x = rank_x - rank_x.mean()
    centered_y = rank_y - rank_y.mean()
    denominator = np.sqrt(np.dot(centered_x, centered_x) * np.dot(centered_y, centered_y))
    if denominator == 0:
        return float("nan")
    return float(np.dot(centered_x, centered_y) / denominator)


def permutation_p_value(
    x: np.ndarray,
    y: np.ndarray,
    observed_rho: float,
    resamples: int,
    seed: int,
) -> float:
    """Two-sided Monte Carlo permutation p-value, preserving ties in y ranks."""
    rank_x = average_ranks(x)
    rank_y = average_ranks(y)
    centered_x = rank_x - rank_x.mean()
    centered_y = rank_y - rank_y.mean()
    denominator = np.sqrt(np.dot(centered_x, centered_x) * np.dot(centered_y, centered_y))

    if denominator == 0:
        return float("nan")

    rng = np.random.default_rng(seed)
    extreme = 0
    observed_magnitude = abs(observed_rho)
    for _ in range(resamples):
        permuted_rho = float(np.dot(centered_x, rng.permutation(centered_y)) / denominator)
        if abs(permuted_rho) >= observed_magnitude - 1e-15:
            extreme += 1

    # The plus-one correction gives a valid Monte Carlo test p-value.
    return (extreme + 1) / (resamples + 1)


def bootstrap_interval(
    x: np.ndarray,
    y: np.ndarray,
    resamples: int,
    seed: int,
) -> tuple[list[float], int]:
    """Return percentile 95% CI from paired question-level bootstrap samples."""
    rng = np.random.default_rng(seed)
    n = len(x)
    estimates = np.empty(resamples, dtype=float)
    valid = 0

    for sample_number in range(resamples):
        indices = rng.integers(0, n, size=n)
        estimate = spearman_rho(x[indices], y[indices])
        estimates[sample_number] = estimate
        if np.isfinite(estimate):
            valid += 1

    finite_estimates = estimates[np.isfinite(estimates)]
    if finite_estimates.size == 0:
        return [float("nan"), float("nan")], 0

    lower, upper = np.quantile(finite_estimates, [0.025, 0.975])
    return [float(lower), float(upper)], valid


def load_observations(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Read and validate the one-row-per-question analysis CSV."""
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {
            "question_id",
            "semantic_entropy_nats",
            "accuracy",
            "mean_token_f1",
        }
        headers = reader.fieldnames
        if headers is None or len(headers) != len(set(headers)):
            raise ValueError("CSV must have a header row with unique column names")
        if not required.issubset(headers):
            raise ValueError(f"CSV must contain columns: {sorted(required)}")

        rows = list(reader)

    if len(rows) != 100:
        raise ValueError(f"Expected 100 question rows, found {len(rows)}")

    question_ids = [row["question_id"] for row in rows]
    if any(not question_id or not question_id.strip() for question_id in question_ids):
        raise ValueError("Question IDs must not be empty or whitespace-only")
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("Question IDs must be unique")

    try:
        entropy = np.array([float(row["semantic_entropy_nats"]) for row in rows])
        accuracy = np.array([float(row["accuracy"]) for row in rows])
        mean_f1 = np.array([float(row["mean_token_f1"]) for row in rows])
    except (TypeError, ValueError) as error:
        raise ValueError("Entropy, accuracy, and mean token F1 must be numeric") from error

    if not (np.isfinite(entropy).all() and np.isfinite(accuracy).all() and np.isfinite(mean_f1).all()):
        raise ValueError("Analysis columns must contain only finite values")
    if (entropy < 0).any():
        raise ValueError("Semantic entropy values must be non-negative")
    if ((accuracy < 0) | (accuracy > 1)).any():
        raise ValueError("Accuracy values must be between 0 and 1")
    if ((mean_f1 < 0) | (mean_f1 > 1)).any():
        raise ValueError("Mean token F1 values must be between 0 and 1")

    return question_ids, entropy, accuracy, mean_f1


def analyze_pair(x: np.ndarray, y: np.ndarray, seed_offset: int) -> dict[str, Any]:
    rho = spearman_rho(x, y)
    confidence_interval, valid_bootstrap = bootstrap_interval(
        x, y, BOOTSTRAP_RESAMPLES, RANDOM_SEED + seed_offset
    )
    p_value = permutation_p_value(
        x, y, rho, PERMUTATION_RESAMPLES, RANDOM_SEED + 10_000 + seed_offset
    )
    return {
        "spearman_rho": rho,
        "two_sided_permutation_p_value": p_value,
        "confidence_interval_95_percentile_bootstrap": confidence_interval,
        "requested_bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "valid_bootstrap_resamples": valid_bootstrap,
        "invalid_bootstrap_resamples": BOOTSTRAP_RESAMPLES - valid_bootstrap,
    }


def main() -> None:
    question_ids, entropy, accuracy, mean_f1 = load_observations(INPUT_PATH)
    results = {
        "input_csv": str(INPUT_PATH),
        "n_questions": len(question_ids),
        "method": {
            "correlation": "Spearman rank correlation with average ranks for ties",
            "p_value": "Two-sided Monte Carlo permutation test; plus-one correction",
            "permutation_resamples": PERMUTATION_RESAMPLES,
            "confidence_interval": "Paired question-level percentile bootstrap, 95%",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "random_seed": RANDOM_SEED,
        },
        "semantic_entropy_vs_question_accuracy": analyze_pair(entropy, accuracy, 1),
        "semantic_entropy_vs_mean_token_f1": analyze_pair(entropy, mean_f1, 2),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2, allow_nan=False)
        output_file.write("\n")

    print(json.dumps(results, indent=2))
    print(f"Saved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
