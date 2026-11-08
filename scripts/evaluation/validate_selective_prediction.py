"""Validate AURAC, significance, and NLI-threshold sensitivity."""
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


CLUSTERS_INPUT = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_clusters.json"
)
AUROC_DATASET_INPUT = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_auroc_dataset.csv"
)
EXISTING_AURAC_INPUT = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_aurac_results.json"
)
OUTPUT_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_selective_validation.json"
)
BOOTSTRAP_RESAMPLES = 20_000
RANDOM_SIMULATIONS = 20_000
RANDOM_SEED = 20261031
THRESHOLDS = (0.4, 0.5, 0.6, 0.7)


def load_dataset(path: Path) -> dict[str, tuple[float, int]]:
    """Load entropy and low-temperature correctness by question ID."""
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    if len(rows) != 100:
        raise ValueError(f"Expected 100 AUROC rows, found {len(rows)}")

    observations: dict[str, tuple[float, int]] = {}
    for row in rows:
        question_id = row.get("question_id")
        entropy_text = row.get("entropy_nats")
        correct_text = row.get("correct")
        if not question_id or not entropy_text or correct_text not in {"0", "1"}:
            raise ValueError("AUROC dataset contains an invalid row")
        entropy = float(entropy_text)
        if not math.isfinite(entropy) or entropy < 0:
            raise ValueError(f"Invalid entropy for {question_id!r}")
        if question_id in observations:
            raise ValueError(f"Duplicate question ID: {question_id!r}")
        observations[question_id] = (entropy, int(correct_text))
    return observations


def load_clusters(path: Path) -> dict[str, dict[str, Any]]:
    """Load the saved pairwise NLI comparisons."""
    with path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    questions = data.get("questions")
    if not isinstance(questions, dict) or len(questions) != 100:
        raise ValueError("Clustering JSON must contain 100 questions")
    return questions


def threshold_clusters(
    record: dict[str, Any],
    threshold: float,
) -> list[list[int]]:
    """Rebuild greedy complete-linkage clusters from saved NLI scores."""
    responses = record.get("responses")
    comparisons = record.get("pairwise_comparisons")
    if not isinstance(responses, list) or not isinstance(comparisons, list):
        raise ValueError("Cluster record is missing responses or comparisons")

    mutual_by_pair: dict[tuple[int, int], bool] = {}
    for comparison in comparisons:
        first, second = comparison["answer_indices"]
        a_to_b = comparison["a_to_b"]
        b_to_a = comparison["b_to_a"]
        mutual_by_pair[(first, second)] = (
            a_to_b["label"].lower() == "entailment"
            and b_to_a["label"].lower() == "entailment"
            and a_to_b["score"] >= threshold
            and b_to_a["score"] >= threshold
        )

    clusters: list[list[int]] = []
    for answer_index in range(len(responses)):
        for cluster in clusters:
            if all(
                mutual_by_pair[(min(answer_index, existing), max(answer_index, existing))]
                for existing in cluster
            ):
                cluster.append(answer_index)
                break
        else:
            clusters.append([answer_index])
    return clusters


def entropy(clusters: list[list[int]]) -> float:
    """Calculate empirical cluster-frequency Shannon entropy in nats."""
    cluster_sizes = sorted(len(cluster) for cluster in clusters)
    total = sum(cluster_sizes)
    if total <= 0:
        raise ValueError("Cannot calculate entropy for no responses")
    return float(
        -sum(
            (size / total) * math.log(size / total)
            for size in cluster_sizes
        )
    )


def aurac(correct: np.ndarray, entropy_values: np.ndarray) -> float:
    """Calculate the finite-sample trapezoidal accuracy-over-coverage area."""
    order = np.asarray(
        sorted(
            range(len(correct)),
            key=lambda index: (-entropy_values[index], index),
        ),
        dtype=int,
    )
    total = len(correct)
    rejected = np.arange(total + 1)
    retained = total - rejected
    rejected_correct = np.concatenate(([0], np.cumsum(correct[order], dtype=int)))
    retained_correct = int(correct.sum()) - rejected_correct
    accuracy = np.full(total + 1, np.nan, dtype=float)
    accuracy[:-1] = retained_correct[:-1] / retained[:-1]
    accuracy[-1] = accuracy[-2]
    coverage = retained / total
    return float(np.trapezoid(accuracy[::-1], coverage[::-1]))


def bootstrap_aurac(
    correct: np.ndarray,
    entropy_values: np.ndarray,
) -> tuple[float, float, int]:
    """Estimate a percentile question-level bootstrap interval for AURAC."""
    rng = np.random.default_rng(RANDOM_SEED)
    estimates = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    for sample_index in range(BOOTSTRAP_RESAMPLES):
        indices = rng.integers(0, len(correct), size=len(correct))
        estimates[sample_index] = aurac(
            correct[indices], entropy_values[indices]
        )
    interval = np.percentile(estimates, [2.5, 97.5])
    return float(interval[0]), float(interval[1]), len(estimates)


def random_auracs(correct: np.ndarray) -> np.ndarray:
    """Generate the null distribution from random rejection orders."""
    rng = np.random.default_rng(RANDOM_SEED)
    values = np.empty(RANDOM_SIMULATIONS, dtype=float)
    for index in range(RANDOM_SIMULATIONS):
        random_entropy = rng.random(len(correct))
        values[index] = aurac(correct, random_entropy)
    return values


def main() -> None:
    observations = load_dataset(AUROC_DATASET_INPUT)
    cluster_records = load_clusters(CLUSTERS_INPUT)
    if set(observations) != set(cluster_records):
        raise ValueError("AUROC and clustering question IDs do not match")

    question_ids = sorted(observations)
    baseline_entropy = np.asarray(
        [observations[question_id][0] for question_id in question_ids],
        dtype=float,
    )
    correct = np.asarray(
        [observations[question_id][1] for question_id in question_ids],
        dtype=int,
    )

    stored_results = json.loads(EXISTING_AURAC_INPUT.read_text(encoding="utf-8"))
    stored_aurac = float(stored_results["aurac"]["value"])
    recomputed_aurac = aurac(correct, baseline_entropy)
    stored_auroc = float(
        json.loads(
            Path(
                "data/generated/trivia_qa_qwen3_20261024_162539_auroc_results.json"
            ).read_text(encoding="utf-8")
        )["auroc"]
    )
    ci_low, ci_high, valid_bootstrap = bootstrap_aurac(
        correct, baseline_entropy
    )

    random_values = random_auracs(correct)
    random_interval = np.percentile(random_values, [2.5, 97.5])
    random_p_value = float(
        (1 + np.count_nonzero(random_values >= recomputed_aurac))
        / (len(random_values) + 1)
    )

    sensitivity: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        threshold_entropy = np.asarray(
            [
                entropy(threshold_clusters(cluster_records[question_id], threshold))
                for question_id in question_ids
            ],
            dtype=float,
        )
        sensitivity.append(
            {
                "threshold": threshold,
                "mean_entropy_nats": float(threshold_entropy.mean()),
                "auroc": float(roc_auc_score(correct, -threshold_entropy)),
                "aurac": aurac(correct, threshold_entropy),
            }
        )

    results = {
        "n_questions": len(correct),
        "correct_answers": int(correct.sum()),
        "baseline": {
            "stored_aurac": stored_aurac,
            "recomputed_aurac": recomputed_aurac,
            "absolute_difference": abs(stored_aurac - recomputed_aurac),
            "stored_auroc": stored_auroc,
            "canonical_threshold_0_5_auroc": sensitivity[1]["auroc"],
            "auroc_difference_from_float_tie_artifacts": (
                sensitivity[1]["auroc"] - stored_auroc
            ),
            "method": (
                "trapezoidal integral of retained-answer accuracy over coverage; "
                "zero-coverage endpoint equals one-retained-answer accuracy"
            ),
        },
        "aurac_bootstrap": {
            "method": "question_level_percentile_bootstrap",
            "resamples": valid_bootstrap,
            "random_seed": RANDOM_SEED,
            "confidence_interval_95": [ci_low, ci_high],
        },
        "random_rejection_test": {
            "method": "one_sided_monte_carlo_randomization_test",
            "simulations": RANDOM_SIMULATIONS,
            "random_seed": RANDOM_SEED,
            "null_mean_aurac": float(random_values.mean()),
            "null_confidence_interval_95": [
                float(random_interval[0]),
                float(random_interval[1]),
            ],
            "observed_minus_null_mean": float(
                recomputed_aurac - random_values.mean()
            ),
            "p_value_greater_or_equal": random_p_value,
        },
        "nli_threshold_sensitivity": sensitivity,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"Recomputed AURAC: {recomputed_aurac:.6f}")
    print(f"AURAC 95% bootstrap CI: [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"Randomization p-value: {random_p_value:.6f}")
    for result in sensitivity:
        print(
            f"Threshold {result['threshold']:.1f}: "
            f"AUROC={result['auroc']:.6f}, AURAC={result['aurac']:.6f}"
        )
    print(f"Saved validation results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
