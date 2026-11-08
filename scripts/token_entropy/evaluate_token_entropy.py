"""Evaluate token predictive entropy against low-temperature correctness."""
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


TOKEN_CSV = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_token_entropy_question.csv"
)
TOKEN_JSON = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_token_entropy.json"
)
LOW_TEMP_JSON = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_low_temp_correctness.json"
)
CLUSTERS_JSON = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_clusters.json"
)
OUTPUT_JSON = Path("data/generated/token_entropy_evaluation.json")
OUTPUT_CSV = Path("data/generated/token_entropy_evaluation.csv")
OUTPUT_REPORT = Path("data/generated/token_entropy_evaluation_report.md")

EXPECTED_QUESTIONS = 100
BOOTSTRAP_RESAMPLES = 20_000
RANDOM_SIMULATIONS = 20_000
RANDOM_SEED = 20261031
TOLERANCE = 1e-12
TOKEN_AGGREGATION_TOLERANCE = 1e-12


def finite_nonnegative(value: float) -> bool:
    return math.isfinite(value) and value >= 0


def load_token_rows() -> tuple[list[str], dict[str, dict[str, float]]]:
    with TOKEN_CSV.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != EXPECTED_QUESTIONS:
        raise ValueError(f"Token CSV must contain 100 rows, found {len(rows)}")
    order: list[str] = []
    values: dict[str, dict[str, float]] = {}
    required = {"question_id", "mean_token_entropy", "correct"}
    for row in rows:
        if set(row) != required | {"std_token_entropy"}:
            raise ValueError("Token CSV schema does not match the saved schema")
        question_id = row.get("question_id")
        if not question_id or question_id in values:
            raise ValueError(f"Duplicate or missing token question ID: {question_id}")
        entropy = float(row["mean_token_entropy"])
        if not finite_nonnegative(entropy):
            raise ValueError(f"Invalid token entropy for {question_id}")
        standard_deviation = float(row["std_token_entropy"])
        if not finite_nonnegative(standard_deviation):
            raise ValueError(f"Invalid token entropy standard deviation for {question_id}")
        if row["correct"] not in {"0", "1"}:
            raise ValueError(f"Invalid token CSV correctness for {question_id}")
        order.append(question_id)
        values[question_id] = {
            "token_entropy": entropy,
            "std_token_entropy": standard_deviation,
            "csv_correct": float(row["correct"]),
        }
    return order, values


def load_low_temp() -> dict[str, dict[str, float]]:
    with LOW_TEMP_JSON.open("r", encoding="utf-8") as file:
        data = json.load(file)
    questions = data.get("questions")
    if not isinstance(questions, list) or len(questions) != EXPECTED_QUESTIONS:
        raise ValueError("Low-temperature JSON must contain 100 questions")
    result: dict[str, dict[str, float]] = {}
    for item in questions:
        if not isinstance(item, dict):
            raise ValueError("Invalid low-temperature question record")
        question_id = item.get("question_id")
        correct = item.get("correct")
        exact_match = item.get("exact_match")
        token_f1 = item.get("token_f1")
        if not isinstance(question_id, str) or question_id in result:
            raise ValueError(f"Duplicate or missing low-temperature ID: {question_id}")
        if correct not in {0, 1} or exact_match not in {True, False}:
            raise ValueError(f"Invalid low-temperature correctness for {question_id}")
        if int(exact_match) != int(correct):
            raise ValueError(f"Correctness fields disagree for {question_id}")
        if not isinstance(token_f1, (int, float)) or not 0 <= token_f1 <= 1:
            raise ValueError(f"Invalid token F1 for {question_id}")
        result[question_id] = {
            "exact_match": int(exact_match),
            "token_f1": float(token_f1),
        }
    return result


def load_semantic_entropy() -> tuple[dict[str, float], dict[str, float]]:
    with CLUSTERS_JSON.open("r", encoding="utf-8") as file:
        data = json.load(file)
    questions = data.get("questions")
    if not isinstance(questions, dict) or len(questions) != EXPECTED_QUESTIONS:
        raise ValueError("Cluster JSON must contain 100 questions")
    result: dict[str, float] = {}
    recomputed: dict[str, float] = {}
    for question_id, item in questions.items():
        entropy = item.get("entropy_nats") if isinstance(item, dict) else None
        if not isinstance(entropy, (int, float)) or not finite_nonnegative(float(entropy)):
            raise ValueError(f"Invalid semantic entropy for {question_id}")
        clusters = item.get("clusters") if isinstance(item, dict) else None
        if not isinstance(clusters, list) or not clusters:
            raise ValueError(f"Missing semantic clusters for {question_id}")
        sizes = [cluster.get("size") for cluster in clusters]
        if not all(type(size) is int and size > 0 for size in sizes):
            raise ValueError(f"Invalid semantic cluster sizes for {question_id}")
        if sum(sizes) != 10:
            raise ValueError(f"Semantic clusters for {question_id} do not total 10")
        result[question_id] = float(entropy)
        recomputed[question_id] = float(
            -sum((size / 10) * math.log(size / 10) for size in sizes)
        )
    return result, recomputed


def load_token_json_ids() -> tuple[set[str], dict[str, dict[str, float]]]:
    with TOKEN_JSON.open("r", encoding="utf-8") as file:
        data = json.load(file)
    answers = data.get("answers")
    if not isinstance(answers, list) or len(answers) != 1000:
        raise ValueError("Token entropy JSON must contain 1,000 answers")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in answers:
        if not isinstance(item, dict):
            raise ValueError("Token entropy JSON contains a non-object answer record")
        question_id = item.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("Token entropy JSON contains a missing question ID")
        grouped.setdefault(question_id, []).append(item)

    if len(grouped) != EXPECTED_QUESTIONS:
        raise ValueError(
            "Token entropy JSON must contain exactly 100 question IDs, "
            f"found {len(grouped)}"
        )

    summaries: dict[str, dict[str, float]] = {}
    for question_id, question_answers in grouped.items():
        if len(question_answers) != 10:
            raise ValueError(
                f"{question_id} must contain exactly 10 answer records, "
                f"found {len(question_answers)}"
            )
        indices = [answer.get("answer_index") for answer in question_answers]
        if any(type(index) is not int for index in indices):
            raise ValueError(f"{question_id} has a non-integer answer index")
        if sorted(indices) != list(range(10)):
            raise ValueError(
                f"{question_id} must contain answer indices 0 through 9, "
                f"found {sorted(indices)}"
            )

        answer_means: list[float] = []
        for answer in question_answers:
            entropy_values = answer.get("token_entropy_nats")
            if (
                not isinstance(entropy_values, list)
                or not entropy_values
                or not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and finite_nonnegative(float(value))
                    for value in entropy_values
                )
            ):
                raise ValueError(
                    f"{question_id} answer {answer['answer_index']} has invalid "
                    "token entropy values"
                )
            recomputed_mean = float(np.mean(entropy_values))
            stored_mean = answer.get("mean_token_entropy_nats")
            if (
                not isinstance(stored_mean, (int, float))
                or not finite_nonnegative(float(stored_mean))
                or abs(float(stored_mean) - recomputed_mean)
                > TOKEN_AGGREGATION_TOLERANCE
            ):
                raise ValueError(
                    f"{question_id} answer {answer['answer_index']} has an "
                    "inconsistent stored mean token entropy"
                )
            answer_means.append(recomputed_mean)

        summaries[question_id] = {
            "mean_token_entropy": float(np.mean(answer_means)),
            "std_token_entropy": float(np.std(answer_means, ddof=0)),
        }

    return set(grouped), summaries


def validate_token_csv_aggregates(
    token_values: dict[str, dict[str, float]],
    token_json_summaries: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Compare CSV question aggregates with independently rebuilt JSON aggregates."""
    mean_differences = [
        abs(token_values[question_id]["token_entropy"] - summary["mean_token_entropy"])
        for question_id, summary in token_json_summaries.items()
    ]
    standard_deviation_differences = [
        abs(
            token_values[question_id]["std_token_entropy"]
            - summary["std_token_entropy"]
        )
        for question_id, summary in token_json_summaries.items()
    ]
    max_mean_difference = max(mean_differences)
    max_standard_deviation_difference = max(standard_deviation_differences)
    if max_mean_difference > TOKEN_AGGREGATION_TOLERANCE:
        raise ValueError(
            "Token CSV mean_token_entropy does not match the independently "
            f"reconstructed JSON aggregates (maximum difference "
            f"{max_mean_difference:.3e})"
        )
    if max_standard_deviation_difference > TOKEN_AGGREGATION_TOLERANCE:
        raise ValueError(
            "Token CSV std_token_entropy does not match the independently "
            f"reconstructed JSON aggregates (maximum difference "
            f"{max_standard_deviation_difference:.3e})"
        )
    return {
        "tolerance": TOKEN_AGGREGATION_TOLERANCE,
        "maximum_mean_absolute_difference": max_mean_difference,
        "maximum_std_absolute_difference": max_standard_deviation_difference,
    }


def aurac(correct: np.ndarray, entropy: np.ndarray, question_ids: list[str]) -> float:
    order = np.asarray(
        sorted(
            range(len(correct)),
            key=lambda index: (-entropy[index], question_ids[index]),
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


def point_metrics(
    entropy: np.ndarray,
    exact_match: np.ndarray,
    token_f1: np.ndarray,
    question_ids: list[str],
) -> dict[str, float]:
    incorrect = 1 - exact_match
    return {
        "spearman_exact_match": float(spearmanr(entropy, exact_match).statistic),
        "spearman_token_f1": float(spearmanr(entropy, token_f1).statistic),
        "auroc_incorrect": float(roc_auc_score(incorrect, entropy)),
        "aurac": aurac(exact_match, entropy, question_ids),
    }


def percentile(values: list[float]) -> list[float]:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return [float("nan"), float("nan")]
    result = np.percentile(finite_values, [2.5, 97.5])
    return [float(result[0]), float(result[1])]


def bootstrap(
    estimators: dict[str, np.ndarray],
    exact_match: np.ndarray,
    token_f1: np.ndarray,
    question_ids: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[float]]]:
    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.integers(
        0, len(exact_match), size=(BOOTSTRAP_RESAMPLES, len(exact_match))
    )
    distributions: dict[str, dict[str, list[float]]] = {
        name: {
            "spearman_exact_match": [],
            "spearman_token_f1": [],
            "auroc_incorrect": [],
            "aurac": [],
        }
        for name in estimators
    }
    undefined = {
        name: {metric: 0 for metric in distributions[name]}
        for name in estimators
    }
    for sample_indices in indices:
        sample_exact = exact_match[sample_indices]
        sample_f1 = token_f1[sample_indices]
        sample_ids = [question_ids[index] for index in sample_indices]
        for name, entropy in estimators.items():
            sample_entropy = entropy[sample_indices]
            correlation = spearmanr(sample_entropy, sample_exact).statistic
            if math.isfinite(correlation):
                distributions[name]["spearman_exact_match"].append(float(correlation))
            else:
                distributions[name]["spearman_exact_match"].append(float("nan"))
                undefined[name]["spearman_exact_match"] += 1
            correlation = spearmanr(sample_entropy, sample_f1).statistic
            if math.isfinite(correlation):
                distributions[name]["spearman_token_f1"].append(float(correlation))
            else:
                distributions[name]["spearman_token_f1"].append(float("nan"))
                undefined[name]["spearman_token_f1"] += 1
            if np.unique(1 - sample_exact).size == 2:
                distributions[name]["auroc_incorrect"].append(
                    float(roc_auc_score(1 - sample_exact, sample_entropy))
                )
            else:
                distributions[name]["auroc_incorrect"].append(float("nan"))
                undefined[name]["auroc_incorrect"] += 1
            distributions[name]["aurac"].append(
                aurac(sample_exact, sample_entropy, sample_ids)
            )
    output: dict[str, dict[str, Any]] = {}
    for name in estimators:
        output[name] = {
            "confidence_intervals_95": {
                metric: percentile(values)
                for metric, values in distributions[name].items()
            },
            "valid_samples": {
                metric: len(values)
                for metric, values in distributions[name].items()
            },
            "undefined_samples": undefined[name],
        }
    return output, {
        name: {
            metric: values
            for metric, values in distributions[name].items()
        }
        for name in estimators
    }


def paired_differences(
    distributions: dict[str, dict[str, list[float]]],
) -> dict[str, Any]:
    token_name = "token_entropy"
    semantic_name = "empirical_semantic_entropy"
    result: dict[str, Any] = {}
    for metric in (
        "spearman_exact_match",
        "auroc_incorrect",
        "aurac",
    ):
        token_values = np.asarray(distributions[token_name][metric], dtype=float)
        semantic_values = np.asarray(
            distributions[semantic_name][metric], dtype=float
        )
        valid = np.isfinite(token_values) & np.isfinite(semantic_values)
        differences = (token_values[valid] - semantic_values[valid]).tolist()
        result[metric] = {
            "point_difference": None,
            "confidence_interval_95": percentile(differences),
            "valid_paired_samples": int(valid.sum()),
            "undefined_paired_samples": int((~valid).sum()),
        }
    return result


def random_baseline(
    exact_match: np.ndarray,
    source_order_ids: list[str],
) -> dict[str, Any]:
    rng = np.random.default_rng(RANDOM_SEED)
    values = np.asarray(
        [
            aurac(
                exact_match,
                rng.random(len(exact_match)),
                source_order_ids,
            )
            for _ in range(RANDOM_SIMULATIONS)
        ],
        dtype=float,
    )
    bootstrap_rng = np.random.default_rng(RANDOM_SEED + 1)
    bootstrap_values: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample_indices = bootstrap_rng.integers(
            0, len(exact_match), size=len(exact_match)
        )
        sample_exact = exact_match[sample_indices]
        sample_ids = [source_order_ids[index] for index in sample_indices]
        bootstrap_values.append(
            aurac(
                sample_exact,
                bootstrap_rng.random(len(sample_exact)),
                sample_ids,
            )
        )
    return {
        "simulations": RANDOM_SIMULATIONS,
        "random_seed": RANDOM_SEED,
        "method": "continuous_uniform_random_entropy_scores",
        "point_estimate_mean": float(values.mean()),
        "simulation_confidence_interval_95": percentile(values.tolist()),
        "question_level_bootstrap": {
            "seed": RANDOM_SEED + 1,
            "confidence_interval_95": percentile(bootstrap_values),
            "valid_samples": len(bootstrap_values),
            "undefined_samples": 0,
        },
    }


def main() -> None:
    source_order, token_values = load_token_rows()
    token_json_ids, token_json_summaries = load_token_json_ids()
    token_aggregation_validation = validate_token_csv_aggregates(
        token_values, token_json_summaries
    )
    low_temp = load_low_temp()
    semantic, recomputed_semantic = load_semantic_entropy()
    all_ids = set(source_order)
    if not (all_ids == token_json_ids == set(low_temp) == set(semantic)):
        raise ValueError("Question IDs do not match across required artifacts")
    for question_id, token_record in token_values.items():
        if int(token_record["csv_correct"]) != low_temp[question_id]["exact_match"]:
            raise ValueError(
                f"Token CSV correctness disagrees with low-temperature label: "
                f"{question_id}"
            )

    question_ids = sorted(all_ids)
    exact_match = np.asarray([low_temp[qid]["exact_match"] for qid in question_ids])
    token_f1 = np.asarray([low_temp[qid]["token_f1"] for qid in question_ids])
    token_entropy = np.asarray(
        [token_values[qid]["token_entropy"] for qid in question_ids]
    )
    semantic_entropy = np.asarray([semantic[qid] for qid in question_ids])
    recomputed_semantic_entropy = np.asarray(
        [recomputed_semantic[qid] for qid in question_ids]
    )
    estimators = {
        "token_entropy": token_entropy,
        "empirical_semantic_entropy": semantic_entropy,
    }
    for name, values in estimators.items():
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{name} contains invalid values")
    point = {
        name: point_metrics(values, exact_match, token_f1, question_ids)
        for name, values in estimators.items()
    }
    bootstrap_results, distributions = bootstrap(
        estimators, exact_match, token_f1, question_ids
    )
    paired = paired_differences(distributions)
    for metric, item in paired.items():
        item["point_difference"] = (
            point["token_entropy"][metric]
            - point["empirical_semantic_entropy"][metric]
        )
    baseline = random_baseline(
        exact_match,
        source_order,
    )
    semantic_representation_differences = np.abs(
        semantic_entropy - recomputed_semantic_entropy
    )
    historical_semantic_audit = {
        "comparison": (
            "saved cluster entropy_nats versus entropy recomputed from saved "
            "cluster sizes using natural-log frequency entropy"
        ),
        "maximum_absolute_difference": float(
            semantic_representation_differences.max()
        ),
        "exactly_unequal_value_count": int(
            np.count_nonzero(semantic_entropy != recomputed_semantic_entropy)
        ),
        "saved_artifact_metrics": {
            "auroc_incorrect": float(
                roc_auc_score(1 - exact_match, semantic_entropy)
            ),
            "spearman_exact_match": float(
                spearmanr(semantic_entropy, exact_match).statistic
            ),
        },
        "recomputed_cluster_size_metrics": {
            "auroc_incorrect": float(
                roc_auc_score(1 - exact_match, recomputed_semantic_entropy)
            ),
            "spearman_exact_match": float(
                spearmanr(
                    recomputed_semantic_entropy, exact_match
                ).statistic
            ),
        },
        "cause": (
            "The historical values recomputed entropy from cluster sizes at "
            "metric time. Tiny floating-point representation differences "
            "between those values and the saved entropy_nats values broke "
            "mathematically tied entropy ranks, changing rank-based AUROC "
            "and Spearman results. Labels and question IDs are identical."
        ),
        "resolution": (
            "Preserve the current saved-artifact metrics; do not replace them "
            "with recomputed cluster-size values."
        ),
    }

    rows = [
        {
            "question_id": qid,
            "token_entropy": float(token_values[qid]["token_entropy"]),
            "exact_match": int(low_temp[qid]["exact_match"]),
            "token_f1": float(low_temp[qid]["token_f1"]),
            "empirical_semantic_entropy": float(semantic[qid]),
        }
        for qid in question_ids
    ]
    output = {
        "metadata": {
            "n_questions": EXPECTED_QUESTIONS,
            "correct_answers": int(exact_match.sum()),
            "incorrect_answers": int(len(exact_match) - exact_match.sum()),
            "question_join": "question_id",
            "target": "separate low-temperature exact-match correctness",
            "token_entropy_direction": "higher means greater uncertainty",
            "auroc_positive_class": "incorrect = 1 - exact_match",
            "aurac": {
                "rejection": "descending entropy",
                "coverage": "retained_questions / total_questions",
                "integration": "trapezoidal accuracy over coverage",
                "zero_coverage": "right-continuous extension from one retained question",
                "ties": "question_id ascending",
            },
            "aurac_convention_audit": {
                "previous_script": "evaluate_aurac.py",
                "difference_found": False,
                "details": (
                    "Both implementations reject descending entropy, break "
                    "exact ties by question_id ascending, use retained "
                    "questions divided by total questions as coverage, and "
                    "extend zero coverage from the one-retained-question "
                    "accuracy before trapezoidal integration."
                ),
            },
            "token_csv_aggregation_tolerance": TOKEN_AGGREGATION_TOLERANCE,
            "bootstrap": {
                "method": "paired question-level percentile bootstrap",
                "resamples": BOOTSTRAP_RESAMPLES,
                "random_seed": RANDOM_SEED,
                "undefined_samples": "reported per metric; not silently discarded",
            },
        },
        "validation": {
            "question_rows": len(rows),
            "question_ids_match": True,
            "one_row_per_question": True,
            "values_finite_nonnegative": True,
            "correctness_binary": True,
            "token_json_answer_records": 1000,
            "token_json_questions": len(token_json_summaries),
            "token_json_each_question_has_indices_0_through_9": True,
            "token_csv_std_token_entropy_validated": True,
            "token_csv_correctness_matches_low_temperature": True,
            "token_csv_aggregation_reconstruction": token_aggregation_validation,
            "historical_semantic_entropy_audit": historical_semantic_audit,
        },
        "point_estimates": point,
        "bootstrap": bootstrap_results,
        "paired_differences_token_minus_semantic": paired,
        "random_rejection_baseline": baseline,
        "rows": rows,
    }
    OUTPUT_JSON.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        fields = [
            "question_id",
            "token_entropy",
            "exact_match",
            "token_f1",
            "empirical_semantic_entropy",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "# Evaluation of Token Predictive Entropy",
        "",
        "## Objective",
        "",
        "For each of 100 TriviaQA questions, ten temperature-1.0 responses "
        "were rescored with teacher forcing. Their question-level mean "
        "reconstructed-text token predictive entropy is evaluated as an "
        "uncertainty predictor for the separate temperature-0.1 answer. "
        "Correctness uses the existing reference-based exact-match "
        "normalization.",
        "",
        "## Validation",
        "",
        "- 100 joined question rows and 1,000 token-entropy answer records.",
        "- Joins used `question_id`; all required ID sets matched.",
        "- Exact-match labels are binary: 27 correct and 73 incorrect.",
        "- Token and empirical semantic entropy values are finite and nonnegative.",
        "- Each token-entropy question has exactly 10 answers with indices 0 through 9.",
        f"- CSV mean and population standard deviation aggregates were independently "
        f"reconstructed from the JSON with tolerance "
        f"{TOKEN_AGGREGATION_TOLERANCE:.0e}; maximum absolute differences were "
        f"{token_aggregation_validation['maximum_mean_absolute_difference']:.3e} "
        f"and "
        f"{token_aggregation_validation['maximum_std_absolute_difference']:.3e}.",
        "- Duplicate, missing, and malformed question records were rejected.",
        "",
        "## Results",
        "",
        "| Estimator | Spearman exact match | Spearman token F1 | AUROC incorrect | AURAC |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in point.items():
        report.append(
            f"| {name} | {values['spearman_exact_match']:.6f} | "
            f"{values['spearman_token_f1']:.6f} | "
            f"{values['auroc_incorrect']:.6f} | {values['aurac']:.6f} |"
        )
    report.extend(
        [
            "",
            "### 95% bootstrap intervals",
            "",
            "| Estimator | Metric | 95% CI | Valid | Undefined |",
            "|---|---|---|---:|---:|",
        ]
    )
    for name, metrics in bootstrap_results.items():
        for metric, interval in metrics["confidence_intervals_95"].items():
            report.append(
                f"| {name} | {metric} | [{interval[0]:.6f}, {interval[1]:.6f}] | "
                f"{metrics['valid_samples'][metric]} | "
                f"{metrics['undefined_samples'][metric]} |"
            )
    report.extend(
        [
            "",
            "### Paired token-minus-semantic differences",
            "",
            "| Metric | Point difference | 95% paired CI | Valid | Undefined |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for metric, item in paired.items():
        report.append(
            f"| {metric} | {item['point_difference']:.6f} | "
            f"[{item['confidence_interval_95'][0]:.6f}, "
            f"{item['confidence_interval_95'][1]:.6f}] | "
            f"{item['valid_paired_samples']} | {item['undefined_paired_samples']} |"
        )
    report.extend(
        [
            "",
            "## Random baseline",
            "",
            f"- Point estimate mean: {baseline['point_estimate_mean']:.6f}",
            f"- Simulation 95% interval: [{baseline['simulation_confidence_interval_95'][0]:.6f}, "
            f"{baseline['simulation_confidence_interval_95'][1]:.6f}]",
            f"- Question-level bootstrap 95% interval: [{baseline['question_level_bootstrap']['confidence_interval_95'][0]:.6f}, "
            f"{baseline['question_level_bootstrap']['confidence_interval_95'][1]:.6f}]",
            "- The point estimate uses continuous uniform random entropy scores "
            "and the same AURAC convention as the prior validation.",
            "",
            "## Audit findings",
            "",
            "- The current `aurac()` implementation matches `evaluate_aurac.py`: "
            "descending-entropy rejection, question-ID-ascending exact-tie "
            "breaking, retained-question coverage, and the same right-continuous "
            "zero-coverage integration endpoint. No convention change was made.",
            "- The historical empirical semantic entropy AUROC (~0.841705) and "
            "Spearman value (~-0.528648) were produced by recomputing entropy "
            "from cluster sizes. The saved `entropy_nats` values differ from "
            "those recomputations by at most "
            f"{historical_semantic_audit['maximum_absolute_difference']:.3e}; "
            "these tiny differences break mathematical ties in rank-based "
            "metrics. Labels and question IDs are identical. The current "
            "saved-artifact results are therefore preserved.",
            "",
            "## Interpretation and limitations",
            "",
            "Higher token entropy is treated as greater uncertainty. AUROC "
            "therefore predicts the incorrect class directly, while AURAC "
            "rejects highest-entropy questions first. Point estimates alone "
            "do not establish superiority; paired bootstrap intervals should "
            "be considered. Token entropy is raw full-vocabulary softmax "
            "uncertainty along reconstructed saved-answer trajectories, not "
            "the temperature/top-k/top-p-filtered generation distribution. "
            "Boundary-crossing tokens and the shared first-token component "
            "remain as documented in the extraction report.",
        ]
    )
    OUTPUT_REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(point, indent=2))
    print(f"Random baseline mean AURAC: {baseline['point_estimate_mean']:.6f}")
    print(f"Saved: {OUTPUT_JSON}, {OUTPUT_CSV}, {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
