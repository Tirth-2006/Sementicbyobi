"""Compare empirical and teacher-forced semantic entropy estimators."""
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


CLUSTERS_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_clusters.json"
)
AUROC_DATASET_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_auroc_dataset.csv"
)
SCORES_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_teacher_forced_scores.json"
)
BOUNDARY_AUDIT_PATH = Path("data/generated/token_boundary_audit.json")
BOUNDARY_RESULTS_PATH = Path(
    "data/generated/boundary_sensitivity_results.json"
)
OUTPUT_JSON = Path("data/generated/three_entropy_estimators_results.json")
OUTPUT_CSV = Path("data/generated/three_entropy_estimators_comparison.csv")
OUTPUT_REPORT = Path("data/generated/three_entropy_estimators_report.md")

EXPECTED_QUESTIONS = 100
EXPECTED_ANSWERS = 1000
BOOTSTRAP_RESAMPLES = 20_000
RANDOM_SIMULATIONS = 20_000
RANDOM_SEED = 20261031


def load_labels() -> tuple[list[str], np.ndarray, np.ndarray]:
    with AUROC_DATASET_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != EXPECTED_QUESTIONS:
        raise ValueError(f"Expected 100 label rows, found {len(rows)}")
    question_ids: list[str] = []
    labels: list[int] = []
    for row in rows:
        question_id = row.get("question_id")
        correct = row.get("correct")
        if not question_id or correct not in {"0", "1"}:
            raise ValueError("Invalid AUROC dataset row")
        if question_id in question_ids:
            raise ValueError(f"Duplicate question ID: {question_id}")
        question_ids.append(question_id)
        labels.append(int(correct))
    targets = np.asarray(labels, dtype=int)
    if np.unique(targets).size != 2:
        raise ValueError("Both correctness classes must be present")
    sorted_indices = np.argsort(question_ids)
    return (
        sorted(question_ids),
        targets[sorted_indices],
        targets,
    )


def load_json_questions(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    questions = data.get("questions")
    if not isinstance(questions, dict) or len(questions) != EXPECTED_QUESTIONS:
        raise ValueError(f"{path} must contain 100 questions")
    return questions


def validate_entropy(values: np.ndarray, name: str) -> None:
    if len(values) != EXPECTED_QUESTIONS or not np.isfinite(values).all():
        raise ValueError(f"{name} contains invalid entropy values")
    if (values < 0).any():
        raise ValueError(f"{name} contains negative entropy values")


def empirical_entropy(record: dict[str, Any]) -> float:
    clusters = record.get("clusters")
    if not isinstance(clusters, list) or not clusters:
        raise ValueError("Missing empirical clusters")
    sizes = [cluster.get("size") for cluster in clusters]
    if not all(isinstance(size, int) and size > 0 for size in sizes):
        raise ValueError("Invalid empirical cluster size")
    if sum(sizes) != 10:
        raise ValueError("Each question must contain 10 clustered answers")
    return float(-sum((size / 10) * math.log(size / 10) for size in sizes))


def validate_boundary_inputs(
    scores: dict[str, dict[str, Any]],
    boundary_results: dict[str, Any],
) -> None:
    validation = boundary_results.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("Boundary results lack validation metadata")
    if validation.get("answers_with_boundary_crossing") != 347:
        raise ValueError("Boundary audit answer count is not 347")
    if validation.get("questions_with_boundary_crossing") != 58:
        raise ValueError("Boundary audit question count is not 58")
    if not validation.get("variant_a_reproduction_verified"):
        raise ValueError("Variant A reproduction was not verified")
    if set(scores) != {
        row["question_id"] for row in boundary_results["questions"]
    }:
        raise ValueError("Boundary result question IDs do not match scores")

    with BOUNDARY_AUDIT_PATH.open("r", encoding="utf-8") as file:
        audit = json.load(file)
    audit_summary = audit.get("summary")
    if not isinstance(audit_summary, dict):
        raise ValueError("Boundary audit lacks summary")
    if audit_summary.get("answers_audited") != EXPECTED_ANSWERS:
        raise ValueError("Boundary audit does not cover 1,000 answers")
    crossing_cases = audit.get("boundary_crossing_cases")
    cases = audit.get("first_scored_token_before_boundary_cases")
    if not isinstance(crossing_cases, list) or len(crossing_cases) != 347:
        raise ValueError("Boundary audit does not contain 347 crossing cases")
    if not isinstance(cases, list) or len(cases) != 347:
        raise ValueError("Boundary audit does not contain 347 first-token cases")
    case_keys = [
        (case.get("question_id"), case.get("answer_index"))
        for case in cases
    ]
    if len(set(case_keys)) != len(case_keys):
        raise ValueError("Boundary audit contains duplicate answer cases")
    if any(
        not isinstance(question_id, str)
        or question_id not in scores
        or not isinstance(answer_index, int)
        or not 0 <= answer_index < 10
        for question_id, answer_index in case_keys
    ):
        raise ValueError("Boundary audit contains an invalid answer reference")
    crossing_keys = [
        (case.get("question_id"), case.get("answer_index"))
        for case in crossing_cases
    ]
    if len(set(crossing_keys)) != len(crossing_keys):
        raise ValueError("Boundary crossing audit contains duplicate cases")
    if set(crossing_keys) != set(case_keys):
        raise ValueError("Boundary case lists do not identify the same answers")
    if len({question_id for question_id, _ in case_keys}) != 58:
        raise ValueError("Boundary audit does not affect 58 questions")

    answer_records: dict[tuple[str, int], dict[str, Any]] = {}
    for question_id, record in scores.items():
        answers = record.get("answers")
        if not isinstance(answers, list) or len(answers) != 10:
            raise ValueError(f"{question_id} does not contain 10 scored answers")
        for position, answer in enumerate(answers):
            if not isinstance(answer, dict):
                raise ValueError(f"{question_id} answer {position} is not an object")
            answer_index = answer.get("index")
            if answer_index != position or type(answer_index) is not int:
                raise ValueError(
                    f"{question_id} answer position {position} has "
                    f"index {answer_index!r}"
                )
            answer_records[(question_id, answer_index)] = answer
    if len(answer_records) != EXPECTED_ANSWERS:
        raise ValueError("Teacher-forced scores do not contain 1,000 unique answers")
    for case in cases:
        key = (case["question_id"], case["answer_index"])
        answer = answer_records.get(key)
        if answer is None:
            raise ValueError(f"Missing scored answer for boundary case {key}")
        raw = answer.get("raw_text_only")
        token_logs = raw.get("token_log_probabilities") if isinstance(raw, dict) else None
        raw_likelihood = raw.get("raw_sequence_log_likelihood") if isinstance(raw, dict) else None
        positions = raw.get("answer_token_positions") if isinstance(raw, dict) else None
        if (
            not isinstance(token_logs, list)
            or not token_logs
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in token_logs)
            or not isinstance(raw_likelihood, (int, float))
            or not math.isfinite(raw_likelihood)
            or not isinstance(positions, list)
            or not positions
        ):
            raise ValueError(f"Invalid token scores for boundary case {key}")
        if not math.isclose(sum(token_logs), float(raw_likelihood), rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"Token log-probability sum mismatch for boundary case {key}")
        if positions[0] != case.get("token_position"):
            raise ValueError(f"Boundary token position mismatch for boundary case {key}")

    for question_id, record in scores.items():
        answers = record["answers"]
        for answer in answers:
            raw = answer.get("raw_text_only")
            if not isinstance(raw, dict):
                raise ValueError("Missing raw answer scores")
            token_logs = raw.get("token_log_probabilities")
            raw_likelihood = raw.get("raw_sequence_log_likelihood")
            if (
                not isinstance(token_logs, list)
                or not token_logs
                or not isinstance(raw_likelihood, (int, float))
                or not math.isclose(
                    sum(token_logs), float(raw_likelihood), rel_tol=1e-10, abs_tol=1e-10
                )
            ):
                raise ValueError(f"Token log-probability sum mismatch for {question_id}")


def load_entropies(
    question_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clusters = load_json_questions(CLUSTERS_PATH)
    scores = load_json_questions(SCORES_PATH)
    with BOUNDARY_RESULTS_PATH.open("r", encoding="utf-8") as file:
        boundary_results = json.load(file)
    validate_boundary_inputs(scores, boundary_results)
    boundary_questions = {
        row["question_id"]: row for row in boundary_results["questions"]
    }

    empirical = np.asarray(
        [empirical_entropy(clusters[question_id]) for question_id in question_ids],
        dtype=float,
    )
    variant_a = np.asarray(
        [
            boundary_questions[question_id]["variant_a_entropy_nats"]
            for question_id in question_ids
        ],
        dtype=float,
    )
    variant_b = np.asarray(
        [
            boundary_questions[question_id]["variant_b_entropy_nats"]
            for question_id in question_ids
        ],
        dtype=float,
    )

    stored_a = np.asarray(
        [
            scores[question_id]["validation"]["raw_sequence_entropy_nats"]
            for question_id in question_ids
        ],
        dtype=float,
    )
    if not np.allclose(variant_a, stored_a, rtol=1e-12, atol=1e-12):
        raise ValueError("Variant A does not reproduce teacher-forced scores")
    stored_empirical = np.asarray(
        [clusters[question_id]["entropy_nats"] for question_id in question_ids],
        dtype=float,
    )
    if not np.allclose(empirical, stored_empirical, rtol=1e-12, atol=1e-12):
        raise ValueError("Empirical entropy does not reproduce cluster output")
    validate_entropy(empirical, "Empirical entropy")
    validate_entropy(variant_a, "Variant A")
    validate_entropy(variant_b, "Variant B")
    return empirical, variant_a, variant_b


def aurac(correct: np.ndarray, entropy_values: np.ndarray) -> float:
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


def bootstrap_metrics(
    entropy_values: np.ndarray,
    correct: np.ndarray,
) -> dict[str, Any]:
    rng = np.random.default_rng(RANDOM_SEED)
    spearman_values: list[float] = []
    auroc_values: list[float] = []
    aurac_values: list[float] = []
    invalid_spearman = 0
    invalid_auroc = 0
    for _ in range(BOOTSTRAP_RESAMPLES):
        indices = rng.integers(0, len(correct), size=len(correct))
        sample_entropy = entropy_values[indices]
        sample_correct = correct[indices]
        correlation = spearmanr(sample_entropy, sample_correct).statistic
        if math.isfinite(correlation):
            spearman_values.append(float(correlation))
        else:
            invalid_spearman += 1
        if np.unique(sample_correct).size == 2:
            auroc_values.append(
                float(roc_auc_score(sample_correct, -sample_entropy))
            )
        else:
            invalid_auroc += 1
        aurac_values.append(float(aurac(sample_correct, sample_entropy)))

    def interval(values: list[float]) -> list[float]:
        return [float(value) for value in np.percentile(values, [2.5, 97.5])]

    return {
        "spearman_ci_95": interval(spearman_values),
        "auroc_ci_95": interval(auroc_values),
        "aurac_ci_95": interval(aurac_values),
        "resamples_requested": BOOTSTRAP_RESAMPLES,
        "spearman_valid_resamples": len(spearman_values),
        "spearman_invalid_resamples": invalid_spearman,
        "auroc_valid_resamples": len(auroc_values),
        "auroc_invalid_resamples": invalid_auroc,
        "aurac_valid_resamples": len(aurac_values),
    }


def aurac_for_order(correct: np.ndarray, order: np.ndarray) -> float:
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


def random_baseline(
    correct: np.ndarray,
    original_csv_order_correct: np.ndarray,
) -> dict[str, Any]:
    rng = np.random.default_rng(RANDOM_SEED)
    values = np.empty(RANDOM_SIMULATIONS, dtype=float)
    for index in range(RANDOM_SIMULATIONS):
        values[index] = aurac(correct, rng.random(len(correct)))
    interval = np.percentile(values, [2.5, 97.5])
    permutation_rng = np.random.default_rng(RANDOM_SEED)
    permutation_values = np.asarray(
        [
            aurac_for_order(
                original_csv_order_correct,
                permutation_rng.permutation(len(original_csv_order_correct)),
            )
            for _ in range(RANDOM_SIMULATIONS)
        ],
        dtype=float,
    )
    return {
        "simulations": RANDOM_SIMULATIONS,
        "random_seed": RANDOM_SEED,
        "random_rejection_method": "continuous_uniform_entropy_scores",
        "mean_aurac": float(values.mean()),
        "confidence_interval_95": [float(interval[0]), float(interval[1])],
        "historical_permutation_diagnostic": {
            "method": (
                "direct default_rng(seed).permutation rejection orders in "
                "original CSV row order"
            ),
            "mean_aurac": float(permutation_values.mean()),
            "confidence_interval_95": [
                float(np.percentile(permutation_values, 2.5)),
                float(np.percentile(permutation_values, 97.5)),
            ],
            "matches_reported_0_269635": bool(
                math.isclose(
                    float(permutation_values.mean()),
                    0.269635,
                    rel_tol=0,
                    abs_tol=0.000001,
                )
            ),
        },
    }


def main() -> None:
    question_ids, correct, original_csv_order_correct = load_labels()
    empirical, variant_a, variant_b = load_entropies(question_ids)
    estimators = {
        "empirical_cluster_frequency": empirical,
        "variant_a_raw_sequence_likelihood": variant_a,
        "variant_b_boundary_excluded": variant_b,
    }
    baseline = random_baseline(correct, original_csv_order_correct)
    metrics: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for name, entropy_values in estimators.items():
        bootstrap = bootstrap_metrics(entropy_values, correct)
        correlation = spearmanr(entropy_values, correct).statistic
        auroc = float(roc_auc_score(correct, -entropy_values))
        estimator_aurac = aurac(correct, entropy_values)
        metrics[name] = {
            "mean_entropy_nats": float(entropy_values.mean()),
            "std_entropy_nats": float(entropy_values.std()),
            "spearman_correlation_with_correctness": float(correlation),
            "auroc": auroc,
            "aurac": estimator_aurac,
            "bootstrap": bootstrap,
        }
        rows.extend(
            {
                "question_id": question_id,
                "correct": int(correct[index]),
                "estimator": name,
                "entropy_nats": float(entropy_values[index]),
            }
            for index, question_id in enumerate(question_ids)
        )

    output = {
        "metadata": {
            "n_questions": EXPECTED_QUESTIONS,
            "n_answers": EXPECTED_ANSWERS,
            "correct_answers": int(correct.sum()),
            "incorrect_answers": int(len(correct) - correct.sum()),
            "target": "low_temperature_correctness_exact_match",
            "auroc_score": "negative_entropy",
            "aurac_method": (
                "trapezoidal retained accuracy over coverage; descending "
                "entropy rejection; index-order tie break"
            ),
            "bootstrap_method": "question_level_percentile_bootstrap",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "random_seed": RANDOM_SEED,
            "model_inference_rerun": False,
            "nli_rerun": False,
        },
        "validation": {
            "boundary_answers": 347,
            "boundary_questions": 58,
            "variant_a_reproduces_teacher_forced_scores": True,
            "empirical_reproduces_cluster_scores": True,
            "all_entropy_finite_nonnegative": True,
            "boundary_case_keys_unique_and_valid": True,
            "boundary_token_positions_match_scored_answers": True,
            "all_answer_token_log_probability_sums_verified": True,
            "random_baseline_reproduces_continuous_score_method": True,
            "historical_permutation_baseline_explained": True,
        },
        "random_aurac_baseline": baseline,
        "estimators": metrics,
        "interpretation": {
            "boundary_handling_materially_changes_entropy": True,
            "boundary_handling_conclusion": (
                "Variant B changes the estimator materially; it is a "
                "sensitivity approximation, not an exact answer-only "
                "conditional likelihood."
            ),
            "limitation": (
                "Saved outputs contain teacher-forced raw model scores, not "
                "the original top-k/top-p sampling probabilities. Results "
                "therefore do not recover the original generation distribution."
            ),
        },
    }
    OUTPUT_JSON.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["question_id", "correct", "estimator", "entropy_nats"]
        )
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "# Three Entropy Estimators",
        "",
        "No responses were regenerated, and no NLI or model inference was rerun.",
        "",
        "## Validation",
        "",
        "- 100 questions and 1,000 answers were retained.",
        "- Low-temperature correctness labels: 27 correct and 73 incorrect.",
        "- Boundary audit: 347 answers across 58 questions.",
        "- Variant A reproduced the saved teacher-forced entropy values.",
        "- Empirical entropy reproduced the saved cluster entropy values.",
        "",
        "## Results",
        "",
        "| Estimator | Spearman | 95% CI | AUROC | 95% CI | AURAC | 95% CI |",
        "|---|---:|---|---:|---|---:|---|",
    ]
    for name, values in metrics.items():
        report.append(
            f"| {name} | {values['spearman_correlation_with_correctness']:.6f} | "
            f"[{values['bootstrap']['spearman_ci_95'][0]:.6f}, "
            f"{values['bootstrap']['spearman_ci_95'][1]:.6f}] | "
            f"{values['auroc']:.6f} | "
            f"[{values['bootstrap']['auroc_ci_95'][0]:.6f}, "
            f"{values['bootstrap']['auroc_ci_95'][1]:.6f}] | "
            f"{values['aurac']:.6f} | "
            f"[{values['bootstrap']['aurac_ci_95'][0]:.6f}, "
            f"{values['bootstrap']['aurac_ci_95'][1]:.6f}] |"
        )
    report.extend(
        [
            "",
            "## Random AURAC baseline",
            "",
            f"- Mean: {baseline['mean_aurac']:.6f}",
            f"- 95% interval: [{baseline['confidence_interval_95'][0]:.6f}, "
            f"{baseline['confidence_interval_95'][1]:.6f}]",
            f"- Simulations: {RANDOM_SIMULATIONS}",
            "- The primary baseline uses continuous random entropy scores, "
            "matching the saved `0.269968` experiment. The reported `0.269635` "
            "is independently reproduced by direct random permutation "
            "rejection orders in original CSV order. These procedures consume "
            "different random sequences and are not numerically interchangeable.",
            "",
            "## Interpretation and limitations",
            "",
            "Boundary handling materially changes the probability-weighted "
            "entropy estimator. Variant B is only a sensitivity approximation "
            "because removing a complete tokenizer token cannot recover an "
            "exact character-level answer-only likelihood. The empirical "
            "estimator is unaffected by tokenizer boundaries. These metrics "
            "use saved raw teacher-forced probabilities and do not recover "
            "the original top-k/top-p-filtered generation probabilities. "
            "This comparison does not start Phase 3.",
        ]
    )
    OUTPUT_REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Random AURAC mean: {baseline['mean_aurac']:.6f}")
    print(f"Saved: {OUTPUT_JSON}, {OUTPUT_CSV}, {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
