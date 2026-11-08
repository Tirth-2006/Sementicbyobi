"""Compare teacher-forced entropy with boundary-excluded token sensitivity."""
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


SCORES_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_teacher_forced_scores.json"
)
AUDIT_PATH = Path("data/generated/token_boundary_audit.json")
CLUSTERS_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_clusters.json"
)
OUTPUT_JSON = Path("data/generated/boundary_sensitivity_results.json")
OUTPUT_CSV = Path("data/generated/boundary_sensitivity_comparison.csv")
OUTPUT_REPORT = Path("data/generated/boundary_sensitivity_report.md")


def stable_weights(scores: list[float]) -> list[float]:
    """Return stable softmax weights for a question's answer scores."""
    values = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Answer scores must be finite")
    shifted = values - values.max()
    weights = np.exp(shifted)
    weights /= weights.sum()
    result = weights.tolist()
    if not math.isclose(sum(result), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Answer weights do not sum to one")
    return result


def cluster_entropy(
    weights: list[float],
    cluster_record: dict[str, Any],
) -> tuple[float, list[float]]:
    """Aggregate answer weights over saved clusters and calculate entropy."""
    clusters = cluster_record.get("clusters")
    if not isinstance(clusters, list) or not clusters:
        raise ValueError("Missing saved clusters")
    probabilities: list[float] = []
    assigned: list[int] = []
    for cluster in clusters:
        indices = cluster.get("indices") if isinstance(cluster, dict) else None
        if not isinstance(indices, list):
            raise ValueError("Invalid cluster indices")
        probabilities.append(sum(weights[index] for index in indices))
        assigned.extend(indices)
    if sorted(assigned) != list(range(len(weights))):
        raise ValueError("Clusters do not cover every answer exactly once")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Cluster probabilities do not sum to one")
    entropy = -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0
    )
    if not math.isfinite(entropy) or entropy < -1e-12:
        raise ValueError("Entropy is not finite and nonnegative")
    return max(0.0, entropy), probabilities


def load_boundary_cases(path: Path) -> set[tuple[str, int]]:
    """Load answer IDs whose first scored token crosses the boundary."""
    with path.open("r", encoding="utf-8") as input_file:
        audit = json.load(input_file)
    cases = audit.get("first_scored_token_before_boundary_cases")
    if not isinstance(cases, list):
        raise ValueError("Boundary audit is missing first-token cases")
    return {
        (case["question_id"], case["answer_index"])
        for case in cases
    }


def main() -> None:
    with SCORES_PATH.open("r", encoding="utf-8") as input_file:
        scores_data = json.load(input_file)
    with CLUSTERS_PATH.open("r", encoding="utf-8") as input_file:
        clusters_data = json.load(input_file)

    scores_questions = scores_data.get("questions")
    cluster_questions = clusters_data.get("questions")
    if not isinstance(scores_questions, dict) or len(scores_questions) != 100:
        raise ValueError("Scores must contain exactly 100 questions")
    if not isinstance(cluster_questions, dict) or set(scores_questions) != set(
        cluster_questions
    ):
        raise ValueError("Score and cluster question IDs do not match")
    boundary_cases = load_boundary_cases(AUDIT_PATH)

    question_rows: list[dict[str, Any]] = []
    affected_answer_count = 0
    for question_id, score_record in scores_questions.items():
        answers = score_record.get("answers")
        if not isinstance(answers, list) or len(answers) != 10:
            raise ValueError(f"{question_id!r} must contain exactly 10 answers")
        raw_scores: list[float] = []
        excluded_scores: list[float] = []
        crossing_count = 0
        for answer in answers:
            index = answer.get("index")
            raw = answer.get("raw_text_only")
            if not isinstance(index, int) or not isinstance(raw, dict):
                raise ValueError(f"Invalid answer score record in {question_id!r}")
            token_scores = raw.get("token_log_probabilities")
            raw_score = raw.get("raw_sequence_log_likelihood")
            if not isinstance(token_scores, list) or not isinstance(raw_score, (int, float)):
                raise ValueError(f"Missing token scores in {question_id!r}:{index}")
            if not token_scores or not np.isfinite(token_scores).all():
                raise ValueError(f"Non-finite token scores in {question_id!r}:{index}")
            raw_scores.append(float(raw_score))
            if (question_id, index) in boundary_cases:
                if len(token_scores) < 2:
                    raise ValueError(
                        f"Cannot exclude the only token in {question_id!r}:{index}"
                    )
                excluded_scores.append(float(sum(token_scores[1:])))
                crossing_count += 1
            else:
                excluded_scores.append(float(raw_score))

        weights_a = stable_weights(raw_scores)
        weights_b = stable_weights(excluded_scores)
        entropy_a, clusters_a = cluster_entropy(
            weights_a, cluster_questions[question_id]
        )
        entropy_b, clusters_b = cluster_entropy(
            weights_b, cluster_questions[question_id]
        )
        stored_entropy = score_record.get("validation", {}).get(
            "raw_sequence_entropy_nats"
        )
        if not isinstance(stored_entropy, (int, float)) or not math.isclose(
            entropy_a, float(stored_entropy), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(
                f"Variant A does not reproduce stored entropy for {question_id!r}"
            )
        affected_answer_count += crossing_count
        question_rows.append(
            {
                "question_id": question_id,
                "boundary_crossing_answers": crossing_count,
                "variant_a_entropy_nats": entropy_a,
                "variant_b_entropy_nats": entropy_b,
                "absolute_entropy_difference_nats": abs(entropy_a - entropy_b),
                "variant_a_cluster_probabilities": clusters_a,
                "variant_b_cluster_probabilities": clusters_b,
            }
        )

    variant_a = np.asarray(
        [row["variant_a_entropy_nats"] for row in question_rows], dtype=float
    )
    variant_b = np.asarray(
        [row["variant_b_entropy_nats"] for row in question_rows], dtype=float
    )
    counts = np.asarray(
        [row["boundary_crossing_answers"] for row in question_rows], dtype=float
    )
    absolute_differences = np.abs(variant_a - variant_b)
    correlation = spearmanr(variant_a, variant_b)
    affected_mask = counts > 0
    if affected_mask.any() and (~affected_mask).any():
        concentration = float(
            np.corrcoef(counts, absolute_differences)[0, 1]
        )
    else:
        concentration = float("nan")
    affected_mean_difference = float(absolute_differences[affected_mask].mean())
    unaffected_mean_difference = float(
        absolute_differences[~affected_mask].mean()
    )
    top_indices = np.argsort(-absolute_differences)[:10]

    summary = {
        "questions": len(question_rows),
        "answers": len(question_rows) * 10,
        "answers_with_boundary_crossing": int(affected_answer_count),
        "questions_with_boundary_crossing": int(affected_mask.sum()),
        "variant_a_entropy": {
            "mean": float(variant_a.mean()),
            "std": float(variant_a.std()),
            "minimum": float(variant_a.min()),
            "maximum": float(variant_a.max()),
        },
        "variant_b_entropy": {
            "mean": float(variant_b.mean()),
            "std": float(variant_b.std()),
            "minimum": float(variant_b.min()),
            "maximum": float(variant_b.max()),
        },
        "absolute_difference_nats": {
            "mean": float(absolute_differences.mean()),
            "maximum": float(absolute_differences.max()),
            "over_0_01": int(np.count_nonzero(absolute_differences > 0.01)),
            "over_0_05": int(np.count_nonzero(absolute_differences > 0.05)),
            "over_0_10": int(np.count_nonzero(absolute_differences > 0.10)),
        },
        "spearman": {
            "rho": float(correlation.statistic),
            "p_value": float(correlation.pvalue),
        },
        "entropy_difference_vs_crossing_answer_count_pearson": concentration,
        "mean_absolute_difference_by_question_impact": {
            "affected_questions": affected_mean_difference,
            "unaffected_questions": unaffected_mean_difference,
        },
        "variant_a_reproduction_verified": True,
    }
    top_questions = [
        {
            "question_id": question_rows[index]["question_id"],
            "boundary_crossing_answers": question_rows[index][
                "boundary_crossing_answers"
            ],
            "variant_a_entropy_nats": question_rows[index]["variant_a_entropy_nats"],
            "variant_b_entropy_nats": question_rows[index]["variant_b_entropy_nats"],
            "absolute_entropy_difference_nats": question_rows[index][
                "absolute_entropy_difference_nats"
            ],
        }
        for index in top_indices
    ]

    output = {
        "metadata": {
            "scores_file": str(SCORES_PATH),
            "audit_file": str(AUDIT_PATH),
            "clusters_file": str(CLUSTERS_PATH),
            "variant_a": "existing full-token raw sequence likelihood",
            "variant_b": (
                "exclude first boundary-crossing token; sensitivity approximation"
            ),
            "model_inference_rerun": False,
            "nli_rerun": False,
        },
        "validation": summary,
        "largest_absolute_differences": top_questions,
        "questions": question_rows,
        "interpretation": {
            "boundary_exclusion": (
                "Variant B removes one complete token from the score when its "
                "character offsets cross the prompt boundary. It does not create "
                "an exact character-level conditional likelihood."
            ),
            "recommendation": (
                "Retain Variant A as the primary existing result and report "
                "Variant B as a sensitivity analysis."
            ),
        },
    }
    with OUTPUT_JSON.open("w", encoding="utf-8") as output_file:
        json.dump(output, output_file, indent=2)
        output_file.write("\n")

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as output_file:
        fields = [
            "question_id",
            "boundary_crossing_answers",
            "variant_a_entropy_nats",
            "variant_b_entropy_nats",
            "absolute_entropy_difference_nats",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in fields} for row in question_rows
        )

    report_lines = [
        "# Token Boundary Sensitivity Report",
        "",
        "No model inference, generation, NLI, or reclustering was performed.",
        "",
        "## Validation",
        "",
        f"- Questions: {summary['questions']}",
        f"- Answers: {summary['answers']}",
        f"- Boundary-crossing answers: {summary['answers_with_boundary_crossing']}",
        f"- Affected questions: {summary['questions_with_boundary_crossing']}",
        "- Variant A reproduced the stored probability-weighted entropy values.",
        "",
        "## Entropy comparison",
        "",
        "| Estimator | Mean | Std | Minimum | Maximum |",
        "|---|---:|---:|---:|---:|",
        f"| Variant A | {variant_a.mean():.6f} | {variant_a.std():.6f} | {variant_a.min():.6f} | {variant_a.max():.6f} |",
        f"| Variant B | {variant_b.mean():.6f} | {variant_b.std():.6f} | {variant_b.min():.6f} | {variant_b.max():.6f} |",
        "",
        f"- Mean absolute difference: {absolute_differences.mean():.6f} nats",
        f"- Maximum absolute difference: {absolute_differences.max():.6f} nats",
        f"- Differences > 0.01 nats: {np.count_nonzero(absolute_differences > 0.01)}",
        f"- Differences > 0.05 nats: {np.count_nonzero(absolute_differences > 0.05)}",
        f"- Differences > 0.10 nats: {np.count_nonzero(absolute_differences > 0.10)}",
        f"- Spearman rho: {correlation.statistic:.6f}",
        f"- Pearson correlation of difference with crossing-answer count: {concentration:.6f}",
        f"- Mean absolute difference, affected questions: {affected_mean_difference:.6f} nats",
        f"- Mean absolute difference, unaffected questions: {unaffected_mean_difference:.6f} nats",
        "",
        "## Ten largest differences",
        "",
        "| Question | Crossing answers | Variant A | Variant B | Absolute difference |",
        "|---|---:|---:|---:|---:|",
    ]
    report_lines.extend(
        f"| {row['question_id']} | {row['boundary_crossing_answers']} | "
        f"{row['variant_a_entropy_nats']:.6f} | {row['variant_b_entropy_nats']:.6f} | "
        f"{row['absolute_entropy_difference_nats']:.6f} |"
        for row in top_questions
    )
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Variant B is a sensitivity approximation: removing a complete "
            "boundary-crossing token does not recover an exact answer-only "
            "conditional likelihood. The observed entropy shift is material, "
            "and the question ranking is only moderately stable (Spearman "
            f"rho {correlation.statistic:.6f}); boundary robustness should "
            "therefore be reported before using Variant A for AUROC/AURAC. "
            "Existing Variant A scores remain unchanged.",
        ]
    )
    OUTPUT_REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Answers affected: {summary['answers_with_boundary_crossing']}/1000")
    print(f"Questions affected: {summary['questions_with_boundary_crossing']}/100")
    print(f"Mean absolute entropy difference: {absolute_differences.mean():.6f} nats")
    print(f"Maximum absolute entropy difference: {absolute_differences.max():.6f} nats")
    print(f"Spearman correlation: {correlation.statistic:.6f}")
    print(f"Saved: {OUTPUT_JSON}, {OUTPUT_CSV}, {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
