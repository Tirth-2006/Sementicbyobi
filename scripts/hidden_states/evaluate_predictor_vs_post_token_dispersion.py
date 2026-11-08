"""Compare predictor-state and post-token dispersion as uncertainty features."""
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluation import evaluate_hidden_state_predictive as base


ORIGINAL_ANSWER_PATH = base.ROOT / (
    "trivia_qa_qwen3_20261024_162539_hidden_state_features.npz"
)
POST_TOKEN_ANSWER_PATH = base.ROOT / (
    "trivia_qa_qwen3_20261024_162539_post_token_hidden_state_features.npz"
)
OUTPUT_METRICS = base.ROOT / "predictor_vs_post_token_dispersion_metrics.csv"
OUTPUT_PREDICTIONS = base.ROOT / "predictor_vs_post_token_dispersion_oof.csv"
OUTPUT_JSON = base.ROOT / "predictor_vs_post_token_dispersion_results.json"
OUTPUT_REPORT = base.ROOT / "predictor_vs_post_token_dispersion_report.md"
BOOTSTRAP_RESAMPLES = 20_000
HIDDEN_STATE_INDEX = 28
HIDDEN_SIZE = 2048


def dispersion_from_npz(
    path: Path,
    question_ids: list[str],
    feature_selector: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=True) as data:
        features = data["features"]
        answer_ids = data["question_id"].astype(str)
        answer_indices = data["answer_index"]
    if answer_ids.shape != (1000,) or answer_indices.shape != (1000,):
        raise ValueError(f"Invalid mapping shapes in {path}")
    base.finite_array(features, f"Features from {path}")
    rows_by_question: dict[str, list[tuple[int, int]]] = {
        question_id: [] for question_id in question_ids
    }
    for row, (question_id, answer_index) in enumerate(
        zip(answer_ids, answer_indices)
    ):
        if question_id not in rows_by_question:
            raise ValueError(f"Unknown question ID in {path}: {question_id}")
        rows_by_question[question_id].append((int(answer_index), row))

    dispersions: list[float] = []
    for question_id in question_ids:
        pairs = sorted(rows_by_question[question_id])
        if [index for index, _ in pairs] != list(range(10)):
            raise ValueError(f"Invalid answer indices for {question_id} in {path}")
        rows = [row for _, row in pairs]
        vectors = feature_selector(features[rows]).astype(np.float64)
        if vectors.shape != (10, HIDDEN_SIZE):
            raise ValueError(f"Unexpected selected shape in {path}: {vectors.shape}")
        centroid = vectors.mean(axis=0, dtype=np.float64)
        distances = np.linalg.norm(vectors - centroid, axis=1)
        dispersion = float(distances.mean(dtype=np.float64))
        if not np.isfinite(dispersion) or dispersion < 0:
            raise ValueError(f"Invalid dispersion for {question_id}")
        dispersions.append(dispersion)
    result = np.asarray(dispersions, dtype=np.float64)
    return result, {
        "path": str(path),
        "input_shape": list(features.shape),
        "input_dtype": str(features.dtype),
        "dispersion_dtype": str(result.dtype),
        "all_finite": bool(np.isfinite(result).all()),
    }


def paired_bootstrap(
    y: np.ndarray,
    scores: dict[str, dict[str, np.ndarray]],
    question_ids: list[str],
) -> dict[str, Any]:
    comparisons = {"B_minus_A": ("B", "A"), "C_minus_A": ("C", "A"), "C_minus_B": ("C", "B")}
    metric_names = (
        "auroc_incorrect",
        "aurac",
        "spearman_uncertainty_vs_correct",
        "accuracy_at_0_5",
        "precision_at_0_5",
        "recall_at_0_5",
        "f1_at_0_5",
    )
    rng = np.random.default_rng(base.RANDOM_SEED)
    samples = rng.integers(0, len(y), size=(BOOTSTRAP_RESAMPLES, len(y)))
    output: dict[str, Any] = {}
    for comparison, (left, right) in comparisons.items():
        distributions = {metric: [] for metric in metric_names}
        undefined = {metric: 0 for metric in metric_names}
        for indices in samples:
            sample_y = y[indices]
            if np.unique(sample_y).size < 2:
                undefined["auroc_incorrect"] += 1
                undefined["spearman_uncertainty_vs_correct"] += 1
                continue
            sample_ids = [question_ids[index] for index in indices]
            left_metrics = base.metric_values(
                sample_y,
                scores[left]["uncertainty"][indices],
                scores[left]["prediction"][indices],
                sample_ids,
            )
            right_metrics = base.metric_values(
                sample_y,
                scores[right]["uncertainty"][indices],
                scores[right]["prediction"][indices],
                sample_ids,
            )
            for metric in metric_names:
                distributions[metric].append(
                    float(left_metrics[metric] - right_metrics[metric])
                )
        intervals = {}
        valid = {}
        for metric, values in distributions.items():
            finite = np.asarray(values, dtype=float)
            finite = finite[np.isfinite(finite)]
            valid[metric] = int(finite.size)
            intervals[metric] = [
                float(np.percentile(finite, 2.5)),
                float(np.percentile(finite, 97.5)),
            ]
        output[comparison] = {
            "difference": f"{left} minus {right}",
            "confidence_intervals_95": intervals,
            "valid_samples": valid,
            "undefined_samples": undefined,
        }
    return {
        "resamples": BOOTSTRAP_RESAMPLES,
        "random_seed": base.RANDOM_SEED,
        "comparisons": output,
    }


def main() -> None:
    question_ids, _, y, semantic, token = base.load_inputs()
    metadata = base.load_json(base.HIDDEN_METADATA_PATH)
    pooling = metadata.get("pooling_strategies")
    if not isinstance(pooling, list) or "mean" not in pooling:
        raise ValueError("Original metadata does not define mean pooling")
    mean_pooling_index = pooling.index("mean")
    predictor_dispersion, predictor_audit = dispersion_from_npz(
        ORIGINAL_ANSWER_PATH,
        question_ids,
        lambda values: values[:, HIDDEN_STATE_INDEX, mean_pooling_index, :],
    )
    post_token_dispersion, post_token_audit = dispersion_from_npz(
        POST_TOKEN_ANSWER_PATH,
        question_ids,
        lambda values: values,
    )
    features = {
        "A": np.column_stack((semantic, token)),
        "B": np.column_stack((semantic, token, predictor_dispersion)),
        "C": np.column_stack((semantic, token, post_token_dispersion)),
    }
    names = {
        "A": "semantic_plus_token_entropy",
        "B": "semantic_plus_token_entropy_plus_predictor_dispersion",
        "C": "semantic_plus_token_entropy_plus_post_token_dispersion",
    }
    results: dict[str, dict[str, Any]] = {}
    scores: dict[str, dict[str, np.ndarray]] = {}
    prediction_rows: list[dict[str, Any]] = []
    for key in ("A", "B", "C"):
        probability, prediction, best_c = base.nested_oof(features[key], y)
        uncertainty = 1.0 - probability
        scores[key] = {"uncertainty": uncertainty, "prediction": prediction}
        results[key] = {
            "configuration": names[key],
            "feature_dimension": int(features[key].shape[1]),
            "best_C_by_outer_fold": best_c,
            **base.metric_values(y, uncertainty, prediction, question_ids),
        }
        prediction_rows.extend(
            {
                "configuration": names[key],
                "question_id": question_id,
                "correct": int(label),
                "predicted_correct_probability": float(probability_value),
                "uncertainty_score": float(uncertainty_value),
                "predicted_correct_at_0_5": int(prediction_value),
            }
            for question_id, label, probability_value, uncertainty_value, prediction_value in zip(
                question_ids, y, probability, uncertainty, prediction
            )
        )
    bootstrap = paired_bootstrap(y, scores, question_ids)

    rows = [results[key] for key in ("A", "B", "C")]
    with OUTPUT_METRICS.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with OUTPUT_PREDICTIONS.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)

    output = {
        "experiment": "predictor_vs_post_token_dispersion",
        "configurations": results,
        "paired_bootstrap": bootstrap,
        "data_audit": {
            "questions": len(question_ids),
            "correct": int(y.sum()),
            "incorrect": int((1 - y).sum()),
            "question_ids_match": True,
            "mean_pooling_index": mean_pooling_index,
            "predictor_dispersion": predictor_audit,
            "post_token_dispersion": post_token_audit,
            "all_features_finite": True,
        },
        "settings": {
            "outer_folds": base.OUTER_FOLDS,
            "inner_folds": base.INNER_FOLDS,
            "shuffle": True,
            "random_state": base.RANDOM_SEED,
            "regularization_grid": list(base.REGULARIZATION_GRID),
            "classifier": "LogisticRegression solver=lbfgs max_iter=1000",
            "scaler": "StandardScaler inside Pipeline",
            "selection_metric": "inner-fold AUROC",
            "uncertainty": "1 - P(correct)",
            "dispersion_arithmetic": "float64",
        },
    }
    OUTPUT_JSON.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Predictor-State vs Post-Token Dispersion",
        "",
        "| Configuration | AUROC | AURAC | Spearman | Accuracy | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("A", "B", "C"):
        row = results[key]
        lines.append(
            f"| {row['configuration']} | {row['auroc_incorrect']:.6f} | "
            f"{row['aurac']:.6f} | {row['spearman_uncertainty_vs_correct']:.6f} | "
            f"{row['accuracy_at_0_5']:.6f} | {row['precision_at_0_5']:.6f} | "
            f"{row['recall_at_0_5']:.6f} | {row['f1_at_0_5']:.6f} |"
        )
    lines.extend(["", "## Paired bootstrap intervals", "", "```json", json.dumps(bootstrap, indent=2), "```"])
    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"Saved metrics to {OUTPUT_METRICS}")
    print(f"Saved predictions to {OUTPUT_PREDICTIONS}")
    print(f"Saved results to {OUTPUT_JSON}")
    print(f"Saved report to {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
