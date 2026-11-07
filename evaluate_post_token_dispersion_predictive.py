"""Evaluate post-token dispersion as an incremental correctness predictor."""
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_hidden_state_predictive as base


POST_TOKEN_FEATURES_PATH = base.ROOT / (
    "trivia_qa_qwen3_20261024_162539_post_token_hidden_state_features.npz"
)
OUTPUT_METRICS = base.ROOT / "post_token_dispersion_predictive_evaluation.csv"
OUTPUT_PREDICTIONS = base.ROOT / (
    "post_token_dispersion_predictive_oof_predictions.csv"
)
OUTPUT_JSON = base.ROOT / "post_token_dispersion_predictive_evaluation.json"
OUTPUT_REPORT = base.ROOT / "post_token_dispersion_predictive_evaluation_report.md"
BOOTSTRAP_RESAMPLES = 20_000


def load_post_token_dispersion(
    question_ids: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(POST_TOKEN_FEATURES_PATH, allow_pickle=True) as data:
        features = data["features"]
        answer_ids = data["question_id"].astype(str)
        answer_indices = data["answer_index"]

    if features.shape != (1000, 2048):
        raise ValueError(f"Unexpected post-token feature shape: {features.shape}")
    if features.dtype != np.float32:
        raise ValueError(f"Unexpected post-token feature dtype: {features.dtype}")
    if answer_ids.shape != (1000,) or answer_indices.shape != (1000,):
        raise ValueError("Invalid post-token mapping shapes")
    base.finite_array(features, "Post-token features")

    rows_by_question: dict[str, list[tuple[int, int]]] = {
        question_id: [] for question_id in question_ids
    }
    for row, (question_id, answer_index) in enumerate(
        zip(answer_ids, answer_indices)
    ):
        if question_id not in rows_by_question:
            raise ValueError(f"Unknown post-token question ID: {question_id}")
        rows_by_question[question_id].append((int(answer_index), row))

    dispersion_by_id: dict[str, float] = {}
    for question_id, pairs in rows_by_question.items():
        pairs.sort()
        if [answer_index for answer_index, _ in pairs] != list(range(10)):
            raise ValueError(f"Invalid answer mapping for {question_id}")
        answer_vectors = features[[row for _, row in pairs]].astype(np.float64)
        centroid = answer_vectors.mean(axis=0, dtype=np.float64)
        distances = np.linalg.norm(answer_vectors - centroid, axis=1)
        dispersion = float(distances.mean(dtype=np.float64))
        if not np.isfinite(dispersion) or dispersion < 0:
            raise ValueError(f"Invalid dispersion for {question_id}")
        dispersion_by_id[question_id] = dispersion

    if set(dispersion_by_id) != set(question_ids):
        raise ValueError("Post-token question IDs do not match evaluator IDs")
    dispersion = np.asarray(
        [dispersion_by_id[question_id] for question_id in question_ids],
        dtype=np.float64,
    )
    return dispersion, {
        "answer_shape": list(features.shape),
        "answer_dtype": str(features.dtype),
        "dispersion_dtype": str(dispersion.dtype),
        "all_finite": bool(np.isfinite(dispersion).all()),
    }


def metric_difference_bootstrap(
    y: np.ndarray,
    combined_uncertainty: np.ndarray,
    combined_predictions: np.ndarray,
    extended_uncertainty: np.ndarray,
    extended_predictions: np.ndarray,
    question_ids: list[str],
) -> dict[str, Any]:
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
    samples = rng.integers(
        0, len(y), size=(BOOTSTRAP_RESAMPLES, len(y))
    )
    distributions = {name: [] for name in metric_names}
    undefined = {name: 0 for name in metric_names}
    for indices in samples:
        sample_y = y[indices]
        if np.unique(sample_y).size < 2:
            undefined["auroc_incorrect"] += 1
            undefined["spearman_uncertainty_vs_correct"] += 1
            continue
        sample_ids = [question_ids[index] for index in indices]
        combined = base.metric_values(
            sample_y,
            combined_uncertainty[indices],
            combined_predictions[indices],
            sample_ids,
        )
        extended = base.metric_values(
            sample_y,
            extended_uncertainty[indices],
            extended_predictions[indices],
            sample_ids,
        )
        for name in metric_names:
            difference = extended[name] - combined[name]
            distributions[name].append(float(difference))

    intervals: dict[str, list[float]] = {}
    valid_samples: dict[str, int] = {}
    for name, values in distributions.items():
        finite_values = np.asarray(values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        valid_samples[name] = int(finite_values.size)
        if finite_values.size == 0:
            intervals[name] = [float("nan"), float("nan")]
        else:
            intervals[name] = [
                float(np.percentile(finite_values, 2.5)),
                float(np.percentile(finite_values, 97.5)),
            ]
    return {
        "comparison": "semantic_plus_token_entropy_plus_post_token_dispersion "
        "minus semantic_plus_token_entropy",
        "resamples": BOOTSTRAP_RESAMPLES,
        "random_seed": base.RANDOM_SEED,
        "confidence_intervals_95": intervals,
        "valid_samples": valid_samples,
        "undefined_samples": undefined,
    }


def main() -> None:
    question_ids, _, y, semantic, token = base.load_inputs()
    dispersion, dispersion_audit = load_post_token_dispersion(question_ids)
    configurations = {
        "semantic_entropy": semantic[:, None],
        "token_entropy": token[:, None],
        "semantic_plus_token_entropy": np.column_stack((semantic, token)),
        "semantic_plus_token_entropy_plus_post_token_dispersion": np.column_stack(
            (semantic, token, dispersion)
        ),
    }
    results: dict[str, dict[str, Any]] = {}
    prediction_rows: list[dict[str, Any]] = []
    for name, features in configurations.items():
        scores, predictions, best_cs = base.nested_oof(features, y)
        uncertainty = 1.0 - scores
        metrics = base.metric_values(y, uncertainty, predictions, question_ids)
        results[name] = {
            "configuration": name,
            "feature_dimension": int(features.shape[1]),
            "best_C_by_outer_fold": best_cs,
            **metrics,
        }
        prediction_rows.extend(
            {
                "configuration": name,
                "question_id": question_id,
                "correct": int(label),
                "predicted_correct_probability": float(score),
                "uncertainty_score": float(1.0 - score),
                "predicted_correct_at_0_5": int(prediction),
            }
            for question_id, label, score, prediction in zip(
                question_ids, y, scores, predictions
            )
        )

    combined = results["semantic_plus_token_entropy"]
    extended = results[
        "semantic_plus_token_entropy_plus_post_token_dispersion"
    ]
    prediction_map: dict[str, dict[str, np.ndarray]] = {}
    for row in prediction_rows:
        prediction_map.setdefault(row["configuration"], {}).setdefault(
            "uncertainty", []
        ).append(row["uncertainty_score"])
        prediction_map[row["configuration"]].setdefault(
            "prediction", []
        ).append(row["predicted_correct_at_0_5"])
    combined_uncertainty = np.asarray(
        prediction_map["semantic_plus_token_entropy"]["uncertainty"]
    )
    combined_predictions = np.asarray(
        prediction_map["semantic_plus_token_entropy"]["prediction"]
    )
    extended_uncertainty = np.asarray(
        prediction_map[
            "semantic_plus_token_entropy_plus_post_token_dispersion"
        ]["uncertainty"]
    )
    extended_predictions = np.asarray(
        prediction_map[
            "semantic_plus_token_entropy_plus_post_token_dispersion"
        ]["prediction"]
    )
    paired_bootstrap = metric_difference_bootstrap(
        y,
        combined_uncertainty,
        combined_predictions,
        extended_uncertainty,
        extended_predictions,
        question_ids,
    )

    rows = []
    for name, result in results.items():
        rows.append(result)
    with OUTPUT_METRICS.open("w", encoding="utf-8", newline="") as file:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with OUTPUT_PREDICTIONS.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(prediction_rows[0])
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    output = {
        "experiment": "post_token_dispersion_predictive_evaluation",
        "question_unit": "one independent question-level sample",
        "configurations": results,
        "paired_bootstrap_metric_differences": paired_bootstrap,
        "data_audit": {
            "questions": len(question_ids),
            "correct": int(y.sum()),
            "incorrect": int((1 - y).sum()),
            "question_ids_match": True,
            "post_token": dispersion_audit,
            "all_features_finite": True,
        },
        "settings": {
            "outer_folds": base.OUTER_FOLDS,
            "inner_folds": base.INNER_FOLDS,
            "shuffle": True,
            "random_state": base.RANDOM_SEED,
            "regularization_grid": list(base.REGULARIZATION_GRID),
            "scaler": "StandardScaler inside Pipeline",
            "classifier": "LogisticRegression solver=lbfgs max_iter=1000",
            "selection_metric": "inner-fold AUROC",
            "uncertainty": "1 - predicted probability of correctness",
            "post_token_dispersion_arithmetic": "float64",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        },
    }
    OUTPUT_JSON.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_lines = [
        "# Post-Token Dispersion Predictive Evaluation",
        "",
        "| Configuration | AUROC | AURAC | Spearman | Accuracy | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        report_lines.append(
            f"| {name} | {result['auroc_incorrect']:.6f} | "
            f"{result['aurac']:.6f} | "
            f"{result['spearman_uncertainty_vs_correct']:.6f} | "
            f"{result['accuracy_at_0_5']:.6f} | {result['f1_at_0_5']:.6f} |"
        )
    report_lines.extend(
        [
            "",
            "## Paired difference",
            "",
            "Difference is extended configuration minus semantic + token entropy.",
            "",
            "```json",
            json.dumps(paired_bootstrap, indent=2),
            "```",
            "",
            "All preprocessing was fitted within training folds. The four "
            "configurations used identical outer and inner split seeds.",
        ]
    )
    OUTPUT_REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"Saved metrics to {OUTPUT_METRICS}")
    print(f"Saved predictions to {OUTPUT_PREDICTIONS}")
    print(f"Saved results to {OUTPUT_JSON}")
    print(f"Saved report to {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
