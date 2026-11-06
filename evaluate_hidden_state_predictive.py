"""Evaluate hidden-state representations for question-level correctness prediction."""
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("data/generated")
HIDDEN_FEATURES_PATH = ROOT / (
    "trivia_qa_qwen3_20261024_162539_hidden_state_question_features.npz"
)
HIDDEN_METADATA_PATH = ROOT / (
    "trivia_qa_qwen3_20261024_162539_hidden_state_metadata.json"
)
CORRECTNESS_PATH = ROOT / (
    "trivia_qa_qwen3_20261024_162539_low_temp_correctness.json"
)
TOKEN_ENTROPY_PATH = ROOT / (
    "trivia_qa_qwen3_20261024_162539_token_entropy_question.csv"
)
SEMANTIC_ENTROPY_PATH = ROOT / (
    "trivia_qa_qwen3_20261024_162539_clusters.json"
)
OUTPUT_METRICS = ROOT / "hidden_state_predictive_evaluation.csv"
OUTPUT_PREDICTIONS = ROOT / "hidden_state_predictive_oof_predictions.csv"
OUTPUT_JSON = ROOT / "hidden_state_predictive_evaluation.json"
OUTPUT_REPORT = ROOT / "hidden_state_predictive_evaluation_report.md"

EXPECTED_QUESTIONS = 100
EXPECTED_HIDDEN_INDICES = 29
EXPECTED_HIDDEN_SIZE = 2048
EXPECTED_POOLING = ("first", "last", "mean", "max")
OUTER_FOLDS = 2
INNER_FOLDS = 2
REGULARIZATION_GRID = (0.1, 1.0)
RANDOM_SEED = 20261106
BOOTSTRAP_RESAMPLES = 100
CLASSIFICATION_THRESHOLD = 0.5


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def finite_array(values: np.ndarray, name: str) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")


def load_inputs() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    metadata = load_json(HIDDEN_METADATA_PATH)
    if metadata.get("num_questions") != EXPECTED_QUESTIONS:
        raise ValueError("Hidden-state metadata question count is invalid")
    if metadata.get("num_hidden_state_indices") != EXPECTED_HIDDEN_INDICES:
        raise ValueError("Hidden-state metadata index count is invalid")
    if metadata.get("hidden_size") != EXPECTED_HIDDEN_SIZE:
        raise ValueError("Hidden-state metadata dimension is invalid")
    if tuple(metadata.get("pooling_strategies", ())) != EXPECTED_POOLING:
        raise ValueError("Hidden-state pooling labels are invalid")

    with np.load(HIDDEN_FEATURES_PATH, allow_pickle=False) as data:
        features = data["features"]
        question_ids = data["question_id"].astype(str)
        saved_correct = data["correct"]
    expected_shape = (
        EXPECTED_QUESTIONS,
        EXPECTED_HIDDEN_INDICES,
        len(EXPECTED_POOLING),
        EXPECTED_HIDDEN_SIZE,
    )
    if features.shape != expected_shape:
        raise ValueError(f"Unexpected hidden-state shape: {features.shape}")
    if features.dtype != np.float32:
        raise ValueError(f"Expected float32 hidden-state features, got {features.dtype}")
    if question_ids.shape != (EXPECTED_QUESTIONS,):
        raise ValueError("Hidden-state question IDs have the wrong shape")
    if saved_correct.shape != (EXPECTED_QUESTIONS,):
        raise ValueError("Hidden-state correctness labels have the wrong shape")
    finite_array(features, "Hidden-state features")

    correctness = load_json(CORRECTNESS_PATH).get("questions")
    if not isinstance(correctness, list) or len(correctness) != EXPECTED_QUESTIONS:
        raise ValueError("Correctness artifact must contain 100 records")
    correctness_by_id: dict[str, int] = {}
    for item in correctness:
        if not isinstance(item, dict):
            raise ValueError("Invalid correctness record")
        question_id = item.get("question_id")
        correct = item.get("correct")
        if (
            not isinstance(question_id, str)
            or question_id in correctness_by_id
            or type(correct) is not int
            or correct not in (0, 1)
        ):
            raise ValueError("Correctness records must have unique binary labels")
        correctness_by_id[question_id] = correct

    with TOKEN_ENTROPY_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        token_rows = list(csv.DictReader(file))
    token_by_id: dict[str, tuple[float, int]] = {}
    for row in token_rows:
        question_id = row.get("question_id")
        value = row.get("mean_token_entropy")
        correct_text = row.get("correct")
        if (
            not isinstance(question_id, str)
            or question_id in token_by_id
            or not isinstance(value, str)
            or correct_text not in {"0", "1"}
        ):
            raise ValueError("Invalid token entropy CSV record")
        entropy = float(value)
        if not math.isfinite(entropy) or entropy < 0:
            raise ValueError(f"Invalid token entropy for {question_id}")
        token_by_id[question_id] = (entropy, int(correct_text))

    semantic_questions = load_json(SEMANTIC_ENTROPY_PATH).get("questions")
    if not isinstance(semantic_questions, dict):
        raise ValueError("Semantic entropy artifact has no question mapping")
    semantic_by_id: dict[str, float] = {}
    for question_id, item in semantic_questions.items():
        if not isinstance(item, dict):
            raise ValueError(f"Invalid semantic entropy record for {question_id}")
        value = item.get("entropy_nats")
        if not isinstance(value, (int, float)):
            raise ValueError(f"Invalid semantic entropy for {question_id}")
        entropy = float(value)
        if not math.isfinite(entropy) or entropy < 0:
            raise ValueError(f"Invalid semantic entropy for {question_id}")
        semantic_by_id[question_id] = entropy

    hidden_ids = set(question_ids.tolist())
    if (
        hidden_ids
        != set(correctness_by_id)
        or hidden_ids != set(token_by_id)
        or hidden_ids != set(semantic_by_id)
    ):
        raise ValueError("Question IDs do not align across evaluation artifacts")
    for question_id, (_, token_correct) in token_by_id.items():
        if token_correct != correctness_by_id[question_id]:
            raise ValueError(
                f"Token entropy correctness disagrees for {question_id}"
            )
    y = np.asarray([correctness_by_id[question_id] for question_id in question_ids])
    if not np.array_equal(saved_correct, y):
        raise ValueError("Saved hidden-state labels disagree with correctness source")
    if np.unique(y).size != 2 or int(y.sum()) != 27:
        raise ValueError("Expected both labels with 27 correct questions")

    semantic = np.asarray([semantic_by_id[question_id] for question_id in question_ids])
    token = np.asarray(
        [token_by_id[question_id][0] for question_id in question_ids]
    )
    finite_array(semantic, "Semantic entropy")
    finite_array(token, "Token entropy")
    return question_ids.tolist(), features, y.astype(int), semantic, token


def aurac(correct: np.ndarray, uncertainty: np.ndarray, question_ids: list[str]) -> float:
    order = np.asarray(
        sorted(
            range(len(correct)),
            key=lambda index: (-float(uncertainty[index]), question_ids[index]),
        ),
        dtype=int,
    )
    rejected = np.concatenate(([0], np.cumsum(correct[order], dtype=int)))
    retained = len(correct) - np.arange(len(correct) + 1)
    retained_correct = int(correct.sum()) - rejected
    accuracy = np.full(len(correct) + 1, np.nan, dtype=float)
    accuracy[:-1] = retained_correct[:-1] / retained[:-1]
    integration = accuracy.copy()
    integration[-1] = accuracy[-2]
    coverage = retained / len(correct)
    return float(np.trapezoid(integration[::-1], coverage[::-1]))


def metric_values(
    y: np.ndarray,
    uncertainty: np.ndarray,
    prediction: np.ndarray,
    question_ids: list[str],
) -> dict[str, float]:
    incorrect = 1 - y
    if np.unique(incorrect).size < 2:
        raise ValueError("Both correctness classes are required for AUROC")
    spearman = spearmanr(uncertainty, y).statistic
    return {
        "auroc_incorrect": float(roc_auc_score(incorrect, uncertainty)),
        "aurac": aurac(y, uncertainty, question_ids),
        "spearman_uncertainty_vs_correct": float(spearman),
        "accuracy_at_0_5": float(accuracy_score(y, prediction)),
        "precision_at_0_5": float(precision_score(y, prediction, zero_division=0)),
        "recall_at_0_5": float(recall_score(y, prediction, zero_division=0)),
        "f1_at_0_5": float(f1_score(y, prediction, zero_division=0)),
    }


def bootstrap_intervals(
    y: np.ndarray,
    uncertainty: np.ndarray,
    prediction: np.ndarray,
    question_ids: list[str],
) -> dict[str, list[float]]:
    rng = np.random.default_rng(RANDOM_SEED)
    samples = rng.integers(0, len(y), size=(BOOTSTRAP_RESAMPLES, len(y)))
    values = {key: [] for key in ("auroc_incorrect", "aurac", "spearman_uncertainty_vs_correct")}
    for indices in samples:
        sample_y = y[indices]
        if np.unique(sample_y).size < 2:
            continue
        sample_ids = [question_ids[index] for index in indices]
        sample_uncertainty = uncertainty[indices]
        sample_prediction = prediction[indices]
        metrics = metric_values(
            sample_y, sample_uncertainty, sample_prediction, sample_ids
        )
        for key in values:
            values[key].append(metrics[key])
    return {
        f"{key}_ci95": [
            float(np.percentile(np.asarray(metric_values_list), 2.5)),
            float(np.percentile(np.asarray(metric_values_list), 97.5)),
        ]
        for key, metric_values_list in values.items()
    }


def make_model(
    features: np.ndarray,
) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def nested_oof(
    features: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    outer = StratifiedKFold(
        n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_SEED
    )
    scores = np.full(len(y), np.nan, dtype=float)
    best_c_by_fold: list[float] = []
    for fold, (train, test) in enumerate(outer.split(features, y)):
        inner = StratifiedKFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=RANDOM_SEED + fold + 1,
        )
        search = GridSearchCV(
            make_model(features),
            {"classifier__C": REGULARIZATION_GRID},
            scoring="roc_auc",
            cv=inner,
            refit=True,
            n_jobs=-1,
        )
        search.fit(features[train], y[train])
        scores[test] = search.predict_proba(features[test])[:, 1]
        best_c_by_fold.append(float(search.best_params_["classifier__C"]))
    if not np.isfinite(scores).all():
        raise ValueError("Nested evaluation did not produce all OOF scores")
    predictions = (scores >= CLASSIFICATION_THRESHOLD).astype(int)
    return scores, predictions, best_c_by_fold


def configuration_features(
    hidden: np.ndarray,
    semantic: np.ndarray,
    token: np.ndarray,
) -> list[tuple[str, str, np.ndarray]]:
    configurations: list[tuple[str, str, np.ndarray]] = []
    for hidden_index in range(hidden.shape[1]):
        for pooling_index, pooling in enumerate(EXPECTED_POOLING):
            configurations.append(
                (
                    f"hidden_state_{hidden_index}_{pooling}",
                    "hidden_state",
                    hidden[:, hidden_index, pooling_index, :],
                )
            )
    configurations.extend(
        [
            ("semantic_entropy", "semantic_entropy", semantic[:, None]),
            ("token_entropy", "token_entropy", token[:, None]),
            (
                "semantic_plus_token_entropy",
                "entropy_combined",
                np.column_stack((semantic, token)),
            ),
        ]
    )
    for hidden_index in range(hidden.shape[1]):
        for pooling_index, pooling in enumerate(EXPECTED_POOLING):
            configurations.append(
                (
                    f"hidden_state_{hidden_index}_{pooling}_plus_entropy",
                    "hidden_plus_entropy",
                    np.column_stack(
                        (
                            hidden[:, hidden_index, pooling_index, :],
                            semantic,
                            token,
                        )
                    ),
                )
            )
    return configurations


def main() -> None:
    question_ids, hidden, y, semantic, token = load_inputs()
    configurations = configuration_features(hidden, semantic, token)
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    for configuration_index, (name, family, features) in enumerate(configurations):
        scores, predictions, best_cs = nested_oof(features, y)
        uncertainty = 1.0 - scores
        metrics = metric_values(y, uncertainty, predictions, question_ids)
        intervals = bootstrap_intervals(
            y, uncertainty, predictions, question_ids
        )
        hidden_index = ""
        pooling = ""
        if name.startswith("hidden_state_"):
            parts = name.split("_")
            hidden_index = parts[2]
            pooling = parts[3]
        row = {
            "configuration": name,
            "family": family,
            "hidden_state_index": hidden_index,
            "pooling_strategy": pooling,
            "feature_dimension": int(features.shape[1]),
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "best_C_by_outer_fold": json.dumps(best_cs),
            **metrics,
            **intervals,
        }
        metrics_rows.append(row)
        result_rows.append(row)
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
        print(f"Evaluated {configuration_index + 1}/{len(configurations)}: {name}")

    with OUTPUT_METRICS.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics_rows[0]))
        writer.writeheader()
        writer.writerows(metrics_rows)
    with OUTPUT_PREDICTIONS.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)

    results = {
        "experiment": "hidden_state_predictive_evaluation",
        "question_unit": "one independent question-level sample",
        "inputs": {
            "hidden_features": str(HIDDEN_FEATURES_PATH),
            "correctness": str(CORRECTNESS_PATH),
            "semantic_entropy": str(SEMANTIC_ENTROPY_PATH),
            "token_entropy": str(TOKEN_ENTROPY_PATH),
        },
        "data_audit": {
            "questions": len(question_ids),
            "correct": int(y.sum()),
            "incorrect": int((1 - y).sum()),
            "hidden_shape": list(hidden.shape),
            "hidden_dtype": str(hidden.dtype),
            "all_features_finite": bool(
                np.isfinite(hidden).all()
                and np.isfinite(semantic).all()
                and np.isfinite(token).all()
            ),
        },
        "settings": {
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "regularization_grid": list(REGULARIZATION_GRID),
            "scaler": "StandardScaler inside Pipeline and fitted on training folds",
            "classifier": "LogisticRegression solver=lbfgs",
            "selection_metric": "training-fold inner-CV AUROC",
            "primary_metric": "AUROC for incorrectness using 1 - P(correct)",
            "spearman_definition": "Spearman(uncertainty score, correctness), expected negative",
            "aurac": "highest uncertainty rejected first; question_id ascending tie-break",
            "classification_threshold": CLASSIFICATION_THRESHOLD,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "random_seed": RANDOM_SEED,
        },
        "configurations": result_rows,
    }
    OUTPUT_JSON.write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_report(results)


def write_report(results: dict[str, Any]) -> None:
    rows = results["configurations"]
    baselines = {
        row["configuration"]: row
        for row in rows
        if row["configuration"]
        in {"semantic_entropy", "token_entropy", "semantic_plus_token_entropy"}
    }
    lines = [
        "# Hidden-State Predictive Evaluation",
        "",
        "## Research question",
        "",
        "This experiment tests whether question-level hidden-state representations "
        "predict the correctness of a separate low-temperature answer. It does not "
        "establish causality or generalization beyond these 100 questions.",
        "",
        "## Data audit",
        "",
        f"- Questions: {results['data_audit']['questions']}.",
        f"- Correct/incorrect labels: {results['data_audit']['correct']}/"
        f"{results['data_audit']['incorrect']}.",
        f"- Hidden feature shape and dtype: `{results['data_audit']['hidden_shape']}`, "
        f"`{results['data_audit']['hidden_dtype']}`.",
        "- Question IDs and labels were joined by exact `question_id` across all "
        "input artifacts; no original artifact was rewritten.",
        "",
        "## Methodology",
        "",
        "- Each question is one independent sample; the 1,000 high-temperature "
        "answers are not treated as independent labeled examples.",
        f"- Every configuration uses {OUTER_FOLDS}-fold outer stratified "
        "cross-validation. "
        f"Regularization is selected by {INNER_FOLDS}-fold inner stratified CV "
        "on the outer-training fold only.",
        "- Standardization is inside the scikit-learn pipeline and is fitted only "
        "on each training fold.",
        "- AUROC is for incorrectness, using uncertainty `1 - P(correct)`.",
        "- Spearman is `Spearman(uncertainty, correctness)`; more uncertainty "
        "should correspond to lower correctness, so negative values are expected.",
        "- AURAC rejects highest uncertainty first and breaks exact ties by "
        "ascending question ID.",
        f"- Classification metrics use threshold `P(correct) >= "
        f"{CLASSIFICATION_THRESHOLD}`.",
        f"- Confidence intervals use {BOOTSTRAP_RESAMPLES:,} question-level "
        f"resamples with seed {RANDOM_SEED}.",
        "",
        "## Results",
        "",
        f"All {len(rows)} configurations are saved in the accompanying CSV and JSON. "
        "The table below gives the baseline point estimates:",
        "",
        "| Configuration | AUROC incorrect | AURAC | Spearman | Accuracy | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in (
        "semantic_entropy",
        "token_entropy",
        "semantic_plus_token_entropy",
    ):
        row = baselines[name]
        lines.append(
            f"| {name} | {row['auroc_incorrect']:.4f} | {row['aurac']:.4f} | "
            f"{row['spearman_uncertainty_vs_correct']:.4f} | "
            f"{row['accuracy_at_0_5']:.4f} | {row['f1_at_0_5']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Hidden-state configurations are reported exhaustively rather than "
            "selecting only the best result. Because 116 hidden-state "
            "configurations are screened, the highest observed score is "
            "descriptive and is not by itself evidence of statistical superiority. "
            "No multiplicity-adjusted claim is made.",
            "",
            "## Limitations",
            "",
            "- The sample contains only 100 questions and 27 correct labels.",
            "- OOF predictions support leakage-controlled evaluation, but the "
            "small sample makes confidence intervals wide.",
            "- Bootstrap intervals quantify question-level resampling variability; "
            "they do not correct for all model-selection uncertainty across the "
            "116 hidden-state configurations.",
            "- Hidden states are representations, not direct uncertainty measures.",
            "- No claim is made about causal effects or performance on new datasets.",
            "",
            "Complete per-configuration metrics, OOF predictions, settings, and "
            "bootstrap intervals are saved in the accompanying output files.",
        ]
    )
    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
