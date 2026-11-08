"""Describe hidden-state feature scale and pooling similarity by index."""
import json
from pathlib import Path

import numpy as np


FEATURES_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_hidden_state_features.npz"
)
METADATA_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_hidden_state_metadata.json"
)
OUTPUT_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_hidden_state_analysis.json"
)
POOLING = ("first", "last", "mean", "max")
REPORT_SECTION_START = "## Hidden-state index and pooling characterization"
REPORT_SECTION_END = "## End hidden-state index characterization"


def main() -> None:
    with METADATA_PATH.open(encoding="utf-8") as file:
        metadata = json.load(file)
    data = np.load(FEATURES_PATH, allow_pickle=True)
    features = data["features"].astype(np.float32)
    question_ids = data["question_id"].astype(str)

    hidden_state_index_count = features.shape[1]
    answer_count = features.shape[0]
    question_indices = {
        question_id: np.flatnonzero(question_ids == question_id)
        for question_id in sorted(set(question_ids))
    }

    norms = np.linalg.norm(features, axis=-1)
    mean_norms = norms.mean(axis=0)
    std_norms = norms.std(axis=0)

    dispersion = np.empty(
        (hidden_state_index_count, len(POOLING)), dtype=np.float64
    )
    for layer in range(hidden_state_index_count):
        for pooling in range(len(POOLING)):
            values = []
            for indices in question_indices.values():
                question_mean = features[indices, layer, pooling].mean(axis=0)
                values.extend(
                    np.linalg.norm(
                        features[indices, layer, pooling] - question_mean,
                        axis=-1,
                    )
                )
            dispersion[layer, pooling] = float(np.mean(values))

    cosine_similarity = {}
    for left_index, left in enumerate(POOLING):
        for right_index in range(left_index + 1, len(POOLING)):
            right = POOLING[right_index]
            left_values = features[:, :, left_index]
            right_values = features[:, :, right_index]
            numerator = np.sum(left_values * right_values, axis=-1)
            denominator = np.linalg.norm(left_values, axis=-1) * np.linalg.norm(
                right_values, axis=-1
            )
            values = numerator / np.maximum(denominator, 1e-12)
            cosine_similarity[f"{left}_vs_{right}"] = values.mean(axis=0).tolist()

    result = {
        "feature_file": str(FEATURES_PATH),
        "question_count": len(question_indices),
        "answer_count": answer_count,
        "pooling_strategies": list(POOLING),
        "hidden_state_index_count": hidden_state_index_count,
        "mean_l2_norm_by_hidden_state_index_pooling": mean_norms.tolist(),
        "std_l2_norm_by_hidden_state_index_pooling": std_norms.tolist(),
        "mean_within_question_l2_dispersion_by_hidden_state_index_pooling": dispersion.tolist(),
        "mean_cosine_similarity_by_hidden_state_index": cosine_similarity,
        "interpretation": (
            "These are descriptive representation-scale and pooling-similarity "
            "statistics. They are not correctness-prediction results."
        ),
    }
    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    report_path = Path(
        metadata.get(
            "report_file",
            "data/generated/trivia_qa_qwen3_20261024_162539_hidden_state_report.md",
        )
    )
    report = report_path.read_text(encoding="utf-8").rstrip()
    if REPORT_SECTION_START in report:
        report = report.split(REPORT_SECTION_START, 1)[0].rstrip()
    report += f"\n\n{REPORT_SECTION_START}\n\n"
    report += (
        "The following summaries describe representation scale and agreement "
        "among pooling strategies; they are not predictive performance metrics.\n\n"
    )
    report += "| Hidden-state index | Mean norm first | last | mean | max | Mean within-question dispersion first | last | mean | max |\n"
    report += "|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for layer in range(hidden_state_index_count):
        report += (
            f"| {layer} | "
            + " | ".join(f"{mean_norms[layer, p]:.4f}" for p in range(4))
            + " | "
            + " | ".join(f"{dispersion[layer, p]:.4f}" for p in range(4))
            + " |\n"
        )
    report += "\nMean cosine similarity between pooling strategies by hidden-state index:\n\n"
    report += "| Hidden-state index | first-last | first-mean | first-max | last-mean | last-max | mean-max |\n"
    report += "|---:|---:|---:|---:|---:|---:|---:|\n"
    pairs = tuple(cosine_similarity)
    for layer in range(hidden_state_index_count):
        report += f"| {layer} | " + " | ".join(
            f"{cosine_similarity[pair][layer]:.4f}" for pair in pairs
        ) + " |\n"
    report += (
        "\nIndex 0 is the token embedding output, indices 1–27 are outputs of "
        "decoder blocks 1–27, and index 28 is the final RMSNorm-normalized "
        "output after decoder block 28. There are 28 decoder blocks in total. "
        "Large norm or cosine differences characterize the "
        "representations but do not establish usefulness for correctness "
        "prediction. A future probe evaluation must use question-grouped "
        f"cross-validation.\n\n{REPORT_SECTION_END}\n"
    )
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved descriptive analysis to {OUTPUT_PATH}")
    print(f"Updated report: {report_path}")
    print(
        "Highest mean-norm layer/pooling: "
        f"{np.unravel_index(np.argmax(mean_norms), mean_norms.shape)}"
    )


if __name__ == "__main__":
    main()
