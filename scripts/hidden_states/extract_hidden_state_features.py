"""Extract causal hidden-state features for saved high-temperature answers."""
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from uncertainty.hidden_states.extraction import (
    POOLING_STRATEGIES,
    CausalModelProtocol,
    TokenizerProtocol,
    answer_positions,  # noqa: F401  (re-exported for sibling scripts)
    build_prompt,
    extract_record,
    pool_hidden_states,  # noqa: F401  (re-exported for sibling scripts)
)
from uncertainty.config import resolve_model_dir
from uncertainty.models import load_quantized_causal_lm


INPUT_PATH = Path("data/generated/trivia_qa_qwen3_20261024_162539.json")
CORRECTNESS_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_low_temp_correctness.json"
)
OUTPUT_FEATURES = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_hidden_state_features.npz"
)
OUTPUT_QUESTIONS = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_hidden_state_question_features.npz"
)
OUTPUT_METADATA = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_hidden_state_metadata.json"
)
OUTPUT_REPORT = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_hidden_state_report.md"
)
EXPECTED_QUESTIONS = 100
EXPECTED_ANSWERS = 1000


def load_model() -> tuple[TokenizerProtocol, CausalModelProtocol]:
    """Load the quantized model from ``UNCERTAINTY_MODEL_DIR``."""
    tokenizer, model, _ = load_quantized_causal_lm()
    return tokenizer, model


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def load_answer_records(
    generation_data: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]], str]:
    generations = generation_data.get("generations")
    metadata = generation_data.get("metadata")
    if not isinstance(generations, dict) or len(generations) != EXPECTED_QUESTIONS:
        raise ValueError("Generation file must contain 100 questions")
    if not isinstance(metadata, dict) or not isinstance(
        metadata.get("few_shot_prompt"), str
    ):
        raise ValueError("Generation metadata must contain few_shot_prompt")

    question_ids = list(generations)
    records: list[dict[str, Any]] = []
    for question_id in question_ids:
        entry = generations[question_id]
        if not isinstance(entry, dict) or not isinstance(entry.get("question"), str):
            raise ValueError(f"Invalid generation entry for {question_id}")
        responses = entry.get("high_temp_generations", {}).get("responses")
        if not isinstance(responses, list) or len(responses) != 10:
            raise ValueError(f"{question_id} must contain exactly 10 responses")
        for answer_index, response_record in enumerate(responses):
            if (
                not isinstance(response_record, dict)
                or not isinstance(response_record.get("response"), str)
            ):
                raise ValueError(f"Invalid answer {question_id}:{answer_index}")
            records.append(
                {
                    "question_id": question_id,
                    "answer_index": answer_index,
                    "question": entry["question"],
                    "answer": response_record["response"],
                }
            )
    if len(records) != EXPECTED_ANSWERS:
        raise ValueError(f"Expected {EXPECTED_ANSWERS} answers, found {len(records)}")
    return question_ids, records, metadata["few_shot_prompt"]


def load_correctness(path: Path) -> dict[str, int]:
    data = load_json(path)
    questions = data.get("questions")
    if not isinstance(questions, list) or len(questions) != EXPECTED_QUESTIONS:
        raise ValueError("Correctness file must contain 100 questions")
    result: dict[str, int] = {}
    for item in questions:
        if not isinstance(item, dict):
            raise ValueError("Invalid correctness record")
        question_id = item.get("question_id")
        correct = item.get("correct")
        if (
            not isinstance(question_id, str)
            or question_id in result
            or type(correct) is not int
            or correct not in (0, 1)
        ):
            raise ValueError("Invalid correctness record")
        result[question_id] = correct
    return result


def write_report(metadata: dict[str, Any], analysis: dict[str, Any]) -> None:
    report = [
        "# Phase 3 Experiment 3: Hidden-State Feature Extraction",
        "",
        "## Objective",
        "",
        "This experiment extracts internal Qwen3-1.7B hidden-state representations "
        "from the 1,000 saved temperature-1.0 answers. The representations are "
        "features for a future question-level cross-validated probe; they are not "
        "treated as uncertainty scores and no predictive probe is trained here.",
        "",
        "## Methodology",
        "",
        "- Prompts were reconstructed with the existing raw few-shot prompt plus "
        "`Question: ...\\nAnswer:` format.",
        "- Saved answers were teacher-forced as `prompt + answer`; responses were "
        "not regenerated.",
        "- Answer tokens were selected using the existing character-offset overlap "
        "rule, including boundary-crossing tokens.",
        "- For answer token position `p`, the representation was taken from the "
        "causal predictor position `p-1`.",
        "- Four pooling strategies were saved for every returned hidden-state "
        "index: "
        "first, last, mean, and max over causal answer-token predictor states.",
        "- Question-level features are the unweighted mean of the ten answer-level "
        "feature vectors for each question. This preserves the existing question "
        "unit and avoids treating the ten responses as independent correctness "
        "examples.",
        "",
        "## Implementation",
        "",
        f"- Model: {metadata['model']}",
        f"- Quantization: {metadata['quantization']}",
        f"- Compute dtype: {metadata['compute_dtype']}",
        f"- Device map: {metadata['device_map']}",
        f"- Hidden-state indices returned: {metadata['num_hidden_state_indices']}",
        "- Hidden-state index convention: index 0 is the token embedding output; "
        "indices 1–27 are outputs of decoder blocks 1–27; index 28 is the "
        "final RMSNorm-normalized output after decoder block 28. There are "
        "28 decoder blocks in total.",
        f"- Hidden dimension: {metadata['hidden_size']}",
        f"- Inference batch size: {metadata['batch_size']}",
        "- Hidden states were moved to CPU after each answer and saved as float16 "
        "per-answer features to limit storage and GPU memory.",
        "",
        "## Descriptive observations",
        "",
        f"- Per-answer feature tensor shape: `{metadata['answer_feature_shape']}`.",
        f"- Question aggregate tensor shape: `{metadata['question_feature_shape']}`.",
        f"- Maximum mean feature norm: {analysis['maximum_mean_feature_norm']:.6f}.",
        f"- Minimum finite feature norm: {analysis['minimum_mean_feature_norm']:.6f}.",
        "",
        "These summaries characterize representation scale only. They do not "
        "measure correctness prediction and should not be interpreted as probe "
        "performance.",
        "",
        "## Limitations and next experiment",
        "",
        "Boundary-crossing tokens are retained for consistency with previous "
        "teacher-forced extraction and therefore are not exact character-level "
        "answer-only states. Four-bit quantization and saved-answer teacher forcing "
        "remain part of the measurement. The next experiment should evaluate "
        "layer-wise probes using question-level cross-validation, with all ten "
        "answer representations grouped within each question split.",
    ]
    OUTPUT_REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size != 1:
        raise ValueError(
            "Only batch size 1 is supported because variable answer token spans "
            "require per-answer offset handling and the RTX 2050 memory budget."
        )

    generation_data = load_json(INPUT_PATH)
    question_ids, records, few_shot_prompt = load_answer_records(generation_data)
    correctness = load_correctness(CORRECTNESS_PATH)
    if set(question_ids) != set(correctness):
        raise ValueError("Generation and correctness question IDs do not match")
    tokenizer, model = load_model()
    answer_features: list[np.ndarray] = []
    answer_positions_output: list[list[int]] = []
    input_token_counts: list[int] = []
    for index, record in enumerate(records, start=1):
        prompt = build_prompt(few_shot_prompt, record["question"])
        features, positions, input_token_count = extract_record(
            model, tokenizer, prompt, record["answer"]
        )
        answer_features.append(features)
        answer_positions_output.append(positions)
        input_token_counts.append(input_token_count)
        if index % 100 == 0:
            print(f"Extracted {index}/{len(records)} answers")

    feature_array = np.stack(answer_features).astype(np.float16, copy=False)
    num_layers = feature_array.shape[1]
    hidden_size = feature_array.shape[3]
    question_features = np.empty(
        (len(question_ids), num_layers, len(POOLING_STRATEGIES), hidden_size),
        dtype=np.float32,
    )
    question_indices: dict[str, list[int]] = {question_id: [] for question_id in question_ids}
    for index, record in enumerate(records):
        question_indices[record["question_id"]].append(index)
    for question_index, question_id in enumerate(question_ids):
        indices = question_indices[question_id]
        if len(indices) != 10:
            raise ValueError(f"{question_id} does not have 10 extracted answers")
        question_features[question_index] = feature_array[indices].astype(
            np.float32
        ).mean(axis=0)

    metadata = {
        "input_file": str(INPUT_PATH),
        "correctness_file": str(CORRECTNESS_PATH),
        "model_path": str(resolve_model_dir().resolve()),
        "model": "Qwen3-1.7B",
        "model_revision": "unavailable",
        "quantization": "4-bit NF4 with double quantization",
        "compute_dtype": "float16",
        "device_map": "auto",
        "tokenizer_call": "add_special_tokens=True, return_offsets_mapping=True",
        "prompt_format": "saved few-shot prompt plus Question/Answer raw text",
        "answer_boundary": (
            "all non-empty tokenizer offsets overlapping the answer span; "
            "boundary-crossing tokens included"
        ),
        "causal_alignment": (
            "hidden state at predictor position p-1 represents the context "
            "used to predict answer token p"
        ),
        "pooling_alignment": (
            "first and last pool the hidden states at the predictor positions "
            "for the first and last selected answer tokens; mean and max pool "
            "all selected predictor positions"
        ),
        "pooling_strategies": list(POOLING_STRATEGIES),
        "feature_axes": [
            "answer",
            "hidden_state_index",
            "pooling_strategy",
            "hidden_dimension",
        ],
        "layer_index_convention": (
            "index 0 is the token embedding output; indices 1 through 27 are "
            "outputs of decoder blocks 1 through 27; index 28 is the final "
            "RMSNorm-normalized output after decoder block 28; there are 28 "
            "decoder blocks in total"
        ),
        "question_aggregation": (
            "unweighted mean of ten answer-level feature vectors per question"
        ),
        "num_questions": len(question_ids),
        "num_answers": len(records),
        "num_hidden_state_indices": num_layers,
        "hidden_size": hidden_size,
        "batch_size": args.batch_size,
        "answer_feature_dtype": str(feature_array.dtype),
        "question_feature_dtype": str(question_features.dtype),
        "answer_feature_shape": list(feature_array.shape),
        "question_feature_shape": list(question_features.shape),
        "report_file": str(OUTPUT_REPORT),
        "responses_regenerated": False,
        "correctness_target": (
            "separate low-temperature reference-based exact-match label; "
            "not used to create answer-level labels"
        ),
    }
    mean_norms = np.linalg.norm(
        question_features.astype(np.float64), axis=-1
    )
    analysis = {
        "minimum_mean_feature_norm": float(mean_norms.min()),
        "maximum_mean_feature_norm": float(mean_norms.max()),
        "mean_feature_norm_by_layer_pooling": mean_norms.mean(axis=0).tolist(),
    }
    metadata["analysis"] = analysis
    OUTPUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_FEATURES,
        features=feature_array,
        question_id=np.asarray([record["question_id"] for record in records]),
        answer_index=np.asarray([record["answer_index"] for record in records]),
        answer_token_positions=np.asarray(answer_positions_output, dtype=object),
        input_token_count=np.asarray(input_token_counts, dtype=np.int32),
    )
    np.savez_compressed(
        OUTPUT_QUESTIONS,
        features=question_features,
        question_id=np.asarray(question_ids),
        correct=np.asarray([correctness[qid] for qid in question_ids], dtype=np.int8),
    )
    OUTPUT_METADATA.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_report(metadata, analysis)
    print(f"Saved per-answer features to {OUTPUT_FEATURES}")
    print(f"Saved question features to {OUTPUT_QUESTIONS}")
    print(f"Saved metadata to {OUTPUT_METADATA}")
    print(f"Saved report to {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
