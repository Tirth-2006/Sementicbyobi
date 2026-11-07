"""Extract and evaluate post-token hidden-state dispersion."""
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.hidden_states import extract_hidden_state_features as original


OUTPUT_FEATURES = Path(
    "data/generated/"
    "trivia_qa_qwen3_20261024_162539_post_token_hidden_state_features.npz"
)
OUTPUT_QUESTIONS = Path(
    "data/generated/"
    "trivia_qa_qwen3_20261024_162539_post_token_hidden_state_question_features.npz"
)
OUTPUT_METADATA = Path(
    "data/generated/"
    "trivia_qa_qwen3_20261024_162539_post_token_hidden_state_metadata.json"
)
HIDDEN_STATE_INDEX = 28
EXPECTED_QUESTIONS = 100
EXPECTED_ANSWERS = 1000
EXPECTED_ANSWERS_PER_QUESTION = 10
HIDDEN_SIZE = 2048


def extract_post_token_vector(
    model: original.CausalModelProtocol,
    tokenizer: original.TokenizerProtocol,
    prompt: str,
    answer: str,
) -> tuple[np.ndarray, list[int], int]:
    encoded = tokenizer(
        prompt + answer,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"][0]
    offsets = encoded["offset_mapping"][0].tolist()
    positions = original.answer_positions(
        offsets, len(prompt), len(answer), input_ids.tolist()
    )
    device = next(model.parameters()).device
    model_inputs = {
        "input_ids": input_ids.unsqueeze(0).to(device),
        "attention_mask": torch.ones_like(input_ids).unsqueeze(0).to(device),
    }
    with torch.inference_mode():
        output = model(
            **model_inputs,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden_states = output.hidden_states
    if not isinstance(hidden_states, tuple) or len(hidden_states) <= HIDDEN_STATE_INDEX:
        raise ValueError("Model did not return hidden state index 28")
    selected = hidden_states[HIDDEN_STATE_INDEX][0, positions, :].float()
    vector = selected.mean(dim=0).cpu().numpy().astype(np.float32, copy=False)
    del output, hidden_states, model_inputs, selected
    if vector.shape != (HIDDEN_SIZE,):
        raise ValueError(f"Unexpected vector shape: {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError("Post-token vector contains non-finite values")
    return vector, positions, int(input_ids.numel())


def validate_and_aggregate(
    answer_features: np.ndarray,
    question_ids: list[str],
    records: list[dict[str, Any]],
    correctness: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    if answer_features.shape != (EXPECTED_ANSWERS, HIDDEN_SIZE):
        raise ValueError(f"Unexpected answer feature shape: {answer_features.shape}")
    if answer_features.dtype != np.float32:
        raise ValueError(f"Unexpected answer feature dtype: {answer_features.dtype}")
    if not np.isfinite(answer_features).all():
        raise ValueError("Answer features contain non-finite values")
    if len(question_ids) != EXPECTED_QUESTIONS or len(records) != EXPECTED_ANSWERS:
        raise ValueError("Unexpected question or answer count")
    if len(set(question_ids)) != EXPECTED_QUESTIONS:
        raise ValueError("Question IDs are not unique")

    indices_by_question: dict[str, list[int]] = {qid: [] for qid in question_ids}
    for row, record in enumerate(records):
        question_id = record["question_id"]
        answer_index = record["answer_index"]
        if question_id not in indices_by_question:
            raise ValueError(f"Unknown question ID: {question_id}")
        if answer_index not in range(EXPECTED_ANSWERS_PER_QUESTION):
            raise ValueError(f"Invalid answer index: {question_id}:{answer_index}")
        indices_by_question[question_id].append(row)
    for question_id, rows in indices_by_question.items():
        answer_indices = sorted(records[row]["answer_index"] for row in rows)
        if answer_indices != list(range(EXPECTED_ANSWERS_PER_QUESTION)):
            raise ValueError(f"Incomplete answer mapping for {question_id}")

    question_features = np.empty(
        (EXPECTED_QUESTIONS, HIDDEN_SIZE), dtype=np.float32
    )
    for question_index, question_id in enumerate(question_ids):
        rows = indices_by_question[question_id]
        question_features[question_index] = answer_features[rows].mean(
            axis=0, dtype=np.float32
        )
    if not np.isfinite(question_features).all():
        raise ValueError("Question features contain non-finite values")
    if set(question_ids) != set(correctness):
        raise ValueError("Generation and correctness question IDs do not match")
    return question_features, np.asarray(
        [correctness[question_id] for question_id in question_ids],
        dtype=np.int8,
    )


def validate_saved_features(
    answer_ids: np.ndarray,
    answer_indices: np.ndarray,
    answer_features: np.ndarray,
    question_ids: np.ndarray,
    question_features: np.ndarray,
    records: list[dict[str, Any]],
) -> None:
    if answer_features.shape != (EXPECTED_ANSWERS, HIDDEN_SIZE):
        raise ValueError("Saved answer feature shape is invalid")
    if question_features.shape != (EXPECTED_QUESTIONS, HIDDEN_SIZE):
        raise ValueError("Saved question feature shape is invalid")
    if answer_ids.shape != (EXPECTED_ANSWERS,) or answer_indices.shape != (
        EXPECTED_ANSWERS,
    ):
        raise ValueError("Saved answer mapping shapes are invalid")
    if question_ids.shape != (EXPECTED_QUESTIONS,):
        raise ValueError("Saved question ID shape is invalid")
    for row, record in enumerate(records):
        if str(answer_ids[row]) != record["question_id"]:
            raise ValueError(f"Saved question mapping mismatch at row {row}")
        if int(answer_indices[row]) != record["answer_index"]:
            raise ValueError(f"Saved answer mapping mismatch at row {row}")
    if not np.isfinite(answer_features).all() or not np.isfinite(
        question_features
    ).all():
        raise ValueError("Saved features contain non-finite values")
    saved_rows_by_question: dict[str, list[int]] = {
        str(question_id): [] for question_id in question_ids
    }
    for row, question_id in enumerate(answer_ids):
        saved_rows_by_question[str(question_id)].append(row)
    for question_index, question_id in enumerate(question_ids):
        rows = saved_rows_by_question[str(question_id)]
        recomputed = answer_features[rows].mean(axis=0, dtype=np.float32)
        if not np.array_equal(recomputed, question_features[question_index]):
            raise ValueError(
                f"Saved question aggregation mismatch for {question_id}"
            )


def main() -> None:
    generation_data = original.load_json(original.INPUT_PATH)
    question_ids, records, few_shot_prompt = original.load_answer_records(
        generation_data
    )
    correctness = original.load_correctness(original.CORRECTNESS_PATH)
    tokenizer, model = original.load_model()

    answer_features: list[np.ndarray] = []
    answer_positions: list[list[int]] = []
    input_token_counts: list[int] = []
    for count, record in enumerate(records, start=1):
        prompt = original.build_prompt(few_shot_prompt, record["question"])
        vector, positions, input_token_count = extract_post_token_vector(
            model, tokenizer, prompt, record["answer"]
        )
        answer_features.append(vector)
        answer_positions.append(positions)
        input_token_counts.append(input_token_count)
        if count % 100 == 0:
            print(f"Extracted {count}/{len(records)} answers")

    answer_array = np.stack(answer_features).astype(np.float32, copy=False)
    question_array, correct_array = validate_and_aggregate(
        answer_array, question_ids, records, correctness
    )
    answer_ids = np.asarray([record["question_id"] for record in records])
    answer_indices = np.asarray(
        [record["answer_index"] for record in records], dtype=np.int64
    )
    question_id_array = np.asarray(question_ids)
    OUTPUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_FEATURES,
        features=answer_array,
        question_id=answer_ids,
        answer_index=answer_indices,
        answer_token_positions=np.asarray(answer_positions, dtype=object),
        input_token_count=np.asarray(input_token_counts, dtype=np.int32),
    )
    np.savez_compressed(
        OUTPUT_QUESTIONS,
        features=question_array,
        question_id=question_id_array,
        correct=correct_array,
    )

    with np.load(OUTPUT_FEATURES, allow_pickle=True) as saved_answers, np.load(
        OUTPUT_QUESTIONS
    ) as saved_questions:
        validate_saved_features(
            saved_answers["question_id"],
            saved_answers["answer_index"],
            saved_answers["features"],
            saved_questions["question_id"],
            saved_questions["features"],
            records,
        )

    dispersions: list[float] = []
    print("\nQuestion dispersion")
    print("question_id,dispersion")
    for question_index, question_id in enumerate(question_ids):
        rows = [
            row
            for row, record in enumerate(records)
            if record["question_id"] == question_id
        ]
        distances = np.linalg.norm(
            answer_array[rows] - question_array[question_index],
            axis=1,
        ).astype(np.float32)
        dispersion = float(np.mean(distances, dtype=np.float32))
        if not np.isfinite(dispersion) or dispersion < 0:
            raise ValueError(f"Invalid dispersion for {question_id}")
        dispersions.append(dispersion)
        print(f"{question_id},{dispersion:.9f}")

    dispersion_array = np.asarray(dispersions, dtype=np.float32)
    print("\nSummary statistics")
    print(f"questions_processed: {len(dispersion_array)}")
    print(f"min: {float(np.min(dispersion_array)):.9f}")
    print(f"max: {float(np.max(dispersion_array)):.9f}")
    print(f"mean: {float(np.mean(dispersion_array, dtype=np.float32)):.9f}")
    print(f"median: {float(np.median(dispersion_array)):.9f}")
    print(f"std: {float(np.std(dispersion_array, dtype=np.float32)):.9f}")
    print("validation_failures: 0")
    print(f"Saved per-answer features to {OUTPUT_FEATURES}")
    print(f"Saved question features to {OUTPUT_QUESTIONS}")
    print(f"Saved metadata to {OUTPUT_METADATA}")

    metadata = {
        "source_script": "extract_hidden_state_features.py",
        "input_file": str(original.INPUT_PATH),
        "model": "Qwen3-1.7B",
        "quantization": "4-bit NF4 with double quantization",
        "compute_dtype": "float16",
        "device_map": "auto",
        "hidden_state_index": HIDDEN_STATE_INDEX,
        "hidden_size": HIDDEN_SIZE,
        "position_alignment": "hidden state at selected answer token position p",
        "pooling": "mean over selected answer-token hidden states",
        "num_questions": EXPECTED_QUESTIONS,
        "num_answers": EXPECTED_ANSWERS,
        "answer_feature_shape": list(answer_array.shape),
        "question_feature_shape": list(question_array.shape),
        "answer_feature_dtype": str(answer_array.dtype),
        "question_feature_dtype": str(question_array.dtype),
        "validation_failures": 0,
        "dispersion": {
            "min": float(np.min(dispersion_array)),
            "max": float(np.max(dispersion_array)),
            "mean": float(np.mean(dispersion_array, dtype=np.float32)),
            "median": float(np.median(dispersion_array)),
            "std": float(np.std(dispersion_array, dtype=np.float32)),
        },
    }
    OUTPUT_METADATA.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
