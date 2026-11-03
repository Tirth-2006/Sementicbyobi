"""Teacher-force saved Qwen answers to estimate sequence likelihoods."""
import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


INPUT_PATH = Path("data/generated/trivia_qa_qwen3_20261024_162539.json")
CLUSTERS_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_clusters.json"
)
OUTPUT_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_teacher_forced_scores.json"
)
MODEL_DIR = Path(".")
EXPECTED_QUESTIONS = 100
EXPECTED_ANSWERS = 1000
EOS_TOKEN_ID = 151645


def load_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load saved generations and existing semantic clusters."""
    with INPUT_PATH.open("r", encoding="utf-8") as input_file:
        generations = json.load(input_file)
    with CLUSTERS_PATH.open("r", encoding="utf-8") as input_file:
        clusters = json.load(input_file)
    if not isinstance(generations.get("generations"), dict):
        raise ValueError("Generation file must contain a generations object")
    if not isinstance(clusters.get("questions"), dict):
        raise ValueError("Cluster file must contain a questions object")
    if set(generations["generations"]) != set(clusters["questions"]):
        raise ValueError("Generation and cluster question IDs do not match")
    return generations, clusters


def build_prompt(few_shot_prompt: str, question: str) -> str:
    """Reconstruct the original raw-text query prompt."""
    return f"{few_shot_prompt}Question: {question}\nAnswer:"


def validate_offsets(
    offsets: list[list[int]],
    prompt_length: int,
    answer_length: int,
    token_ids: list[int],
) -> list[int]:
    """Find token positions overlapping the answer character span."""
    answer_start = prompt_length
    answer_end = prompt_length + answer_length
    positions = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and start < answer_end and end > answer_start
    ]
    if not positions:
        raise ValueError("No answer tokens were found from offset mapping")
    if positions[0] == 0:
        raise ValueError("First answer token has no preceding causal logit")
    if any(token_ids[index] is None for index in positions):
        raise ValueError("Answer token ID is missing")
    return positions


def score_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    answer: str,
    include_eos: bool,
) -> dict[str, Any]:
    """Score answer tokens under the raw model distribution."""
    full_text = prompt + answer
    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"][0]
    offsets = encoded["offset_mapping"][0].tolist()
    answer_positions = validate_offsets(
        offsets,
        len(prompt),
        len(answer),
        input_ids.tolist(),
    )

    if include_eos:
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("Tokenizer has no EOS token")
        input_ids = torch.cat(
            [input_ids, torch.tensor([eos_id], dtype=input_ids.dtype)]
        )
        attention_mask = torch.ones_like(input_ids)
        score_positions = answer_positions + [len(input_ids) - 1]
    else:
        attention_mask = torch.ones_like(input_ids)
        score_positions = answer_positions

    device = next(model.parameters()).device
    model_inputs = {
        "input_ids": input_ids.unsqueeze(0).to(device),
        "attention_mask": attention_mask.unsqueeze(0).to(device),
    }
    with torch.inference_mode():
        logits = model(**model_inputs).logits[0].float()

    labels = model_inputs["input_ids"][0]
    log_probs = torch.log_softmax(logits[:-1], dim=-1)
    selected_log_probs = torch.stack(
        [
            log_probs[position - 1, labels[position]]
            for position in score_positions
        ]
    )
    if not torch.isfinite(selected_log_probs).all():
        raise ValueError("Non-finite answer token log probability")

    values = selected_log_probs.detach().cpu().tolist()
    return {
        "token_count": len(values),
        "token_log_probabilities": values,
        "raw_sequence_log_likelihood": float(sum(values)),
        "length_normalized_log_likelihood": float(sum(values) / len(values)),
        "answer_token_positions": score_positions,
        "input_token_count": int(input_ids.numel()),
        "eos_included": include_eos,
    }


def log_normalize(scores: list[float]) -> list[float]:
    """Normalize log scores using a stable log-sum-exp operation."""
    tensor = torch.tensor(scores, dtype=torch.float64)
    normalized = torch.log_softmax(tensor, dim=0).exp()
    values = normalized.tolist()
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite normalized answer weight")
    if not math.isclose(sum(values), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Normalized answer weights do not sum to one")
    return values


def entropy(probabilities: list[float]) -> float:
    """Calculate Shannon entropy from normalized cluster probabilities."""
    if not probabilities or not math.isclose(
        sum(probabilities), 1.0, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("Cluster probabilities must sum to one")
    value = -sum(probability * math.log(probability) for probability in probabilities)
    if not math.isfinite(value) or value < -1e-12:
        raise ValueError("Entropy must be finite and non-negative")
    return max(0.0, value)


def cluster_probabilities(
    weights: list[float],
    cluster_record: dict[str, Any],
) -> list[float]:
    """Aggregate normalized answer weights into saved semantic clusters."""
    probabilities = []
    assigned: list[int] = []
    for cluster in cluster_record["clusters"]:
        indices = cluster["indices"]
        probability = sum(weights[index] for index in indices)
        probabilities.append(probability)
        assigned.extend(indices)
    if sorted(assigned) != list(range(len(weights))):
        raise ValueError("Saved clusters do not cover each answer exactly once")
    total = sum(probabilities)
    if not math.isclose(total, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Cluster probabilities do not sum to one")
    return probabilities


def main() -> None:
    generations, cluster_data = load_inputs()
    few_shot_prompt = generations["metadata"].get("few_shot_prompt")
    if not isinstance(few_shot_prompt, str):
        raise ValueError("Saved metadata does not contain few_shot_prompt")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    model.eval()

    questions_output: dict[str, Any] = {}
    total_answers = 0
    failures: list[dict[str, Any]] = []
    for question_id, record in generations["generations"].items():
        question_record = cluster_data["questions"][question_id]
        question = record.get("question")
        responses = record.get("high_temp_generations", {}).get("responses")
        if not isinstance(question, str) or not isinstance(responses, list):
            raise ValueError(f"Invalid generation record for {question_id!r}")
        if len(responses) != 10:
            raise ValueError(f"{question_id!r} does not contain 10 answers")

        prompt = build_prompt(few_shot_prompt, question)
        answer_results: list[dict[str, Any]] = []
        for answer_index, response_record in enumerate(responses):
            answer = response_record.get("response")
            if not isinstance(answer, str):
                raise ValueError(f"Invalid answer {question_id!r}:{answer_index}")
            total_answers += 1
            result: dict[str, Any] = {
                "index": answer_index,
                "response": answer,
                "raw_text_preserved": True,
            }
            try:
                raw = score_text(model, tokenizer, prompt, answer, False)
                eos = score_text(model, tokenizer, prompt, answer, True)
                result["raw_text_only"] = raw
                result["eos_inclusive_sensitivity"] = eos
            except Exception as error:
                result["scoring_failure"] = {
                    "reason": type(error).__name__,
                    "message": str(error),
                }
                failures.append(
                    {
                        "question_id": question_id,
                        "answer_index": answer_index,
                        "reason": type(error).__name__,
                        "message": str(error),
                    }
                )
            answer_results.append(result)

        successful = [
            result for result in answer_results if "raw_text_only" in result
        ]
        if len(successful) != len(answer_results):
            raw_status = "failed"
        else:
            raw_scores = [
                result["raw_text_only"]["raw_sequence_log_likelihood"]
                for result in successful
            ]
            eos_scores = [
                result["eos_inclusive_sensitivity"]["raw_sequence_log_likelihood"]
                for result in successful
            ]
            normalized_raw = log_normalize(raw_scores)
            normalized_length = log_normalize(
                [
                    result["raw_text_only"]["length_normalized_log_likelihood"]
                    for result in successful
                ]
            )
            normalized_eos = log_normalize(eos_scores)
            for result, raw_weight, length_weight in zip(
                successful, normalized_raw, normalized_length
            ):
                result["raw_text_only"]["raw_sequence_weight"] = raw_weight
                result["raw_text_only"]["length_normalized_weight"] = length_weight
            for result, eos_weight in zip(successful, normalized_eos):
                result["eos_inclusive_sensitivity"]["raw_sequence_weight"] = (
                    eos_weight
                )
            raw_cluster_weights = cluster_probabilities(
                normalized_raw, question_record
            )
            length_cluster_weights = cluster_probabilities(
                normalized_length, question_record
            )
            eos_cluster_weights = cluster_probabilities(
                normalized_eos, question_record
            )
            raw_status = {
                "raw_sequence_weight_sum": sum(normalized_raw),
                "length_normalized_weight_sum": sum(normalized_length),
                "eos_inclusive_weight_sum": sum(normalized_eos),
                "raw_sequence_cluster_probability_sum": sum(raw_cluster_weights),
                "length_normalized_cluster_probability_sum": sum(
                    length_cluster_weights
                ),
                "eos_inclusive_cluster_probability_sum": sum(eos_cluster_weights),
                "raw_sequence_entropy_nats": entropy(raw_cluster_weights),
                "length_normalized_entropy_nats": entropy(length_cluster_weights),
                "eos_inclusive_entropy_nats": entropy(eos_cluster_weights),
                "empirical_entropy_nats": question_record["entropy_nats"],
                "raw_sequence_cluster_probabilities": raw_cluster_weights,
                "length_normalized_cluster_probabilities": length_cluster_weights,
                "eos_inclusive_cluster_probabilities": eos_cluster_weights,
            }

        questions_output[question_id] = {
            "question": question,
            "reconstructed_prompt": prompt,
            "answers": answer_results,
            "validation": raw_status,
        }

    if len(questions_output) != EXPECTED_QUESTIONS or total_answers != EXPECTED_ANSWERS:
        raise ValueError(
            f"Expected {EXPECTED_QUESTIONS} questions and {EXPECTED_ANSWERS} answers; "
            f"found {len(questions_output)} and {total_answers}"
        )

    output = {
        "metadata": {
            "input_file": str(INPUT_PATH),
            "clusters_file": str(CLUSTERS_PATH),
            "model_dir": str(MODEL_DIR),
            "quantization": "4-bit NF4 with float16 compute",
            "prompt_format": "saved raw few-shot prompt plus Question/Answer text",
            "probability_convention": "raw model softmax, not top-k/top-p filtered",
            "primary_scoring": "saved answer text without EOS",
            "eos_sensitivity": (
                "EOS included as sensitivity analysis; original EOS emission is unknown"
            ),
            "length_normalization": (
                "mean answer-token log probability; not claimed as exact paper reproduction"
            ),
            "num_questions": len(questions_output),
            "num_answers": total_answers,
            "failure_count": len(failures),
        },
        "validation": {
            "all_answers_scored": not failures,
            "all_scores_finite": not failures,
            "normalized_weights_sum_to_one": not failures,
            "cluster_probabilities_sum_to_one": not failures,
            "entropies_finite_nonnegative": not failures,
            "embedded_newline_answers": [
                {
                    "question_id": question_id,
                    "answer_index": answer["index"],
                    "response": answer["response"],
                }
                for question_id, question in questions_output.items()
                for answer in question["answers"]
                if "\n" in answer["response"]
            ],
        },
        "failures": failures,
        "questions": questions_output,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(output, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")

    print(f"Questions scored: {len(questions_output)}")
    print(f"Answers scored: {total_answers}")
    print(f"Scoring failures: {len(failures)}")
    print(
        "Embedded-newline answers: "
        f"{len(output['validation']['embedded_newline_answers'])}"
    )
    print(f"Saved scores to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
