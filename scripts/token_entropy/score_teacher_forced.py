"""Teacher-force saved Qwen answers to estimate sequence likelihoods."""
import json
from pathlib import Path
from typing import Any

from uncertainty.models import load_quantized_causal_lm
from uncertainty.token_uncertainty.teacher_forcing import (
    build_prompt,
    cluster_probabilities,
    entropy,
    log_normalize,
    score_text,
)


INPUT_PATH = Path("data/generated/trivia_qa_qwen3_20261024_162539.json")
CLUSTERS_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_clusters.json"
)
OUTPUT_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_teacher_forced_scores.json"
)
EXPECTED_QUESTIONS = 100
EXPECTED_ANSWERS = 1000


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


def main() -> None:
    generations, cluster_data = load_inputs()
    few_shot_prompt = generations["metadata"].get("few_shot_prompt")
    if not isinstance(few_shot_prompt, str):
        raise ValueError("Saved metadata does not contain few_shot_prompt")

    tokenizer, model, model_dir = load_quantized_causal_lm()

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
            "model_dir": str(model_dir),
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
