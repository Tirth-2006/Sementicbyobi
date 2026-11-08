"""Calculate teacher-forced next-token predictive entropy for saved answers."""
import csv
import json
from pathlib import Path
from typing import Any

import torch

from uncertainty.models import load_quantized_causal_lm
from uncertainty.token_uncertainty.predictive_entropy import (
    build_prompt,
    score_entropy,
)


INPUT_PATH = Path("data/generated/trivia_qa_qwen3_20261024_162539.json")
SCORES_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_teacher_forced_scores.json"
)
LOW_TEMP_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_low_temp_correctness.json"
)
BOUNDARY_AUDIT_PATH = Path("data/generated/token_boundary_audit.json")
OUTPUT_JSON = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_token_entropy.json"
)
OUTPUT_CSV = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_token_entropy_question.csv"
)
OUTPUT_REPORT = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_token_entropy_report.md"
)
EXPECTED_QUESTIONS = 100
EXPECTED_ANSWERS = 1000


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> None:
    generation_data = load_json(INPUT_PATH)
    scores_data = load_json(SCORES_PATH)
    low_temp_data = load_json(LOW_TEMP_PATH)
    boundary_data = load_json(BOUNDARY_AUDIT_PATH)
    generations = generation_data.get("generations")
    scores = scores_data.get("questions")
    low_temp_questions = low_temp_data.get("questions")
    if not isinstance(generations, dict) or len(generations) != EXPECTED_QUESTIONS:
        raise ValueError("Generation input must contain 100 questions")
    if not isinstance(scores, dict) or set(scores) != set(generations):
        raise ValueError("Teacher-forced score IDs do not match generations")
    if not isinstance(low_temp_questions, list) or len(low_temp_questions) != EXPECTED_QUESTIONS:
        raise ValueError("Low-temperature correctness must contain 100 questions")
    correctness = {
        item["question_id"]: int(item["correct"])
        for item in low_temp_questions
        if isinstance(item, dict)
        and isinstance(item.get("question_id"), str)
        and item.get("correct") in {0, 1}
    }
    if len(correctness) != EXPECTED_QUESTIONS or set(correctness) != set(generations):
        raise ValueError("Low-temperature correctness IDs do not match generations")

    boundary_summary = boundary_data.get("summary")
    if not isinstance(boundary_summary, dict):
        raise ValueError("Boundary audit lacks summary")
    if (
        boundary_summary.get("answers_audited") != EXPECTED_ANSWERS
        or boundary_summary.get("answers_with_boundary_crossing_token") != 347
        or boundary_summary.get("answers_with_first_scored_token_before_boundary") != 347
    ):
        raise ValueError("Boundary audit counts do not match expected values")

    few_shot_prompt = generation_data.get("metadata", {}).get("few_shot_prompt")
    if not isinstance(few_shot_prompt, str):
        raise ValueError("Generation metadata lacks few_shot_prompt")

    tokenizer, model, model_dir = load_quantized_causal_lm()

    answer_outputs: list[dict[str, Any]] = []
    question_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    boundary_cases = {
        (case["question_id"], case["answer_index"])
        for case in boundary_data.get("boundary_crossing_cases", [])
    }
    for question_id, record in generations.items():
        question = record.get("question")
        responses = record.get("high_temp_generations", {}).get("responses")
        if not isinstance(question, str) or not isinstance(responses, list) or len(responses) != 10:
            raise ValueError(f"Invalid responses for {question_id}")
        score_answers = scores[question_id].get("answers")
        if not isinstance(score_answers, list) or len(score_answers) != 10:
            raise ValueError(f"Invalid saved scores for {question_id}")
        prompt = build_prompt(few_shot_prompt, question)
        means: list[float] = []
        for answer_index, response_record in enumerate(responses):
            answer = response_record.get("response")
            saved_answer = score_answers[answer_index]
            if not isinstance(answer, str) or saved_answer.get("index") != answer_index:
                raise ValueError(f"Answer ordering mismatch for {question_id}:{answer_index}")
            expected_positions = saved_answer["raw_text_only"]["answer_token_positions"]
            try:
                result = score_entropy(
                    model,
                    tokenizer,
                    prompt,
                    answer,
                    expected_positions,
                )
                result.update(
                    {
                        "question_id": question_id,
                        "answer_index": answer_index,
                        "response": answer,
                        "boundary_crossing_token": (
                            question_id,
                            answer_index,
                        )
                        in boundary_cases,
                    }
                )
                answer_outputs.append(result)
                means.append(result["mean_token_entropy_nats"])
            except Exception as error:
                failures.append(
                    {
                        "question_id": question_id,
                        "answer_index": answer_index,
                        "reason": type(error).__name__,
                        "message": str(error),
                    }
                )
        if len(means) != 10:
            continue
        question_rows.append(
            {
                "question_id": question_id,
                "mean_token_entropy": float(sum(means) / len(means)),
                "std_token_entropy": float(
                    torch.tensor(means, dtype=torch.float64).std(unbiased=False)
                ),
                "correct": correctness[question_id],
            }
        )

    if len(answer_outputs) != EXPECTED_ANSWERS or failures:
        raise ValueError(f"Expected 1,000 successful answers; failures: {failures}")
    if len(question_rows) != EXPECTED_QUESTIONS:
        raise ValueError("Expected 100 question-level rows")

    output = {
        "metadata": {
            "input_file": str(INPUT_PATH),
            "teacher_forced_scores_file": str(SCORES_PATH),
            "low_temperature_correctness_file": str(LOW_TEMP_PATH),
            "boundary_audit_file": str(BOUNDARY_AUDIT_PATH),
            "model_dir": str(model_dir),
            "model": "Qwen3-1.7B",
            "model_path": str(model_dir.resolve()),
            "model_revision": "unavailable",
            "quantization": "4-bit NF4 with double quantization",
            "compute_dtype": "float16",
            "device_map": "auto",
            "tokenizer_call": "add_special_tokens=True, return_offsets_mapping=True",
            "probability_convention": "raw model softmax; no temperature/top-k/top-p filtering",
            "entropy_formula": "H = -sum(exp(log_softmax(z)) * log_softmax(z))",
            "logarithm_base": "natural logarithm (nats)",
            "answer_boundary": "all tokens whose offsets overlap the answer span; crossing tokens included in full",
            "eos_handling": "EOS excluded; no EOS token appended",
            "question_aggregation": "unweighted mean of 10 answer-level mean token entropies",
            "question_standard_deviation": "population standard deviation across the 10 answer-level means",
            "model_inference_rerun": True,
            "responses_regenerated": False,
        },
        "validation": {
            "questions": EXPECTED_QUESTIONS,
            "answers": EXPECTED_ANSWERS,
            "scoring_failures": 0,
            "all_entropy_finite_nonnegative": True,
            "answer_token_counts_match_offsets": True,
            "causal_shift_verified": True,
            "selected_logits": "model logits_to_keep at positions p-1; no full sequence-by-vocabulary logits retained",
            "independent_entropy_checks": sum(
                len(item["independent_entropy_verification"])
                for item in answer_outputs
            ),
            "boundary_crossing_answers": len(
                {key for key in boundary_cases}
            ),
            "question_rows": len(question_rows),
        },
        "answers": answer_outputs,
    }
    OUTPUT_JSON.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "question_id",
                "mean_token_entropy",
                "std_token_entropy",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(question_rows)

    example_values = answer_outputs[0]["independent_entropy_verification"][:3]
    report = [
        "# Phase 3 Experiment 1: Token Predictive Entropy",
        "",
        "## Method",
        "",
        "Each saved high-temperature answer was rescored with teacher forcing "
        "using the local Qwen3-1.7B model in 4-bit NF4 quantization with "
        "float16 compute. For answer token position `p`, logits at `p-1` "
        "were used because a causal language model's position `p-1` predicts "
        "token `p`. Entropy was calculated from the full raw softmax "
        "distribution in natural-log units.",
        "",
        "Boundary-crossing tokens were included in full whenever their tokenizer "
        "offsets overlapped the answer span. EOS was not appended or included. "
        "Question-level means are unweighted means of the ten answer-level "
        "means; question-level standard deviation is the population standard "
        "deviation across those ten answer-level means, not a pooled token "
        "standard deviation.",
        "",
        "## Validation",
        "",
        f"- Questions: {EXPECTED_QUESTIONS}",
        f"- Answers: {EXPECTED_ANSWERS}",
        "- Scoring failures: 0",
        "- All entropy values finite and nonnegative: yes",
        "- Causal shift verified: yes (`logits[p-1]` predicts token `p`)",
        f"- Independent entropy checks: {output['validation']['independent_entropy_checks']}",
        "- Answer-token counts matched offset-derived positions: yes",
        "- Boundary-crossing answers: 347",
        "",
        "Example independent entropy checks:",
        "",
        "```json",
        json.dumps(example_values, indent=2),
        "```",
        "",
        "## Limitations",
        "",
        "These are raw model-softmax entropies, not the top-k/top-p-filtered "
        "sampling distribution used during generation. Four-bit quantization "
        "and reconstructed-text scoring introduce approximation. Boundary "
        "crossing is retained for consistency with the existing scorer, so "
        "these are not exact character-level answer-only entropies. EOS "
        "uncertainty is not represented because original EOS emission was not "
        "saved.",
    ]
    OUTPUT_REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Questions scored: {EXPECTED_QUESTIONS}")
    print(f"Answers scored: {EXPECTED_ANSWERS}")
    print("Scoring failures: 0")
    print(f"Independent entropy checks: {output['validation']['independent_entropy_checks']}")
    print(f"Example entropy checks: {json.dumps(example_values)}")
    print(f"Saved: {OUTPUT_JSON}, {OUTPUT_CSV}, {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
