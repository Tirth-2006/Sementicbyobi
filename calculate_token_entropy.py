"""Calculate teacher-forced next-token predictive entropy for saved answers."""
import csv
import json
import math
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


INPUT_PATH = Path("data/generated/trivia_qa_qwen3_20261024_162539.json")
SCORES_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_teacher_forced_scores.json"
)
LOW_TEMP_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_low_temp_correctness.json"
)
BOUNDARY_AUDIT_PATH = Path("data/generated/token_boundary_audit.json")
MODEL_DIR = Path(".")
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


class TokenizerProtocol(Protocol):
    eos_token_id: int | None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def decode(self, *args: Any, **kwargs: Any) -> str:
        ...


class CausalModelProtocol(Protocol):
    def eval(self) -> Any:
        ...

    def parameters(self) -> Any:
        ...

    def __call__(self, **kwargs: Any) -> Any:
        ...


def build_prompt(few_shot_prompt: str, question: str) -> str:
    """Reconstruct the original raw-text prompt."""
    return f"{few_shot_prompt}Question: {question}\nAnswer:"


def validate_offsets(
    offsets: list[list[int]],
    prompt_length: int,
    answer_length: int,
    token_ids: list[int],
) -> list[int]:
    """Find non-empty tokens overlapping the answer character span."""
    answer_end = prompt_length + answer_length
    positions = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and start < answer_end and end > prompt_length
    ]
    if not positions:
        raise ValueError("No answer tokens found from offset mapping")
    if positions[0] == 0:
        raise ValueError("First answer token has no preceding causal logit")
    if any(token_ids[index] is None for index in positions):
        raise ValueError("Answer token ID is missing")
    return positions


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Calculate raw-softmax Shannon entropy for one vocabulary logit row."""
    row = logits.float()
    log_probs = torch.log_softmax(row, dim=-1)
    probabilities = log_probs.exp()
    return -(probabilities * log_probs).sum()


def entropy_from_logits_logsumexp(logits: torch.Tensor) -> torch.Tensor:
    """Calculate entropy using log-sum-exp and the logit expectation."""
    row = logits.float()
    log_partition = torch.logsumexp(row, dim=-1)
    probabilities = torch.exp(row - log_partition)
    return log_partition - (probabilities * row).sum()


def score_entropy(
    model: CausalModelProtocol,
    tokenizer: TokenizerProtocol,
    prompt: str,
    answer: str,
    expected_positions: list[int] | None = None,
) -> dict[str, Any]:
    """Calculate entropy for each answer token using teacher forcing."""
    full_text = prompt + answer
    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"][0]
    offsets = encoded["offset_mapping"][0].tolist()
    positions = validate_offsets(
        offsets,
        len(prompt),
        len(answer),
        input_ids.tolist(),
    )
    if expected_positions is not None and positions != expected_positions:
        raise ValueError("Token positions differ from saved teacher-forced scores")

    device = next(model.parameters()).device
    model_inputs = {
        "input_ids": input_ids.unsqueeze(0).to(device),
        "attention_mask": torch.ones_like(input_ids).unsqueeze(0).to(device),
    }
    if any(position <= 0 or position >= input_ids.numel() for position in positions):
        raise ValueError("Answer token position cannot be causally aligned")

    predictor_positions = [position - 1 for position in positions]
    keep_indices = torch.tensor(predictor_positions, dtype=torch.long, device=device)
    model_inputs["logits_to_keep"] = keep_indices
    with torch.inference_mode():
        selected_logits = model(**model_inputs).logits[0].float()
    if selected_logits.shape[0] != len(positions):
        raise ValueError("Selected model logits do not match answer positions")

    entropy_values: list[float] = []
    verification: list[dict[str, Any]] = []
    for sample_index, position in enumerate(positions):
        prediction_row = selected_logits[sample_index]
        entropy = entropy_from_logits(prediction_row)
        if not torch.isfinite(entropy) or entropy < 0:
            raise ValueError("Token entropy is not finite and nonnegative")
        value = float(entropy.detach().cpu())
        entropy_values.append(value)
        if sample_index < 3:
            independent = entropy_from_logits_logsumexp(prediction_row)
            difference = abs(value - float(independent.detach().cpu()))
            if difference > 1e-5:
                raise ValueError("Independent entropy verification failed")
            verification.append(
                {
                    "answer_token_position": position,
                    "predictor_logit_position": position - 1,
                    "target_token_id": int(input_ids[position].cpu()),
                    "entropy_nats": value,
                    "independent_entropy_nats": float(independent.cpu()),
                    "absolute_difference": difference,
                }
            )

    if len(entropy_values) != len(positions):
        raise ValueError("Entropy count does not match answer-token count")
    return {
        "answer_token_count": len(entropy_values),
        "answer_token_positions": positions,
        "token_entropy_nats": entropy_values,
        "mean_token_entropy_nats": float(sum(entropy_values) / len(entropy_values)),
        "std_token_entropy_nats": float(
            torch.tensor(entropy_values, dtype=torch.float64).std(unbiased=False)
        ),
        "maximum_token_entropy_nats": max(entropy_values),
        "causal_shift_verified": True,
        "independent_entropy_verification": verification,
        "eos_included": False,
    }


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

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = cast(
        TokenizerProtocol,
        AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True),
    )
    model = cast(
        CausalModelProtocol,
        AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
            local_files_only=True,
        ),
    )
    model.eval()

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
            "model_dir": str(MODEL_DIR),
            "model": "Qwen3-1.7B",
            "model_path": str(MODEL_DIR.resolve()),
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
