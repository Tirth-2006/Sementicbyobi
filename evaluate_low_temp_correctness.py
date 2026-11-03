"""Evaluate low-temperature TriviaQA answers against reference answers."""
import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_INPUT = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539.json"
)
DEFAULT_OUTPUT = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_low_temp_correctness.json"
)


def normalize_answer(answer: str) -> str:
    """Apply the same SQuAD-style normalization as evaluate_correctness.py."""
    answer = answer.lower()
    answer = "".join(
        character for character in answer if character not in string.punctuation
    )
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    return " ".join(answer.split())


def token_f1(prediction: str, reference: str) -> float:
    """Calculate token-overlap F1 between one prediction and reference."""
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)

    overlap = sum(
        (Counter(prediction_tokens) & Counter(reference_tokens)).values()
    )
    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def score_low_temp_response(
    response: str,
    references: List[str],
) -> Dict[str, Any]:
    """Score one low-temperature answer using the existing evaluator logic."""
    normalized_response = normalize_answer(response)
    normalized_references = {
        normalize_answer(reference): reference for reference in references
    }
    matched_reference = normalized_references.get(normalized_response)
    reference_scores = [
        (token_f1(response, reference), reference) for reference in references
    ]
    best_f1, _ = max(reference_scores, default=(0.0, None))
    exact_match = matched_reference is not None
    return {
        "exact_match": exact_match,
        "correct": int(exact_match),
        "matched_reference": matched_reference,
        "token_f1": best_f1,
    }


def evaluate_dataset(
    dataset: Dict[str, Any],
    source_file: Path,
) -> Dict[str, Any]:
    """Create one correctness record for each question's low-temp answer."""
    generations = dataset.get("generations")
    if not isinstance(generations, dict):
        raise ValueError("Input JSON must contain a 'generations' object")

    questions = []
    for question_id, entry in generations.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Question {question_id!r} is not an object")

        question = entry.get("question")
        if not isinstance(question, str):
            raise ValueError(f"Question {question_id!r} has invalid question text")

        references = entry.get("reference_answers")
        if not isinstance(references, list) or not all(
            isinstance(reference, str) for reference in references
        ):
            raise ValueError(
                f"Question {question_id!r} has invalid reference answers"
            )

        low_temp_generation = entry.get("low_temp_generation")
        if not isinstance(low_temp_generation, dict):
            raise ValueError(
                f"Question {question_id!r} has no valid low-temperature generation"
            )
        response = low_temp_generation.get("response")
        if not isinstance(response, str):
            raise ValueError(
                f"Question {question_id!r} has an invalid low-temperature response"
            )

        questions.append(
            {
                "question_id": question_id,
                "question": question,
                "low_temp_response": response,
                **score_low_temp_response(response, references),
            }
        )

    return {
        "metadata": {
            "source_file": str(source_file),
            "correctness": "normalized_exact_match_and_token_f1",
            "generation": "low_temperature",
            "temperature": 0.1,
            "normalization": [
                "lowercase",
                "remove_ascii_punctuation",
                "remove_articles",
                "collapse_whitespace",
            ],
            "num_questions": len(questions),
        },
        "questions": questions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate low-temperature answers against TriviaQA references"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as input_file:
        dataset = json.load(input_file)
    if not isinstance(dataset, dict):
        raise ValueError(f"Input JSON must contain an object: {args.input}")

    results = evaluate_dataset(dataset, args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2, ensure_ascii=False)

    correct_count = sum(question["correct"] for question in results["questions"])
    mean_f1 = (
        sum(question["token_f1"] for question in results["questions"])
        / len(results["questions"])
        if results["questions"]
        else 0.0
    )
    print(f"Evaluated {results['metadata']['num_questions']} questions")
    print(f"Low-temperature exact matches: {correct_count}/{len(results['questions'])}")
    print(f"Mean low-temperature token F1: {mean_f1:.4f}")
    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
