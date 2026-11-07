"""Label generated answers against TriviaQA reference answers."""
import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_INPUT = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_clusters.json"
)
DEFAULT_OUTPUT = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_correctness_f1.json"
)


def normalize_answer(answer: str) -> str:
    """Apply the standard SQuAD-style answer normalization."""
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


def label_response(response: str, references: List[str]) -> Dict[str, Any]:
    """Score one response against all references using EM and token F1."""
    normalized_response = normalize_answer(response)
    normalized_references = {
        normalize_answer(reference): reference for reference in references
    }
    matched_reference = normalized_references.get(normalized_response)
    reference_scores = [
        (token_f1(response, reference), reference) for reference in references
    ]
    best_f1, best_f1_reference = max(reference_scores, default=(0.0, None))
    exact_match = matched_reference is not None
    return {
        "response": response,
        "normalized_response": normalized_response,
        "exact_match": exact_match,
        "correct": exact_match,
        "matched_reference": matched_reference,
        "token_f1": best_f1,
        "best_f1_reference": best_f1_reference,
    }


def evaluate_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Add correctness labels and per-question accuracy to clustering results."""
    questions = dataset.get("questions")
    if not isinstance(questions, dict):
        raise ValueError("Input JSON must contain a 'questions' object")

    evaluated_questions: Dict[str, Any] = {}
    for question_id, entry in questions.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Question {question_id!r} is not an object")

        references = entry.get("reference_answers")
        responses = entry.get("responses")
        if not isinstance(references, list) or not all(
            isinstance(reference, str) for reference in references
        ):
            raise ValueError(
                f"Question {question_id!r} has invalid reference answers"
            )
        if not isinstance(responses, list):
            raise ValueError(f"Question {question_id!r} has invalid responses")

        labeled_responses = []
        for index, item in enumerate(responses):
            if not isinstance(item, dict) or not isinstance(
                item.get("response"), str
            ):
                raise ValueError(
                    f"Question {question_id!r} response {index} is invalid"
                )
            labeled = label_response(item["response"], references)
            labeled["index"] = item.get("index", index)
            labeled_responses.append(labeled)

        exact_match_count = sum(
            1 for response in labeled_responses if response["exact_match"]
        )
        mean_token_f1 = sum(
            response["token_f1"] for response in labeled_responses
        ) / len(labeled_responses) if labeled_responses else 0.0
        evaluated_questions[question_id] = {
            "question": entry.get("question"),
            "reference_answers": references,
            "responses": labeled_responses,
            "entropy_nats": entry.get("entropy_nats"),
            "clusters": entry.get("clusters"),
            "num_responses": len(labeled_responses),
            "exact_match_count": exact_match_count,
            "incorrect_count": len(labeled_responses) - exact_match_count,
            "accuracy": exact_match_count / len(labeled_responses)
            if labeled_responses
            else 0.0,
            "mean_token_f1": mean_token_f1,
        }

    return {
        "metadata": {
            "source_file": dataset.get("metadata", {}).get("source_file"),
            "correctness": "normalized_exact_match_and_token_f1",
            "normalization": [
                "lowercase",
                "remove_ascii_punctuation",
                "remove_articles",
                "collapse_whitespace",
            ],
            "num_questions": len(evaluated_questions),
        },
        "questions": evaluated_questions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate generated answers against TriviaQA references"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as file:
        dataset = json.load(file)
    if not isinstance(dataset, dict):
        raise ValueError(f"Input JSON must contain an object: {args.input}")

    results = evaluate_dataset(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    total_responses = sum(
        question["num_responses"] for question in results["questions"].values()
    )
    total_exact_matches = sum(
        question["exact_match_count"]
        for question in results["questions"].values()
    )
    total_f1 = sum(
        response["token_f1"]
        for question in results["questions"].values()
        for response in question["responses"]
    )
    print(f"Evaluated {results['metadata']['num_questions']} questions")
    print(f"Exact-match responses: {total_exact_matches}/{total_responses}")
    print(f"Mean token F1: {total_f1 / total_responses:.4f}")
    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
