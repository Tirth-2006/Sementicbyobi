"""Reference-based correctness labelling for TriviaQA-style answers.

Two *different* targets are scored here and must not be confused:

* **Sampled (high-temperature) answers**: the ten T=1.0 responses used as
  uncertainty *features*. See :func:`evaluate_sampled_dataset`.
* **Low-temperature target answer**: the single separate T=0.1 response whose
  correctness is the *prediction target*. See :func:`evaluate_low_temp_dataset`.

Both use the same normalized exact-match rule (lowercase, strip ASCII
punctuation and articles, collapse whitespace) plus a token-F1 diagnostic.
Exact match is a strict proxy: valid paraphrases can be labelled incorrect.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

NORMALIZATION_STEPS = [
    "lowercase",
    "remove_ascii_punctuation",
    "remove_articles",
    "collapse_whitespace",
]


def normalize_answer(answer: str) -> str:
    """Apply SQuAD-style answer normalization."""
    answer = answer.lower()
    answer = "".join(c for c in answer if c not in string.punctuation)
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    return " ".join(answer.split())


def token_f1(prediction: str, reference: str) -> float:
    """Token-overlap F1 between one prediction and one reference."""
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)

    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def match_references(response: str, references: List[str]) -> Dict[str, Any]:
    """Exact-match and best-F1 scoring of one response against all references."""
    normalized_response = normalize_answer(response)
    normalized_references = {normalize_answer(r): r for r in references}
    matched_reference = normalized_references.get(normalized_response)
    best_f1, best_f1_reference = max(
        ((token_f1(response, r), r) for r in references), default=(0.0, None)
    )
    return {
        "normalized_response": normalized_response,
        "exact_match": matched_reference is not None,
        "matched_reference": matched_reference,
        "token_f1": best_f1,
        "best_f1_reference": best_f1_reference,
    }


# --------------------------------------------------------------------------
# Sampled (high-temperature) answers: uncertainty features
# --------------------------------------------------------------------------
def label_sampled_response(response: str, references: List[str]) -> Dict[str, Any]:
    """Label one sampled high-temperature response."""
    scored = match_references(response, references)
    return {
        "response": response,
        "normalized_response": scored["normalized_response"],
        "exact_match": scored["exact_match"],
        "correct": scored["exact_match"],
        "matched_reference": scored["matched_reference"],
        "token_f1": scored["token_f1"],
        "best_f1_reference": scored["best_f1_reference"],
    }


def evaluate_sampled_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Add correctness labels to clustered high-temperature samples."""
    questions = dataset.get("questions")
    if not isinstance(questions, dict):
        raise ValueError("Input JSON must contain a 'questions' object")

    evaluated: Dict[str, Any] = {}
    for question_id, entry in questions.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Question {question_id!r} is not an object")
        references = entry.get("reference_answers")
        responses = entry.get("responses")
        if not isinstance(references, list) or not all(
            isinstance(r, str) for r in references
        ):
            raise ValueError(f"Question {question_id!r} has invalid reference answers")
        if not isinstance(responses, list):
            raise ValueError(f"Question {question_id!r} has invalid responses")

        labeled = []
        for index, item in enumerate(responses):
            if not isinstance(item, dict) or not isinstance(item.get("response"), str):
                raise ValueError(f"Question {question_id!r} response {index} is invalid")
            record = label_sampled_response(item["response"], references)
            record["index"] = item.get("index", index)
            labeled.append(record)

        exact = sum(1 for r in labeled if r["exact_match"])
        evaluated[question_id] = {
            "question": entry.get("question"),
            "reference_answers": references,
            "responses": labeled,
            "entropy_nats": entry.get("entropy_nats"),
            "clusters": entry.get("clusters"),
            "num_responses": len(labeled),
            "exact_match_count": exact,
            "incorrect_count": len(labeled) - exact,
            "accuracy": exact / len(labeled) if labeled else 0.0,
            "mean_token_f1": (
                sum(r["token_f1"] for r in labeled) / len(labeled) if labeled else 0.0
            ),
        }

    return {
        "metadata": {
            "source_file": dataset.get("metadata", {}).get("source_file"),
            "correctness": "normalized_exact_match_and_token_f1",
            "generation": "high_temperature_samples",
            "normalization": NORMALIZATION_STEPS,
            "num_questions": len(evaluated),
        },
        "questions": evaluated,
    }


# --------------------------------------------------------------------------
# Low-temperature target answer: prediction label
# --------------------------------------------------------------------------
def score_low_temp_response(response: str, references: List[str]) -> Dict[str, Any]:
    """Score the single low-temperature target response (binary label)."""
    scored = match_references(response, references)
    return {
        "exact_match": scored["exact_match"],
        "correct": int(scored["exact_match"]),
        "matched_reference": scored["matched_reference"],
        "token_f1": scored["token_f1"],
    }


def evaluate_low_temp_dataset(
    dataset: Dict[str, Any], source_file: Path | str
) -> Dict[str, Any]:
    """One correctness record per question's low-temperature answer."""
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
            isinstance(r, str) for r in references
        ):
            raise ValueError(f"Question {question_id!r} has invalid reference answers")
        low_temp = entry.get("low_temp_generation")
        if not isinstance(low_temp, dict):
            raise ValueError(
                f"Question {question_id!r} has no valid low-temperature generation"
            )
        response = low_temp.get("response")
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
            "normalization": NORMALIZATION_STEPS,
            "num_questions": len(questions),
        },
        "questions": questions,
    }
