"""Evaluate semantic entropy across an existing generated-answer dataset."""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import pipeline

from scripts.semantic.cluster_answers import (
    ENTAILMENT_THRESHOLD,
    MODEL_NAME,
    NLIModel,
    calculate_entropy,
    cluster_answers,
)


DEFAULT_INPUT = Path("data/generated/trivia_qa_qwen3_20261024_162539.json")
DEFAULT_OUTPUT = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_clusters.json"
)


def load_dataset(input_path: Path) -> Dict[str, Any]:
    """Load a generated dataset JSON file."""
    with input_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)
    if not isinstance(dataset, dict):
        raise ValueError(f"Input JSON must contain an object: {input_path}")
    return dataset


def load_question_from_entry(
    entry: Dict[str, Any],
    question_id: str,
) -> Tuple[str, List[str]]:
    """Extract one question and its high-temperature responses."""
    try:
        question = entry["question"]
        response_group = entry["high_temp_generations"]["responses"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Question {question_id!r} does not contain the expected "
            "high-temperature response structure"
        ) from error

    if not isinstance(question, str):
        raise ValueError(f"Question {question_id!r} has no valid question text")
    if not isinstance(response_group, list) or not response_group:
        raise ValueError(
            f"Question {question_id!r} has no valid response list"
        )

    responses: List[str] = []
    for index, item in enumerate(response_group):
        if not isinstance(item, dict):
            raise ValueError(
                f"Question {question_id!r} response {index} is not an object"
            )
        response = item.get("response")
        if not isinstance(response, str):
            raise ValueError(
                f"Question {question_id!r} response {index} does not "
                "contain a string 'response' field"
            )
        responses.append(response)

    return question, responses


def evaluate_dataset(
    nli: NLIModel,
    dataset: Dict[str, Any],
    source_file: Path,
) -> Dict[str, Any]:
    """Cluster every question and calculate frequency-based entropy."""
    generations = dataset.get("generations")
    if not isinstance(generations, dict):
        raise ValueError("Input JSON must contain a 'generations' object")

    questions: Dict[str, Any] = {}
    for question_id, entry in generations.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Question {question_id!r} is not an object")

        question, answers = load_question_from_entry(entry, question_id)
        clusters, comparisons = cluster_answers(nli, question, answers)
        questions[question_id] = {
            "question": question,
            "reference_answers": entry.get("reference_answers", []),
            "responses": [
                {"index": index, "response": answer}
                for index, answer in enumerate(answers)
            ],
            "clusters": [
                {
                    "indices": cluster,
                    "responses": [answers[index] for index in cluster],
                    "size": len(cluster),
                }
                for cluster in clusters
            ],
            "entropy_nats": calculate_entropy(clusters, len(answers)),
            "num_responses": len(answers),
            "num_pairs": len(comparisons),
            "mutually_entailing_pairs": sum(
                item["mutually_entail"] for item in comparisons
            ),
            "pairwise_comparisons": comparisons,
        }

    return {
        "metadata": {
            "source_file": str(source_file),
            "model": MODEL_NAME,
            "clustering": "greedy_complete_linkage",
            "entailment_threshold": ENTAILMENT_THRESHOLD,
            "entropy": "frequency_based_shannon_entropy",
            "num_questions": len(questions),
        },
        "questions": questions,
    }


def save_results(output_path: Path, results: Dict[str, Any]) -> None:
    """Save dataset-wide evaluation results as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate semantic entropy on generated answer data"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    dataset = load_dataset(args.input)
    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading {MODEL_NAME} on device {device}...")
    nli = pipeline("text-classification", model=MODEL_NAME, device=device)

    results = evaluate_dataset(nli, dataset, args.input)
    save_results(args.output, results)
    print(f"Evaluated {results['metadata']['num_questions']} questions")
    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
