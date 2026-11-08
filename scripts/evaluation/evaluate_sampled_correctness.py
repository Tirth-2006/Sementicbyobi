"""CLI: label the *sampled high-temperature* answers against TriviaQA references.

This scores the ten T=1.0 responses per question (the uncertainty features'
source). It is NOT the low-temperature target evaluator; see
``evaluate_low_temp_correctness.py`` for that. Logic lives in
:mod:`uncertainty.evaluation.correctness`.
"""
import argparse
import json
from pathlib import Path

from uncertainty.evaluation.correctness import evaluate_sampled_dataset

DEFAULT_INPUT = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_clusters.json"
)
DEFAULT_OUTPUT = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_correctness_f1.json"
)


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

    results = evaluate_sampled_dataset(dataset)
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
