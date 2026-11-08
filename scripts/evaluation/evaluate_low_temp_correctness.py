"""CLI: evaluate the *low-temperature target* answers against TriviaQA references.

The T=0.1 answer per question provides the binary prediction label. This is
distinct from ``evaluate_sampled_correctness.py``, which scores the sampled
T=1.0 answers. Logic lives in :mod:`uncertainty.evaluation.correctness`.
"""
import argparse
import json
from pathlib import Path

from uncertainty.evaluation.correctness import evaluate_low_temp_dataset

DEFAULT_INPUT = Path("data/generated/trivia_qa_qwen3_20261024_162539.json")
DEFAULT_OUTPUT = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_low_temp_correctness.json"
)


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

    results = evaluate_low_temp_dataset(dataset, args.input)
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
