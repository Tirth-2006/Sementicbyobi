"""Export a balanced answer-pair sample for manual semantic validation."""
import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INPUT = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_clusters.json"
)
DEFAULT_OUTPUT = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_manual_cluster_validation.csv"
)
DEFAULT_PAIRS = 50
DEFAULT_SEED = 20261102
FIELDS = [
    "review_id",
    "question_id",
    "question",
    "answer_a",
    "answer_b",
    "nli_a_to_b_label",
    "nli_a_to_b_score",
    "nli_b_to_a_label",
    "nli_b_to_a_score",
    "nli_mutually_entail",
    "manual_semantically_equivalent",
    "review_notes",
]


def load_candidates(path: Path) -> list[dict[str, Any]]:
    """Load pairwise decisions and attach answer text for manual review."""
    with path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    questions = data.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("Input must contain a non-empty 'questions' object")

    candidates: list[dict[str, Any]] = []
    for question_id, record in questions.items():
        if not isinstance(record, dict):
            raise ValueError(f"Invalid record for {question_id!r}")
        question = record.get("question")
        responses = record.get("responses")
        comparisons = record.get("pairwise_comparisons")
        if not isinstance(question, str) or not isinstance(responses, list):
            raise ValueError(f"Invalid question or responses for {question_id!r}")
        if not isinstance(comparisons, list):
            raise ValueError(f"Missing pairwise comparisons for {question_id!r}")

        for comparison in comparisons:
            indices = comparison.get("answer_indices")
            if (
                not isinstance(indices, list)
                or len(indices) != 2
                or not all(isinstance(index, int) for index in indices)
            ):
                raise ValueError(f"Invalid pair indices for {question_id!r}")
            first, second = indices
            try:
                answer_a = responses[first]["response"]
                answer_b = responses[second]["response"]
            except (IndexError, KeyError, TypeError) as error:
                raise ValueError(
                    f"Invalid response indices for {question_id!r}"
                ) from error

            a_to_b = comparison.get("a_to_b")
            b_to_a = comparison.get("b_to_a")
            if not isinstance(a_to_b, dict) or not isinstance(b_to_a, dict):
                raise ValueError(f"Invalid NLI results for {question_id!r}")
            candidates.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "answer_a": answer_a,
                    "answer_b": answer_b,
                    "nli_a_to_b_label": a_to_b.get("label"),
                    "nli_a_to_b_score": a_to_b.get("score"),
                    "nli_b_to_a_label": b_to_a.get("label"),
                    "nli_b_to_a_score": b_to_a.get("score"),
                    "nli_mutually_entail": comparison.get("mutually_entail"),
                }
            )
    return candidates


def sample_candidates(
    candidates: list[dict[str, Any]],
    pair_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select a deterministic balanced sample of predicted positives/negatives."""
    if pair_count <= 0 or pair_count % 2:
        raise ValueError("--pairs must be a positive even number")
    positives = [
        candidate
        for candidate in candidates
        if candidate["nli_mutually_entail"] is True
    ]
    negatives = [
        candidate
        for candidate in candidates
        if candidate["nli_mutually_entail"] is False
    ]
    per_class = pair_count // 2
    if len(positives) < per_class or len(negatives) < per_class:
        raise ValueError(
            f"Not enough pairs for a balanced sample of {pair_count}: "
            f"{len(positives)} positive and {len(negatives)} negative candidates"
        )

    rng = np.random.default_rng(seed)
    selected_indices = np.concatenate(
        [
            rng.choice(len(positives), size=per_class, replace=False),
            rng.choice(len(negatives), size=per_class, replace=False),
        ]
    )
    selected = [
        (positives if position < per_class else negatives)[index]
        for position, index in enumerate(selected_indices)
    ]
    rng.shuffle(selected)
    for review_number, candidate in enumerate(selected, start=1):
        candidate["review_id"] = f"pair_{review_number:03d}"
        candidate["manual_semantically_equivalent"] = ""
        candidate["review_notes"] = ""
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export answer pairs for manual semantic-equivalence labels"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pairs", type=int, default=DEFAULT_PAIRS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    candidates = load_candidates(args.input)
    selected = sample_candidates(candidates, args.pairs, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(selected)

    positive_count = sum(
        row["nli_mutually_entail"] is True for row in selected
    )
    print(f"Candidate pairs: {len(candidates)}")
    print(f"Exported pairs: {len(selected)}")
    print(f"NLI predicted equivalent: {positive_count}")
    print(f"NLI predicted non-equivalent: {len(selected) - positive_count}")
    print(f"Output file: {args.output}")
    print(
        "Fill manual_semantically_equivalent with 1 for equivalent or 0 "
        "for non-equivalent; optionally add review_notes."
    )


if __name__ == "__main__":
    main()
