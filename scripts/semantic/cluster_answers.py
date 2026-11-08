"""CLI: cluster one generated question's answers using bidirectional NLI.

Reusable logic lives in :mod:`uncertainty.semantic.clustering`; this script is a
thin command-line wrapper.
"""
import argparse
from pathlib import Path
from typing import List

from uncertainty.semantic.clustering import (
    NLI_MODEL_NAME as MODEL_NAME,
    PairComparison,
    cluster_answers,
    load_nli_pipeline,
    load_question,
)
from uncertainty.semantic.semantic_entropy import calculate_entropy

DEFAULT_INPUT = Path("data/generated/trivia_qa_qwen3_20261024_162539.json")
DEFAULT_QUESTION_ID = "bb_6487"


def print_selected_comparisons(
    answers: List[str],
    comparisons: List[PairComparison],
) -> None:
    """Print pairwise decisions involving the vascular-related answers."""
    selected_terms = {
        "vascular tissue",
        "vascular systems",
        "vascular tissues",
        "Vessels",
        "Vascular plants",
    }
    print("\nSelected pairwise NLI decisions:")
    for comparison in comparisons:
        first, second = comparison["answer_indices"]
        if answers[first] not in selected_terms and answers[second] not in selected_terms:
            continue
        print(
            f"  [{first}] {answers[first]!r} <-> "
            f"[{second}] {answers[second]!r}: "
            f"{'mutual entailment' if comparison['mutually_entail'] else 'not mutual entailment'} "
            f"(A->B: {comparison['a_to_b']['label']} "
            f"{comparison['a_to_b']['score']:.3f}; "
            f"B->A: {comparison['b_to_a']['label']} "
            f"{comparison['b_to_a']['score']:.3f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster one generated question's answers with DeBERTa NLI"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Generated JSON file",
    )
    parser.add_argument(
        "--question-id",
        default=DEFAULT_QUESTION_ID,
        help="Question ID to inspect",
    )
    args = parser.parse_args()

    print(f"Loading {MODEL_NAME}...")
    nli = load_nli_pipeline(MODEL_NAME)

    question, answers = load_question(args.input, args.question_id)
    clusters, comparisons = cluster_answers(nli, question, answers)
    entropy = calculate_entropy(clusters, len(answers))

    print(f"\nQuestion ID: {args.question_id}")
    print(f"Question: {question}")
    print(f"Responses: {len(answers)}")
    print("\nClusters:")
    for cluster_number, cluster in enumerate(clusters, start=1):
        print(f"  Cluster {cluster_number} (size={len(cluster)}):")
        for index in cluster:
            print(f"    [{index}] {answers[index]}")

    print_selected_comparisons(answers, comparisons)

    print("\nFrequency-based semantic entropy:")
    print(f"  H = {entropy:.6f} nats")
    print(
        f"  Formula: -sum((cluster_size/{len(answers)}) * "
        f"ln(cluster_size/{len(answers)}))"
    )
    print(f"\nMutually entailing pairs: "
          f"{sum(item['mutually_entail'] for item in comparisons)}"
          f"/{len(comparisons)}")


if __name__ == "__main__":
    main()