"""Cluster generated answers for one question using bidirectional NLI."""
import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, List, Tuple, TypedDict, cast

import torch
from transformers import pipeline


MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
DEFAULT_INPUT = Path("data/generated/trivia_qa_qwen3_20261024_162539.json")
DEFAULT_QUESTION_ID = "bb_6487"
ENTAILMENT_THRESHOLD = 0.5


class NLIResult(TypedDict):
    """One Transformers text-classification result."""

    label: str
    score: float


class PairComparison(TypedDict):
    """Bidirectional NLI result for one answer pair."""

    answer_indices: List[int]
    mutually_entail: bool
    a_to_b: NLIResult
    b_to_a: NLIResult


NLIModel = Callable[..., List[dict[str, Any]]]


def load_question(input_path: Path, question_id: str) -> Tuple[str, List[str]]:
    """Load one question and its high-temperature responses from JSON."""
    with input_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    try:
        entry = dataset["generations"][question_id]
    except KeyError as error:
        raise KeyError(
            f"Question ID {question_id!r} was not found in {input_path}"
        ) from error

    try:
        question = entry["question"]
        responses = [
            item["response"]
            for item in entry["high_temp_generations"]["responses"]
        ]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Question {question_id!r} does not contain the expected "
            "high-temperature response structure"
        ) from error

    if not responses:
        raise ValueError(f"Question {question_id!r} has no generated responses")

    return question, responses


def compare_answers(
    nli: NLIModel,
    question: str,
    answer_a: str,
    answer_b: str,
) -> Tuple[bool, NLIResult, NLIResult]:
    """Check whether two answers mutually entail one another."""
    premise = f"Question: {question}\nAnswer: {answer_a}"
    hypothesis = f"Question: {question}\nAnswer: {answer_b}"

    result_ab = cast(NLIResult, nli(premise, text_pair=hypothesis)[0])
    result_ba = cast(NLIResult, nli(hypothesis, text_pair=premise)[0])

    mutually_entail = (
        result_ab["label"].lower() == "entailment"
        and result_ba["label"].lower() == "entailment"
        and result_ab["score"] >= ENTAILMENT_THRESHOLD
        and result_ba["score"] >= ENTAILMENT_THRESHOLD
    )
    return mutually_entail, result_ab, result_ba


def cluster_answers(
    nli: NLIModel,
    question: str,
    answers: List[str],
) -> Tuple[List[List[int]], List[PairComparison]]:
    """Build conservative clusters using complete-linkage mutual entailment."""
    clusters: List[List[int]] = []
    comparisons: List[PairComparison] = []

    for first in range(len(answers)):
        for second in range(first + 1, len(answers)):
            mutually_entail, result_ab, result_ba = compare_answers(
                nli, question, answers[first], answers[second]
            )
            comparisons.append(
                {
                    "answer_indices": [first, second],
                    "mutually_entail": mutually_entail,
                    "a_to_b": result_ab,
                    "b_to_a": result_ba,
                }
            )

    comparison_by_pair = {
        tuple(item["answer_indices"]): item for item in comparisons
    }

    for first in range(len(answers)):
        matching_clusters = []
        for cluster_index, cluster in enumerate(clusters):
            all_entail = True
            for existing in cluster:
                pair = (min(first, existing), max(first, existing))
                comparison = comparison_by_pair[pair]
                if not comparison["mutually_entail"]:
                    all_entail = False
                    break
            if all_entail:
                matching_clusters.append(cluster_index)

        if matching_clusters:
            clusters[matching_clusters[0]].append(first)
        else:
            clusters.append([first])

    return clusters, comparisons


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


def calculate_entropy(clusters: List[List[int]], total_answers: int) -> float:
    """Calculate frequency-based entropy over semantic clusters."""
    entropy = 0.0
    for cluster in clusters:
        probability = len(cluster) / total_answers
        entropy -= probability * math.log(probability)
    return entropy


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

    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading {MODEL_NAME} on device {device}...")
    nli = pipeline("text-classification", model=MODEL_NAME, device=device)

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