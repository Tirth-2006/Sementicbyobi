"""Measure greedy clustering sensitivity to answer ordering."""
import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INPUT = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_clusters.json"
)
DEFAULT_OUTPUT = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_order_sensitivity.json"
)
DEFAULT_PERMUTATIONS = 1000
DEFAULT_SEED = 20261102


def canonical_partition(clusters: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    """Represent a partition independently of cluster and member ordering."""
    return tuple(sorted(tuple(sorted(cluster)) for cluster in clusters))


def build_compatibility_map(
    comparisons: list[dict[str, Any]],
) -> dict[tuple[int, int], bool]:
    """Extract saved mutual-entailment decisions by original answer indices."""
    compatibility: dict[tuple[int, int], bool] = {}
    for comparison in comparisons:
        indices = comparison.get("answer_indices")
        if (
            not isinstance(indices, list)
            or len(indices) != 2
            or not all(isinstance(index, int) for index in indices)
        ):
            raise ValueError("Invalid answer_indices in pairwise comparison")
        first, second = indices
        pair = (min(first, second), max(first, second))
        mutually_entail = comparison.get("mutually_entail")
        if not isinstance(mutually_entail, bool):
            raise ValueError("Invalid mutually_entail value in comparison")
        compatibility[pair] = mutually_entail
    return compatibility


def greedy_clusters(
    answer_count: int,
    order: list[int],
    compatibility: dict[tuple[int, int], bool],
) -> list[list[int]]:
    """Apply the same first-compatible complete-linkage assignment."""
    clusters: list[list[int]] = []
    for answer_index in order:
        for cluster in clusters:
            if all(
                compatibility[(min(answer_index, existing), max(answer_index, existing))]
                for existing in cluster
            ):
                cluster.append(answer_index)
                break
        else:
            clusters.append([answer_index])

    assigned = [index for cluster in clusters for index in cluster]
    if sorted(assigned) != list(range(answer_count)):
        raise ValueError("Greedy clustering did not assign every answer exactly once")
    return clusters


def calculate_entropy(clusters: list[list[int]]) -> float:
    """Calculate frequency-based Shannon entropy in nats."""
    sizes = [len(cluster) for cluster in clusters]
    total = sum(sizes)
    if total <= 0:
        raise ValueError("Cannot calculate entropy without answers")
    return float(
        -sum(
            (size / total) * math.log(size / total)
            for size in sizes
        )
    )


def validate_record(
    question_id: str,
    record: dict[str, Any],
) -> tuple[int, tuple[tuple[int, ...], ...], float, dict[tuple[int, int], bool]]:
    """Validate one saved clustering record and extract reusable data."""
    responses = record.get("responses")
    comparisons = record.get("pairwise_comparisons")
    original_clusters = record.get("clusters")
    original_entropy = record.get("entropy_nats")
    if not isinstance(responses, list) or not responses:
        raise ValueError(f"{question_id!r} has no valid responses")
    if not isinstance(comparisons, list):
        raise ValueError(f"{question_id!r} has no pairwise comparisons")
    if not isinstance(original_clusters, list):
        raise ValueError(f"{question_id!r} has no valid clusters")
    if isinstance(original_entropy, bool) or not isinstance(original_entropy, (int, float)):
        raise ValueError(f"{question_id!r} has invalid entropy")
    if not math.isfinite(original_entropy) or original_entropy < 0:
        raise ValueError(f"{question_id!r} has invalid entropy")

    cluster_indices: list[list[int]] = []
    for cluster in original_clusters:
        if not isinstance(cluster, dict) or not isinstance(cluster.get("indices"), list):
            raise ValueError(f"{question_id!r} has an invalid cluster")
        cluster_indices.append(cluster["indices"])

    answer_count = len(responses)
    compatibility = build_compatibility_map(comparisons)
    expected_pairs = answer_count * (answer_count - 1) // 2
    if len(compatibility) != expected_pairs:
        raise ValueError(
            f"{question_id!r} has {len(compatibility)} unique pairs; "
            f"expected {expected_pairs}"
        )
    original_partition = canonical_partition(cluster_indices)
    expected_partition = canonical_partition(
        greedy_clusters(answer_count, list(range(answer_count)), compatibility)
    )
    if original_partition != expected_partition:
        raise ValueError(
            f"{question_id!r} does not match the saved greedy clustering"
        )
    return answer_count, original_partition, float(original_entropy), compatibility


def summarize_question(
    record: dict[str, Any],
    question_id: str,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Run shuffled greedy assignments and summarize one question."""
    answer_count, original_partition, original_entropy, compatibility = (
        validate_record(question_id, record)
    )
    entropies = np.empty(permutations, dtype=float)
    partitions: set[tuple[tuple[int, ...], ...]] = set()
    changed_count = 0

    for permutation_index in range(permutations):
        order = rng.permutation(answer_count).tolist()
        clusters = greedy_clusters(answer_count, order, compatibility)
        partition = canonical_partition(clusters)
        partitions.add(partition)
        entropies[permutation_index] = calculate_entropy(clusters)
        if partition != original_partition:
            changed_count += 1

    return {
        "question_id": question_id,
        "answer_count": answer_count,
        "original_entropy_nats": original_entropy,
        "permutations": permutations,
        "changed_clusterings": changed_count,
        "changed_clustering_rate": changed_count / permutations,
        "unique_partitions": len(partitions),
        "entropy_nats": {
            "minimum": float(entropies.min()),
            "maximum": float(entropies.max()),
            "mean": float(entropies.mean()),
            "std": float(entropies.std()),
            "minimum_delta_from_original": float(
                (entropies - original_entropy).min()
            ),
            "maximum_delta_from_original": float(
                (entropies - original_entropy).max()
            ),
            "mean_delta_from_original": float(
                (entropies - original_entropy).mean()
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure clustering and entropy sensitivity to answer order"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--permutations",
        type=int,
        default=DEFAULT_PERMUTATIONS,
        help="Number of shuffled orders per question",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.permutations <= 0:
        raise ValueError("--permutations must be positive")

    with args.input.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    questions = data.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("Input must contain a non-empty 'questions' object")

    rng = np.random.default_rng(args.seed)
    question_results = [
        summarize_question(record, question_id, args.permutations, rng)
        for question_id, record in questions.items()
        if isinstance(record, dict)
    ]
    if len(question_results) != len(questions):
        raise ValueError("Every question record must be an object")

    changed_questions = sum(
        result["changed_clusterings"] > 0 for result in question_results
    )
    all_changed = sum(
        result["changed_clusterings"] for result in question_results
    )
    total_runs = len(question_results) * args.permutations
    entropy_ranges = [
        result["entropy_nats"]["maximum"] - result["entropy_nats"]["minimum"]
        for result in question_results
    ]
    output = {
        "metadata": {
            "input_file": str(args.input),
            "num_questions": len(question_results),
            "permutations_per_question": args.permutations,
            "seed": args.seed,
            "nli_rerun": False,
            "method": "saved_pairwise_decisions_with_greedy_complete_linkage",
        },
        "summary": {
            "questions_with_order_dependent_clusterings": changed_questions,
            "question_order_dependence_rate": (
                changed_questions / len(question_results)
            ),
            "changed_clusterings": all_changed,
            "total_shuffled_runs": total_runs,
            "overall_changed_clustering_rate": all_changed / total_runs,
            "mean_question_entropy_range_nats": float(np.mean(entropy_ranges)),
            "maximum_question_entropy_range_nats": float(np.max(entropy_ranges)),
        },
        "questions": question_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(output, output_file, indent=2)
        output_file.write("\n")

    print(f"Questions: {len(question_results)}")
    print(f"Permutations per question: {args.permutations}")
    print(
        "Questions with changed clusterings: "
        f"{changed_questions}/{len(question_results)}"
    )
    print(f"Overall changed-clustering rate: {all_changed / total_runs:.6f}")
    print(f"Mean entropy range: {np.mean(entropy_ranges):.6f} nats")
    print(f"Maximum entropy range: {np.max(entropy_ranges):.6f} nats")
    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
