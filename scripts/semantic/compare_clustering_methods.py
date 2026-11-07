"""Compare greedy clustering with order-independent graph components."""
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


CLUSTERS_INPUT = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_clusters.json"
)
AUROC_DATASET_INPUT = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_auroc_dataset.csv"
)
OUTPUT_PATH = Path(
    "data/generated/trivia_qa_qwen3_20261024_162539_clustering_comparison.json"
)


def canonical_partition(clusters: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    """Represent a partition independently of cluster ordering."""
    return tuple(sorted(tuple(sorted(cluster)) for cluster in clusters))


def entropy(clusters: list[list[int]]) -> float:
    """Calculate frequency-based Shannon entropy in nats."""
    sizes = [len(cluster) for cluster in clusters]
    total = sum(sizes)
    if total <= 0:
        raise ValueError("Cannot calculate entropy without answers")
    return float(-sum((size / total) * math.log(size / total) for size in sizes))


def compatibility_map(
    comparisons: list[dict[str, Any]],
    answer_count: int,
) -> dict[tuple[int, int], bool]:
    """Validate and index saved pairwise mutual-entailment decisions."""
    result: dict[tuple[int, int], bool] = {}
    for comparison in comparisons:
        indices = comparison.get("answer_indices")
        if (
            not isinstance(indices, list)
            or len(indices) != 2
            or not all(isinstance(index, int) for index in indices)
        ):
            raise ValueError("Invalid answer_indices in pairwise comparison")
        first, second = indices
        if not 0 <= first < answer_count or not 0 <= second < answer_count:
            raise ValueError("Pairwise index is outside the response range")
        pair = (min(first, second), max(first, second))
        if pair in result:
            raise ValueError("Duplicate pairwise comparison")
        mutually_entail = comparison.get("mutually_entail")
        if not isinstance(mutually_entail, bool):
            raise ValueError("Invalid mutually_entail value")
        result[pair] = mutually_entail

    expected_pairs = answer_count * (answer_count - 1) // 2
    if len(result) != expected_pairs:
        raise ValueError(f"Expected {expected_pairs} pairs, found {len(result)}")
    return result


def greedy_clusters(
    answer_count: int,
    compatibility: dict[tuple[int, int], bool],
) -> list[list[int]]:
    """Reproduce the original first-compatible greedy assignment."""
    clusters: list[list[int]] = []
    for answer_index in range(answer_count):
        for cluster in clusters:
            if all(
                compatibility[(min(answer_index, existing), max(answer_index, existing))]
                for existing in cluster
            ):
                cluster.append(answer_index)
                break
        else:
            clusters.append([answer_index])
    return clusters


def connected_components(
    answer_count: int,
    compatibility: dict[tuple[int, int], bool],
) -> list[list[int]]:
    """Build order-independent components of the compatibility graph."""
    adjacency = [[] for _ in range(answer_count)]
    for (first, second), compatible in compatibility.items():
        if compatible:
            adjacency[first].append(second)
            adjacency[second].append(first)

    components: list[list[int]] = []
    unseen = set(range(answer_count))
    while unseen:
        start = min(unseen)
        stack = [start]
        unseen.remove(start)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def load_dataset(path: Path) -> dict[str, tuple[float, int]]:
    """Load saved question-level entropy and correctness."""
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if len(rows) != 100:
        raise ValueError(f"Expected 100 rows, found {len(rows)}")
    result: dict[str, tuple[float, int]] = {}
    for row in rows:
        question_id = row.get("question_id")
        entropy_text = row.get("entropy_nats")
        correct_text = row.get("correct")
        if not question_id or not entropy_text or correct_text not in {"0", "1"}:
            raise ValueError("Invalid AUROC dataset row")
        if question_id in result:
            raise ValueError(f"Duplicate question ID: {question_id}")
        result[question_id] = (float(entropy_text), int(correct_text))
    return result


def aurac(question_ids: list[str], entropy_values: np.ndarray, correct: np.ndarray) -> float:
    """Calculate AURAC using descending entropy rejection and ID tie-breaking."""
    order = np.asarray(
        sorted(
            range(len(question_ids)),
            key=lambda index: (-entropy_values[index], question_ids[index]),
        ),
        dtype=int,
    )
    total = len(correct)
    retained = total - np.arange(total + 1)
    rejected_correct = np.concatenate(([0], np.cumsum(correct[order], dtype=int)))
    accuracy = np.full(total + 1, np.nan, dtype=float)
    accuracy[:-1] = (int(correct.sum()) - rejected_correct[:-1]) / retained[:-1]
    accuracy[-1] = accuracy[-2]
    coverage = retained / total
    return float(np.trapezoid(accuracy[::-1], coverage[::-1]))


def main() -> None:
    with CLUSTERS_INPUT.open("r", encoding="utf-8") as input_file:
        cluster_data = json.load(input_file)
    questions = cluster_data.get("questions")
    if not isinstance(questions, dict) or len(questions) != 100:
        raise ValueError("Clustering JSON must contain 100 questions")

    dataset = load_dataset(AUROC_DATASET_INPUT)
    if set(dataset) != set(questions):
        raise ValueError("Clustering and AUROC question IDs do not match")

    question_results: list[dict[str, Any]] = []
    component_entropies: dict[str, float] = {}
    greedy_entropies: dict[str, float] = {}
    partition_changes = 0

    for question_id, record in questions.items():
        if not isinstance(record, dict):
            raise ValueError(f"Invalid record for {question_id}")
        responses = record.get("responses")
        comparisons = record.get("pairwise_comparisons")
        saved_clusters = record.get("clusters")
        if not isinstance(responses, list) or not isinstance(comparisons, list):
            raise ValueError(f"Invalid response data for {question_id}")
        if not isinstance(saved_clusters, list):
            raise ValueError(f"Invalid saved clusters for {question_id}")

        answer_count = len(responses)
        compatibility = compatibility_map(comparisons, answer_count)
        greedy = greedy_clusters(answer_count, compatibility)
        components = connected_components(answer_count, compatibility)
        saved_partition = canonical_partition(
            [cluster["indices"] for cluster in saved_clusters]
        )
        greedy_partition = canonical_partition(greedy)
        component_partition = canonical_partition(components)
        if saved_partition != greedy_partition:
            raise ValueError(f"Saved greedy partition mismatch for {question_id}")

        greedy_value = entropy(greedy)
        component_value = entropy(components)
        greedy_entropies[question_id] = greedy_value
        component_entropies[question_id] = component_value
        changed = greedy_partition != component_partition
        partition_changes += changed
        question_results.append(
            {
                "question_id": question_id,
                "greedy_cluster_sizes": sorted(map(len, greedy)),
                "component_cluster_sizes": sorted(map(len, components)),
                "greedy_entropy_nats": greedy_value,
                "component_entropy_nats": component_value,
                "entropy_delta_components_minus_greedy": component_value - greedy_value,
                "partition_differs": changed,
            }
        )

    question_ids = sorted(dataset)
    correct = np.asarray([dataset[q][1] for q in question_ids], dtype=int)
    greedy_values = np.asarray([greedy_entropies[q] for q in question_ids])
    component_values = np.asarray([component_entropies[q] for q in question_ids])
    results = {
        "metadata": {
            "n_questions": len(question_results),
            "nli_rerun": False,
            "greedy_method": "first-compatible-greedy-complete-linkage",
            "alternative_method": (
                "connected_components_of_mutual_entailment_graph"
            ),
            "alternative_order_independent": True,
            "alternative_quality_caution": (
                "Connected components can merge incompatible answers through "
                "transitive bridges."
            ),
        },
        "comparison": {
            "questions_with_different_partitions": partition_changes,
            "partition_difference_rate": partition_changes / len(question_results),
            "mean_entropy_greedy_nats": float(greedy_values.mean()),
            "mean_entropy_components_nats": float(component_values.mean()),
            "mean_entropy_delta_nats": float(
                (component_values - greedy_values).mean()
            ),
            "maximum_absolute_entropy_delta_nats": float(
                np.max(np.abs(component_values - greedy_values))
            ),
        },
        "metrics": {
            "greedy": {
                "auroc": float(roc_auc_score(correct, -greedy_values)),
                "aurac": aurac(question_ids, greedy_values, correct),
            },
            "connected_components": {
                "auroc": float(roc_auc_score(correct, -component_values)),
                "aurac": aurac(question_ids, component_values, correct),
            },
        },
        "questions": question_results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)
        output_file.write("\n")

    print(f"Questions with different partitions: {partition_changes}/100")
    print(
        f"Greedy: AUROC={results['metrics']['greedy']['auroc']:.6f}, "
        f"AURAC={results['metrics']['greedy']['aurac']:.6f}"
    )
    print(
        "Connected components: "
        f"AUROC={results['metrics']['connected_components']['auroc']:.6f}, "
        f"AURAC={results['metrics']['connected_components']['aurac']:.6f}"
    )
    print(f"Saved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
