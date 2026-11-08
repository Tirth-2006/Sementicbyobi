"""Bidirectional-NLI semantic clustering of sampled answers.

The NLI model is injected as a callable, so the clustering logic is testable
without downloading any model. ``load_nli_pipeline`` builds the real
DeBERTa pipeline lazily (importing ``transformers`` only when called).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, List, Tuple, TypedDict, cast

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
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


def load_nli_pipeline(model_name: str = NLI_MODEL_NAME) -> NLIModel:
    """Create the Transformers NLI pipeline (heavy; imports lazily)."""
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    return cast(
        NLIModel,
        pipeline("text-classification", model=model_name, device=device),
    )


def load_question(input_path: Path, question_id: str) -> Tuple[str, List[str]]:
    """Load one question and its high-temperature responses from JSON."""
    with Path(input_path).open("r", encoding="utf-8") as file:
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
            item["response"] for item in entry["high_temp_generations"]["responses"]
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
    threshold: float = ENTAILMENT_THRESHOLD,
) -> Tuple[bool, NLIResult, NLIResult]:
    """Check whether two answers mutually entail one another."""
    premise = f"Question: {question}\nAnswer: {answer_a}"
    hypothesis = f"Question: {question}\nAnswer: {answer_b}"

    result_ab = cast(NLIResult, nli(premise, text_pair=hypothesis)[0])
    result_ba = cast(NLIResult, nli(hypothesis, text_pair=premise)[0])

    mutually_entail = (
        result_ab["label"].lower() == "entailment"
        and result_ba["label"].lower() == "entailment"
        and result_ab["score"] >= threshold
        and result_ba["score"] >= threshold
    )
    return mutually_entail, result_ab, result_ba


def cluster_answers(
    nli: NLIModel,
    question: str,
    answers: List[str],
    threshold: float = ENTAILMENT_THRESHOLD,
) -> Tuple[List[List[int]], List[PairComparison]]:
    """Build conservative clusters using greedy first-compatible complete linkage.

    A new answer joins the first existing cluster only if it mutually entails
    every member of that cluster; otherwise it starts a new cluster. The
    result depends on answer order.
    """
    clusters: List[List[int]] = []
    comparisons: List[PairComparison] = []

    for first in range(len(answers)):
        for second in range(first + 1, len(answers)):
            mutually_entail, result_ab, result_ba = compare_answers(
                nli, question, answers[first], answers[second], threshold
            )
            comparisons.append(
                {
                    "answer_indices": [first, second],
                    "mutually_entail": mutually_entail,
                    "a_to_b": result_ab,
                    "b_to_a": result_ba,
                }
            )

    comparison_by_pair = {tuple(item["answer_indices"]): item for item in comparisons}

    for first in range(len(answers)):
        matching_clusters = []
        for cluster_index, cluster in enumerate(clusters):
            all_entail = True
            for existing in cluster:
                pair = (min(first, existing), max(first, existing))
                if not comparison_by_pair[pair]["mutually_entail"]:
                    all_entail = False
                    break
            if all_entail:
                matching_clusters.append(cluster_index)

        if matching_clusters:
            clusters[matching_clusters[0]].append(first)
        else:
            clusters.append([first])

    return clusters, comparisons
