"""MANUAL check (not collected by pytest): downloads and runs DeBERTa NLI.

Run:  python manual_checks/check_nli.py
"""

import torch
from transformers import pipeline

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

device = 0 if torch.cuda.is_available() else -1

print("Loading DeBERTa-NLI...")

nli = pipeline(
    "text-classification",
    model=MODEL_NAME,
    device=device
)

def compare_answers(question, answer_a, answer_b):
    premise = f"Question: {question}\nAnswer: {answer_a}"
    hypothesis = f"Question: {question}\nAnswer: {answer_b}"

    result_ab = nli(
        premise,
        text_pair=hypothesis
    )

    result_ba = nli(
        hypothesis,
        text_pair=premise
    )

    print("\nAnswer A:", answer_a)
    print("Answer B:", answer_b)
    print("A -> B:", result_ab)
    print("B -> A:", result_ba)

compare_answers(
    "What is the capital of France?",
    "Paris is the capital of France.",
    "France's capital city is Paris."
)

compare_answers(
    "What is the capital of France?",
    "Paris is the capital of France.",
    "Lyon is the capital of France."
)
