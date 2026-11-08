"""The tracked manual review must support the confusion counts quoted in the docs."""
import csv
from collections import Counter
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "annotations" / "nli_pairs_review_filled.csv"


def unique_pairs():
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    pairs = {}
    for row in rows:
        answers = frozenset((row["answer_a"].lower().strip(), row["answer_b"].lower().strip()))
        pairs.setdefault((row["question_id"], answers), row)
    return rows, list(pairs.values())


def test_row_and_question_counts():
    rows, _ = unique_pairs()
    assert len(rows) == 50
    assert len({r["question_id"] for r in rows}) == 37


def test_unique_pair_confusion_matrix_matches_documentation():
    _, pairs = unique_pairs()
    assert len(pairs) == 44
    counts = Counter(
        (r["nli_mutually_entail"] == "True", r["manual_semantically_equivalent"] == "1")
        for r in pairs
    )
    assert counts[(True, True)] == 19   # TP
    assert counts[(True, False)] == 1   # FP
    assert counts[(False, True)] == 1   # FN
    assert counts[(False, False)] == 23  # TN
