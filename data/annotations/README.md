# Annotations

`nli_pairs_review_filled.csv` is a small, **single-annotator** manual review of answer pairs used to
audit the NLI-based semantic clustering.

* 50 rows covering 37 questions; 44 unique pairs after removing duplicate rows and A/B-swapped
  duplicates (marked in `review_notes`).
* Pairs were selected using model predictions, and pairs are nested within questions, so this is an
  audit aid, **not** a definitive human benchmark.
* On the 44 unique pairs: TP 19, FP 1, FN 1, TN 23 (accuracy 95.45%).

| Column | Meaning |
|---|---|
| `review_id`, `question_id`, `question` | Row and question identifiers |
| `answer_a`, `answer_b` | The two sampled answers compared |
| `nli_a_to_b_label` / `_score`, `nli_b_to_a_label` / `_score` | DeBERTa-NLI output in each direction |
| `nli_mutually_entail` | Model decision (both directions entailment, each score ≥ 0.5) |
| `manual_semantically_equivalent` | Human label: 1 = same meaning for the question, 0 = different |
| `review_notes` | Free-text notes, including duplicate markers |

Produced by `scripts/analysis/prepare_manual_cluster_validation.py` and scored by
`scripts/analysis/evaluate_manual_cluster_validation.py`.
