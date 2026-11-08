# Semantic entropy (raw score)

Empirical semantic-cluster entropy from greedy complete-linkage bidirectional NLI clustering of
10 answers at T=1.0. Higher entropy = higher uncertainty. AUROC treats incorrectness as positive;
Spearman is against correctness (1 = correct).

| Metric | Point estimate | 95% bootstrap interval |
|---|---:|---:|
| Spearman | -0.528199 | [-0.651525, -0.381724] |
| AUROC | 0.841451 | [0.753715, 0.914173] |
| AURAC | 0.462878 | [0.326030, 0.604548] |

Bootstrap: 20,000 question-level resamples, seed `20261031`.

**Random-rejection reference (same sample):** mean AURAC 0.269635, 95% simulation interval
[0.187234, 0.360535] (20,000 simulations); semantic AURAC exceeds the mean by 0.193243. This is not an
external benchmark.

**AURAC convention:** reject in descending uncertainty; ties by ascending `question_id`; trapezoidal
integration of retained accuracy over coverage; zero-coverage endpoint uses the one-retained-question
accuracy.

Files: [`aurac_curve.csv`](aurac_curve.csv) (rejection curve, rows 0-100% rejected).

Paired comparison with token entropy: see [`../token_entropy/README.md`](../token_entropy/README.md).
