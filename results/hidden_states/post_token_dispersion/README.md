# Post-token dispersion (`p`) predictive test

Adds post-token dispersion (hidden-state index 28, mean pooling) to semantic + token entropy.
Dispersion = mean Euclidean distance of the 10 answer vectors from their centroid (not variance).

* [`report.md`](report.md): metrics table and paired bootstrap interval (20,000 resamples, seed `20261106`).
* [`metrics.csv`](metrics.csv): per-configuration metrics.

The paired AUROC and AURAC intervals include zero; no incremental improvement is established.
Standalone raw-score dispersion (AUROC 0.757991, 95% CI [0.638400, 0.862871]) is in the main README.
