# Predictor-state (`p-1`) vs post-token (`p`) dispersion

* [`report.md`](report.md): three-way metrics table and paired bootstrap intervals.
* [`metrics.csv`](metrics.csv): per-configuration metrics.

Predictor-state dispersion has the best point estimates (AUROC 0.839168, AURAC 0.515165), but the
paired AUROC/AURAC intervals against the entropy baseline, and between `p-1` and `p`, include zero.
The two alignments are different representations and are not interchangeable.
