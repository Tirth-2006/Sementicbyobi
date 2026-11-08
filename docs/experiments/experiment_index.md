# Experiment index

Current index of experiments (8 November 2026). Run every script from the repository root.
Most scripts read **frozen artifacts** under `data/generated/` that are not all committed; see
[reproducibility](../reproducibility.md). The October development log is archived in
[`docs/archive/development_log_2026-10-24.md`](../archive/development_log_2026-10-24.md).

| ID | Experiment | Main script(s) | Needs | Reported in |
|---|---|---|---|---|
| E1 | Dual-temperature TriviaQA generation (10 × T=1.0, 1 × T=0.1) | `scripts/data/generate_data.py` | GPU, `UNCERTAINTY_MODEL_DIR`, TriviaQA | README setup |
| E2 | Bidirectional-NLI clustering → empirical semantic entropy | `scripts/semantic/cluster_answers.py`, `evaluate_semantic_entropy.py` | E1 output, DeBERTa-NLI | README §1 |
| E3a | Low-temperature **target** correctness labels | `scripts/evaluation/evaluate_low_temp_correctness.py` | E1 output | README setup |
| E3b | Sampled high-temperature answer labels (diagnostic) | `scripts/evaluation/evaluate_sampled_correctness.py` | E2 output | — |
| E4 | Teacher-forced answer scoring | `scripts/token_entropy/score_teacher_forced.py` | E1, E2, GPU | README §2 |
| E5 | Token predictive entropy; boundary audit and sensitivity | `scripts/token_entropy/calculate_token_entropy.py`, `scripts/analysis/audit_token_boundaries.py`, `evaluate_boundary_sensitivity.py` | E1, E3a, E4, GPU | README §2, robustness |
| E6 | Raw-score evaluation (Spearman / AUROC / AURAC, bootstrap, random reference) | `scripts/evaluation/evaluate_auroc.py`, `evaluate_aurac.py`, `scripts/analysis/compare_entropy_estimators.py` | E2, E3a, E5 | [results/baselines](../../results/baselines/semantic_entropy/README.md) |
| E7 | Predictor-state (`p-1`) hidden-state extraction | `scripts/hidden_states/extract_hidden_state_features.py` | E1, E3a, GPU | README §3 |
| E8 | Nested-CV predictive-probe screening (235 configurations) | `scripts/evaluation/evaluate_hidden_state_predictive.py` | E2, E3a, E5, E7 | [results/predictive_probes](../../results/predictive_probes/README.md) |
| E9 | Post-token (`p`) extraction, dispersion and predictive test | `scripts/hidden_states/evaluate_post_token_hidden_state_dispersion.py`, `evaluate_post_token_dispersion_predictive.py` | E1, E3a, GPU | [results/hidden_states/post_token_dispersion](../../results/hidden_states/post_token_dispersion/README.md) |
| E10 | Predictor-state vs post-token dispersion comparison | `scripts/hidden_states/evaluate_predictor_vs_post_token_dispersion.py` | E8, E9 | [results/hidden_states/predictor_vs_post_token](../../results/hidden_states/predictor_vs_post_token/README.md) |
| R1 | Clustering order sensitivity | `scripts/analysis/evaluate_order_sensitivity.py` | E1, DeBERTa-NLI | README robustness |
| R2 | Greedy vs connected-components clustering | `scripts/semantic/compare_clustering_methods.py` | E2 | README robustness |
| R3 | Manual NLI-pair validation | `scripts/analysis/prepare_manual_cluster_validation.py`, `evaluate_manual_cluster_validation.py` | `data/annotations/` | README robustness |

## Offline checks

* `python -m pytest` runs the offline unit tests (no model weights, no downloads).
* `manual_checks/` holds opt-in scripts that load real models/datasets; they are never run by CI.
