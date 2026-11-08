# Predictive-probe screening (nested CV)

235 configurations: 116 hidden-only (29 indices × 4 poolings), 3 entropy baselines, and 116
hidden + both entropies. Out-of-fold probabilities are converted to uncertainty `1 - P(correct)`.

**Protocol:** question-level nested 2-fold stratified CV, shuffled, seed `20261106`; `StandardScaler`
inside the training-fold pipeline; logistic regression (`lbfgs`, `max_iter=1000`),
C ∈ {0.1, 1.0} chosen by inner-fold AUROC; 100 bootstrap resamples (preliminary).

| Configuration | AUROC | AURAC | Spearman | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|
| Semantic entropy only | 0.838153 | 0.479576 | -0.521014 | 0.73 | 0.000000 |
| Token entropy only | 0.802131 | 0.479362 | -0.464677 | 0.73 | 0.000000 |
| Semantic + token entropy | 0.821918 | 0.494675 | -0.495109 | 0.77 | 0.342857 |
| Best hidden-only by AUROC (`hidden_state_28_mean`) | 0.771689 | 0.462826 | -0.417858 | — | — |
| Best hidden + entropy by AUROC (`hidden_state_28_mean_plus_entropy`) | 0.804668 | 0.481866 | -0.468578 | — | — |

Caveats: no multiplicity correction across 116 hidden-state alternatives; "best" must name its metric
(AURAC leaders differ from AUROC leaders). Do not compare these AUROC/AURAC values with the raw-score
tables in `../baselines/`; the protocols differ.

Follow-up on dispersion features: [`../hidden_states/`](../hidden_states/predictor_vs_post_token/README.md).
