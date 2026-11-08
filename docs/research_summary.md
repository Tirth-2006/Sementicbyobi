# Research summary

**Last updated:** 8 November 2026 · **Status:** exploratory results on one 100-question TriviaQA sample with Qwen3-1.7B.

This page summarises the *current* state. The authoritative numbers and tables are in the
[README](../README.md); the earlier (31 October) summary is kept in
[`docs/archive/`](archive/research_summary_2026-10-31.md).

## Question

Can signals computed from ten high-temperature (T = 1.0) responses and the model's hidden states
predict whether a *separate* low-temperature (T = 0.1) answer matches a TriviaQA reference?
One question is one labelled observation (27 correct / 73 incorrect). The target is normalized
exact match, not human-verified truth.

## What has been completed

| Stage | Status | Where to look |
|---|---|---|
| Dual-temperature generation (10 × T=1.0, 1 × T=0.1 per question) | Done | `scripts/data/generate_data.py` |
| NLI clustering and empirical semantic entropy | Done | `scripts/semantic/` |
| Correctness labels (low-temperature target) | Done | `scripts/evaluation/evaluate_low_temp_correctness.py` |
| Teacher-forced token scoring and token predictive entropy | Done | `scripts/token_entropy/` |
| Raw-score evaluation: Spearman, AUROC, AURAC, bootstrap CIs, random-rejection reference | Done | [`results/baselines/`](../results/baselines/semantic_entropy/README.md) |
| Hidden-state extraction (predictor-state `p-1`) and 235-configuration probe screening | Done (exploratory) | [`results/predictive_probes/`](../results/predictive_probes/README.md) |
| Post-token (`p`) vs predictor-state (`p-1`) dispersion as an added feature | Done | [`results/hidden_states/`](../results/hidden_states/predictor_vs_post_token/README.md) |
| Robustness: order sensitivity, connected components, manual NLI review, token-boundary sensitivity | Done | README "Robustness checks" |

## Headline findings (with the caveats that travel with them)

* **Raw scores.** Empirical semantic entropy (AUROC 0.841, AURAC 0.463) and token predictive
  entropy (AUROC 0.852, AURAC 0.538) both rank incorrect answers above correct ones in this sample.
  Token entropy has higher point estimates, but every paired 95% bootstrap interval for the
  token-minus-semantic difference includes zero, so **neither signal is shown to be better**.
* **Random-rejection reference.** Semantic-entropy AURAC exceeds the same-sample random-rejection mean
  (0.2696) by 0.193. This is not an external benchmark.
* **Probes.** Nested-CV logistic-regression probes on hidden states did not clearly beat the entropy
  baselines (best hidden-only AUROC 0.772 vs 0.838 for semantic entropy alone); the screen covered
  116 hidden-state alternatives with no multiplicity correction.
* **Dispersion.** Adding predictor-state dispersion to semantic + token entropy has the best point
  estimates (AUROC 0.839, AURAC 0.515), but paired AUROC and AURAC intervals include zero, so no
  incremental improvement is established.

## What may and may not be claimed

*Supported:* an exploratory study of semantic entropy, token predictive entropy and hidden-state
features for question-level correctness prediction on a 100-question TriviaQA sample with Qwen3-1.7B.

*Not supported:* a general or validated hallucination detector, calibrated uncertainty,
cross-model or cross-dataset generalization, or a fully reproducible release.

See [limitations](limitations.md) and [reproducibility](reproducibility.md).

## Next steps (not yet done)

1. Larger and additional datasets; more model sizes/families.
2. Probability-weighted semantic entropy using saved generation-time log-probabilities.
3. Compare the recorded file hashes against the official Hugging Face release to pin the upstream
   revision (see [reproducibility](reproducibility.md)).
4. Human-labelled evaluation of NLI equivalence and of answer correctness.
