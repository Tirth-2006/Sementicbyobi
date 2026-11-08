# Limitations

The results are **exploratory**. They describe associations in one small sample and do not establish
a deployable hallucination detector.

## Data and target

* One frozen 100-question TriviaQA slice with 27 correct and 73 incorrect low-temperature labels.
  Estimates and model selection are unstable at this size.
* The target is normalized reference-based exact match, not human-verified factual truth.
  Valid paraphrases can be labelled incorrect.
* Only one model family and size (Qwen3-1.7B, 4-bit NF4) was evaluated. No other dataset, model or
  task has been tested, so no generalization claim is supported.

## Cross-temperature design

* Features come from ten T=1.0 responses; the target belongs to a *separate* T=0.1 response.
  The result is a cross-temperature, question-level association, not a per-answer correctness test.

## Semantic entropy

* The entropy is **empirical cluster-frequency entropy**, not the original paper's
  probability-weighted semantic entropy. Generation-time token log-probabilities and the filtered
  (top-k/top-p) sampling distribution were not saved, so original sequence probabilities cannot be
  reconstructed.
* Clustering is greedy first-compatible complete linkage and therefore order-dependent
  (answer-order permutation changed partitions for 7 questions).
* The NLI entailment threshold (0.5) is a chosen heuristic, not calibrated on a large
  human-labelled set. The manual review (44 pairs, 37 questions, one annotator, model-selected pairs)
  is not a definitive benchmark.

## Token entropy and hidden states

* Scores and hidden states are **reconstructed** from saved answers and prompts via teacher forcing,
  not captured during generation.
* Token entropy uses the raw full-vocabulary softmax, not the sampling distribution. 347 of 1,000
  answers contain a token crossing the prompt/answer boundary; a sensitivity variant correlated at
  Spearman 0.706 with the primary values.
* Predictor-state (`p-1`) and post-token (`p`) vectors are different representations. Hidden-state
  vectors are not calibrated uncertainty scores.

## Statistics

* Bootstrap intervals are question-level. Paired comparisons of raw entropies and of probe variants
  include zero, so **no statistical superiority** is established between token and semantic entropy,
  or for the dispersion features.
* The 235-configuration probe screen (116 hidden-state alternatives) has no multiplicity correction
  and uses only 100 bootstrap resamples; "best configuration" must name its metric.
* Probe-comparison bootstraps resample fixed out-of-fold predictions; models are not retrained inside
  each resample. Nested 2-fold CV on 100 questions is itself noisy.
* The random-rejection AURAC baseline is a same-sample reference, not an external benchmark.
* Threshold-dependent metrics (recall, F1 at 0.5) are exploratory.

## Reproducibility

* The exact Qwen checkpoint revision/hash was not recorded (see
  [reproducibility](reproducibility.md) for how to record it).
* Full generated-answer JSON and NPZ tensors are not committed; a fresh clone cannot re-run every
  stage without them.
