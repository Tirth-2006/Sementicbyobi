# Semantic Entropy-Based Hallucination Detection

**Last Updated:** 2026-10-31  
**Project Status:** Experiment 1 completed; detection-performance evaluation pending  
**Model:** Qwen3-1.7B (4-bit quantized)  
**Hardware:** NVIDIA RTX 2050  
**Dataset:** TriviaQA  
**Research Reference:** Farquhar et al., *Detecting Hallucinations in Large Language Models Using Semantic Entropy* (Nature, 2024)

---

# 1. Project Overview

This project investigates whether semantic uncertainty can help identify hallucinations in Large Language Models (LLMs).

The central research question is:

> Can semantic entropy serve as a useful signal for identifying incorrect or potentially hallucinated LLM responses?

The project is inspired by the semantic entropy framework proposed by Farquhar et al. (2024).

The implementation uses Qwen3-1.7B, TriviaQA, and an NLI-based semantic clustering approach.

**Important methodological distinction:** The current implementation is an adaptation of the original semantic entropy approach, not an exact reproduction of the paper. In particular, the current entropy calculation uses empirical semantic-cluster frequencies rather than the original probability-weighted semantic sequence probabilities.

The research is being developed incrementally, with statistical validation before making claims about detection performance.

---

# 2. Development Environment

| Component | Configuration |
|---|---|
| Operating System | Windows |
| Python | 3.13 |
| Virtual Environment | hallucination |
| Model | Qwen3-1.7B |
| Quantization | 4-bit |
| GPU | NVIDIA RTX 2050 |
| RAM | 16 GB |
| Framework | Transformers |
| NLI Model | cross-encoder/nli-deberta-v3-small |
| Dataset | TriviaQA |

Project directory:

```text
C:\Users\TIRTH\OneDrive\Desktop\Hallucination
```

Virtual environment:

```text
C:\Users\TIRTH\venvs\hallucination
```

Activation:

```powershell
& "C:\Users\TIRTH\venvs\hallucination\Scripts\Activate.ps1"
```

---

# 3. Implementation Progress

## Step 1: Model Setup and Verification

**Status: Completed**

Qwen3-1.7B was downloaded and configured locally using 4-bit quantization.

The model was successfully loaded and tested.

Initial verification included:

- Successful model loading.
- Text generation.
- Token entropy extraction.
- Hidden-state extraction.
- GPU memory inspection.

The model returned 29 hidden-state layers, with the final hidden-state tensor having shape:

```text
(1, 33, 2048)
```

The pooled answer representation had shape:

```text
(2048,)
```

This established that the local environment could support the initial experiments.

Note: Local model execution is infrastructure setup, not a completed research contribution.

## Step 2: Data Generation Pipeline

**Status: Completed**

A modular data-generation pipeline was implemented.

### Main components

| File | Responsibility |
|---|---|
| `src/config.py` | Model, generation and dataset configuration |
| `src/data_loader.py` | TriviaQA loading |
| `src/prompt_builder.py` | Few-shot prompt construction |
| `src/generator.py` | Qwen3 generation wrapper |
| `generate_data.py` | Pipeline orchestration |

### Generation configuration

| Parameter | Value |
|---|---|
| Model | Qwen3-1.7B |
| Quantization | 4-bit |
| Few-shot examples | 5 |
| Low temperature | 0.1 |
| High temperature | 1.0 |
| High-temperature generations | 10 per question |
| Top-p | 0.8 |
| Top-k | 20 |
| Maximum new tokens | 64 |
| Thinking mode | Disabled |

### Dual-temperature generation

Two generation settings were used:

**Low-temperature generation (0.1)**

Used as a relatively deterministic answer-quality baseline.

**High-temperature generation (1.0)**

Used to generate diverse responses for semantic uncertainty estimation.

### Main dataset

```text
data/generated/trivia_qa_qwen3_20261024_162539.json
```

Dataset statistics:

- 100 TriviaQA questions.
- 10 high-temperature responses per question.
- 1,000 high-temperature responses in total.
- All 100 questions processed successfully.
- No per-question generation failures.

Generation took approximately 81 minutes.

The generated dataset also contains the associated question information, reference answers and generation metadata.

---

# 4. Semantic Clustering

**Status: Completed**

The next stage was grouping generated answers according to their semantic meaning.

The NLI model used was:

```text
cross-encoder/nli-deberta-v3-small
```

### Clustering approach

The implemented method is:

**Greedy complete-linkage clustering based on bidirectional DeBERTa entailment.**

The procedure:

1. Take a generated answer.
2. Compare it with existing cluster members using NLI.
3. Evaluate entailment in both directions.
4. Require mutual entailment.
5. Require the entailment score to be at least 0.5.
6. Add the answer to a cluster only if it satisfies the condition against every existing member.
7. Otherwise, create a new cluster.

The implementation uses greedy first-compatible-cluster assignment.

### Important implementation details

- Question-conditioned NLI comparisons.
- Bidirectional entailment.
- Entailment threshold: 0.5.
- Complete-linkage membership requirement.
- Greedy cluster assignment.

For 10 responses, there are 45 unordered answer pairs, corresponding to 90 directional NLI evaluations when both directions are checked.

### Limitations

- Greedy clustering is order-dependent.
- The threshold of 0.5 has not been calibrated against a human-labelled semantic-equivalence dataset.
- NLI entailment is an imperfect proxy for semantic equivalence.
- The current implementation is an adaptation, not an exact reproduction of the clustering procedure in Farquhar et al.

---

# 5. Empirical Semantic Entropy Calculation

**Status: Completed**

After clustering, semantic entropy was calculated for each question.

The underlying idea is that a model producing semantically different answers has greater semantic uncertainty.

Let:

- \(N\) = total number of generated answers.
- \(C_i\) = the \(i\)-th semantic cluster.
- \(p_i\) = empirical frequency of cluster \(i\).

The cluster probability is estimated as:

\[
p_i = \frac{|C_i|}{N}
\]

Entropy is calculated using Shannon entropy:

\[
H_{\text{empirical}} =
-\sum_i p_i \ln(p_i)
\]

Natural logarithms are used, so entropy is measured in nats.

### Interpretation

- One semantic cluster: entropy is zero.
- Multiple similarly sized clusters: entropy increases.
- More semantic disagreement: generally higher entropy.

### Important distinction from the original paper

The current implementation uses empirical cluster frequencies:

\[
p(C_i) = \frac{|C_i|}{N}
\]

The probability-weighted semantic entropy framework instead considers semantic class probabilities:

\[
P(C_i \mid x) =
\sum_{y \in C_i} P(y \mid x)
\]

The latter incorporates model sequence probabilities rather than relying only on sampled cluster frequencies.

Therefore, the current implementation should be described as **empirical semantic-cluster entropy**, not as an exact implementation of Farquhar et al.'s probability-weighted estimator.

### Output

```text
data/generated/trivia_qa_qwen3_20261024_162539_clusters.json
```

The entropy evaluation script is:

```text
evaluate_semantic_entropy.py
```

The entropy value is stored under:

```text
questions.<question_id>.entropy_nats
```

---

# 6. Answer Quality Evaluation

**Status: Completed**

The generated responses were evaluated against the available TriviaQA reference answers.

Two metrics were calculated.

## 6.1 Exact Match

The answer is considered correct when its normalized text exactly matches a reference answer.

Normalization includes:

- Lowercasing.
- Removing ASCII punctuation.
- Removing articles.
- Collapsing whitespace.

Results:

| Metric | Result |
|---|---:|
| Total questions | 100 |
| Total responses | 1,000 |
| Exact matches | 239 |
| Exact-match rate | 23.9% |

## 6.2 Token F1

Token F1 measures lexical overlap between the generated answer and reference answers.

It uses token-level precision and recall, with multiset token matching.

Results:

| Metric | Result |
|---|---:|
| Mean token F1 | 0.310384 |

### Important limitations

Exact match and token F1 are reference-based evaluation proxies.

They do not guarantee factual correctness.

An answer can be factually correct but receive a low score because its wording differs from the available references.

Similarly, lexical overlap does not necessarily establish factual equivalence.

### Output

```text
data/generated/trivia_qa_qwen3_20261024_162539_correctness_f1.json
```

The evaluation results were independently checked:

- Zero exact-match mismatches.
- Zero token F1 mismatches.
- Zero correctness/exact-match consistency errors.
- Zero per-question total-count errors.

---

# 7. Merged Analysis Dataset

**Status: Completed**

The generation, semantic entropy and answer-quality evaluation outputs were merged into one question-level CSV.

Output:

```text
data/generated/trivia_qa_qwen3_20261024_162539_entropy_accuracy_f1.csv
```

The CSV contains 100 rows, one per question.

### Columns

| Column | Description |
|---|---|
| `question_id` | TriviaQA question identifier |
| `semantic_entropy_nats` | Empirical semantic entropy |
| `accuracy` | Question-level exact-match accuracy |
| `mean_token_f1` | Mean token F1 across generated responses |

### Validation

- All three source datasets contained 100 questions.
- Question IDs matched across all three datasets.
- No missing or duplicate question IDs.
- Entropy values matched the original evaluation output.
- CSV was re-imported and row count verified.

No statistical analysis was performed during the initial merge.

---

# 8. Experiment 1: Spearman Correlation Analysis

**Status: Completed**

## 8.1 Research Question

The first statistical experiment investigated:

> Is higher semantic entropy associated with lower answer quality?

Two relationships were examined:

1. Semantic entropy versus question-level exact-match accuracy.
2. Semantic entropy versus mean token F1.

The unit of analysis was the question.

Therefore, the experiment used 100 observations rather than treating all 1,000 generated responses as independent observations.

## 8.2 Why Spearman Correlation?

Spearman's rank correlation measures the strength and direction of a monotonic relationship between two variables.

It was selected because we wanted to determine whether questions with higher semantic entropy generally had lower answer quality without assuming a linear relationship.

The calculation uses average ranks for tied observations.

\[
\rho_s =
\operatorname{Pearson}(R_X,R_Y)
\]

Where:

- \(R_X\) represents the ranks of semantic entropy.
- \(R_Y\) represents the ranks of accuracy or mean F1.

A negative correlation indicates that higher entropy tends to accompany lower answer quality.

## 8.3 Statistical Methodology

The analysis used:

| Method | Configuration |
|---|---|
| Correlation | Spearman rank correlation |
| Tie handling | Average ranks |
| Permutation test | Two-sided Monte Carlo |
| Permutation resamples | 100,000 |
| P-value correction | Plus-one correction |
| Confidence interval | Paired percentile bootstrap |
| Bootstrap resamples | 20,000 |
| Confidence level | 95% |
| Random seed | 20261031 |

### Permutation test

The two-sided permutation test evaluated whether the observed association was unusually extreme under the null hypothesis of no association.

The accuracy or F1 ranks were randomly shuffled while the entropy ranks remained fixed.

The procedure was repeated 100,000 times.

The p-value used the plus-one correction:

\[
p = \frac{b+1}{B+1}
\]

Where:

- \(b\) = number of permutation results at least as extreme as the observed result.
- \(B\) = total number of permutations.

### Bootstrap confidence intervals

Paired question-level bootstrap resampling was used.

Each bootstrap sample selected 100 question indices with replacement, preserving the pairing between entropy and answer quality.

20,000 resamples were used to estimate the 95% percentile confidence interval.

---

## 8.4 Results

### Analysis A: Semantic Entropy vs. Question-Level Accuracy

\[
\boxed{\rho_s = -0.56783}
\]

| Statistic | Result |
|---|---:|
| Spearman correlation | -0.5678306628 |
| 95% bootstrap CI | [-0.7053493, -0.4116450] |
| Permutation p-value | 0.0000099999 |
| Bootstrap resamples requested | 20,000 |
| Valid bootstrap resamples | 20,000 |
| Invalid bootstrap resamples | 0 |

**Interpretation:**

A moderately strong negative association was observed between empirical semantic entropy and question-level exact-match accuracy.

Questions with higher semantic entropy tended to have lower accuracy.

### Analysis B: Semantic Entropy vs. Mean Token F1

\[
\boxed{\rho_s = -0.54956}
\]

| Statistic | Result |
|---|---:|
| Spearman correlation | -0.5495623897 |
| 95% bootstrap CI | [-0.6988321, -0.3742840] |
| Permutation p-value | 0.0000099999 |
| Bootstrap resamples requested | 20,000 |
| Valid bootstrap resamples | 20,000 |
| Invalid bootstrap resamples | 0 |

**Interpretation:**

A moderately strong negative association was also observed between empirical semantic entropy and mean token F1.

Questions with higher semantic entropy tended to have lower lexical overlap with reference answers.

---

## 8.5 Overall Findings

Both statistical analyses produced negative Spearman correlations.

| Relationship | Spearman ρ | 95% CI | Permutation p |
|---|---:|---|---:|
| Entropy vs. accuracy | -0.568 | [-0.705, -0.412] | 0.00001 |
| Entropy vs. mean F1 | -0.550 | [-0.699, -0.374] | 0.00001 |

The results provide preliminary evidence that empirical semantic entropy is associated with answer quality in this 100-question TriviaQA experiment using Qwen3-1.7B.

Both confidence intervals lie below zero.

The permutation p-value of approximately \(10^{-5}\) is the smallest value possible with 100,000 permutations using the plus-one correction. It should not be interpreted as the exact underlying probability.

### What these results establish

- Semantic entropy has a negative rank association with exact-match accuracy in this sample.
- Semantic entropy has a negative rank association with mean token F1 in this sample.
- Both observed associations are statistically significant under the implemented permutation tests.

### What these results do not establish

- That semantic entropy can reliably classify individual hallucinations.
- That semantic entropy generalizes to other models or datasets.
- That semantic entropy alone is sufficient for hallucination detection.
- That correlation establishes causation.
- That reference-based accuracy and F1 perfectly represent factual correctness.

---

# 9. Statistical Analysis Script and Validation

**Status: Completed**

Main script:

```text
analyze_entropy_correlation.py
```

Saved results:

```text
data/generated/trivia_qa_qwen3_20261024_162539_spearman_results.json
```

### Script functionality

1. Loads the merged CSV.
2. Validates the input data.
3. Computes Spearman correlation using average ranks.
4. Runs two-sided Monte Carlo permutation tests.
5. Calculates paired bootstrap confidence intervals.
6. Saves results as JSON.

### Input validation

The updated script checks:

- Required CSV columns.
- Unique column headers.
- Exactly 100 question rows.
- Nonblank question IDs.
- Unique question IDs.
- Numeric and finite analysis values.
- Non-negative entropy.
- Accuracy within [0, 1].
- Mean token F1 within [0, 1].

### Bootstrap validation

Both analyses reported:

| Measure | Accuracy | Mean F1 |
|---|---:|---:|
| Requested | 20,000 | 20,000 |
| Valid | 20,000 | 20,000 |
| Invalid | 0 | 0 |

The updated script completed successfully.

### Remaining statistical limitations

1. Percentile bootstrap confidence intervals may have imperfect coverage with heavily tied or discrete variables.
2. The permutation test assumes exchangeability of question-level observations under the null hypothesis.
3. Dependencies between questions, if present, may affect that assumption.
4. Statistical significance does not establish practical detection performance.

---

# 10. Current Research Limitations

The following limitations should be considered before making broader claims.

## 10.1 Model and Dataset Scope

The current experiment uses:

- One model: Qwen3-1.7B.
- One dataset: TriviaQA.
- 100 questions.

The findings need validation on additional questions, datasets and models.

## 10.2 Clustering Limitations

The current clustering approach is greedy and order-dependent.

The entailment threshold of 0.5 has not been calibrated using human-labelled semantic equivalence judgments.

## 10.3 Entropy Estimation

The current implementation uses empirical cluster-frequency entropy.

It does not yet implement the full probability-weighted semantic entropy estimator described by Farquhar et al.

## 10.4 Evaluation Limitations

Exact match and token F1 are imperfect proxies for factual correctness.

Human evaluation or stronger factuality verification may be necessary for more reliable conclusions.

## 10.5 Statistical Limitations

The current correlation results are exploratory.

The bootstrap intervals and permutation tests depend on assumptions that should be considered when interpreting the results.

---

# 11. Next Research Milestone: Detection Performance Evaluation

**Status: Not yet implemented**

The next experiment will investigate whether semantic entropy can distinguish correct and incorrect responses.

This is different from the completed correlation experiment.

The correlation experiment asked:

> Are higher entropy values associated with lower answer quality?

The detection experiment will ask:

> Can entropy distinguish correct answers from incorrect answers?

### Planned evaluation metrics

**AUROC (Area Under the Receiver Operating Characteristic Curve)**

Measures how well entropy ranks incorrect responses above correct responses across classification thresholds.

**AUPRC (Area Under the Precision-Recall Curve)**

Measures the precision-recall trade-off and is particularly relevant when correct and incorrect responses are imbalanced.

**Precision**

The proportion of responses flagged as incorrect that are actually incorrect.

**Recall**

The proportion of incorrect responses successfully identified.

**F1-score**

The harmonic mean of precision and recall at a chosen threshold.

### Important methodological decision

The current entropy values are question-level measurements, while correctness labels exist at the response level.

Before calculating AUROC or AUPRC, the evaluation must explicitly define how question-level entropy is assigned to individual responses.

One possible approach is to assign each response the entropy of its corresponding question. This would test whether question-level uncertainty is informative about individual response correctness.

However, the interpretation and dependence structure must be considered carefully because all ten responses from a question share the same entropy value.

The evaluation should avoid treating these repeated entropy values as independent question-level observations.

### Planned steps

1. Inspect the existing generation, clustering and correctness JSON schemas.
2. Define the response-level evaluation table.
3. Join response correctness labels with question-level entropy.
4. Verify all joins and sample counts.
5. Calculate AUROC and AUPRC.
6. Evaluate precision and recall at selected thresholds.
7. Analyze performance and limitations.

No new answer generation or clustering is required for this initial evaluation.

---

# 12. References

1. Farquhar et al. (2024). *Detecting Hallucinations in Large Language Models Using Semantic Entropy*. Nature.
   - https://www.nature.com/articles/s41586-024-07421-0
   - https://arxiv.org/abs/2302.09664

2. Official semantic uncertainty implementation:
   - https://github.com/jlko/semantic_uncertainty

3. Qwen3 model documentation and generation guidelines.

4. TriviaQA dataset documentation.

---

# 13. Current Project Status

| Stage | Status |
|---|---|
| Local Qwen3 setup | Completed |
| Data generation pipeline | Completed |
| 100-question dataset generation | Completed |
| NLI-based semantic clustering | Completed |
| Empirical semantic entropy | Completed |
| Exact-match evaluation | Completed |
| Token F1 evaluation | Completed |
| Merged analysis CSV | Completed |
| Spearman correlation analysis | Completed |
| Permutation significance testing | Completed |
| Bootstrap confidence intervals | Completed |
| Statistical input validation | Completed |
| AUROC/AUPRC evaluation | Pending |
| Robustness experiments | Pending |
| Additional dataset/model validation | Pending |

---

# 14. Important Research Position

The current findings support an initial relationship between empirical semantic entropy and reference-based answer quality.

However, the project has not yet demonstrated a reliable hallucination detection system.

The next stage is to evaluate whether semantic entropy provides useful discrimination between correct and incorrect responses.

The research should maintain a clear distinction between:

- Semantic uncertainty.
- Factual correctness.
- Hallucination detection.
- Probability-weighted semantic entropy.
- Empirical semantic-cluster entropy.

The objective is not simply to build an engineering pipeline, but to investigate the predictive usefulness, limitations and generalizability of semantic uncertainty as a signal for hallucination detection.

**Current milestone: Experiment 1 completed. Detection-performance evaluation is the next research objective.**
