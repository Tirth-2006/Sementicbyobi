# Methodology and prerequisites

The full description of methods (equations, token alignment, hidden-state pooling, dispersion,
probes, AURAC convention) is in the [README](../../README.md#methods). This page records the
practical prerequisites and the design decisions that matter for correctness. The original
October quick-start is archived at
[`docs/archive/quick_start_2026-10.md`](../archive/quick_start_2026-10.md).

## Prerequisites

* **Python ≥ 3.10** (`pyproject.toml`: `requires-python = ">=3.10"`).
* **Offline unit tests** need only `requirements-test.txt` (CPU; torch optional).
* **Model-backed stages** need a CUDA-capable PyTorch build plus `requirements.txt`
  (transformers, accelerate, bitsandbytes, datasets).
* **Local Qwen3-1.7B checkpoint.** Point every model-backed script at it with an environment
  variable (there is no built-in default path):

  ```bash
  export UNCERTAINTY_MODEL_DIR=/path/to/Qwen3-1.7B      # Linux / macOS
  $env:UNCERTAINTY_MODEL_DIR = "D:\models\Qwen3-1.7B"   # Windows PowerShell
  ```

  The directory must contain `config.json`; scripts load it with `local_files_only=True`.
* **NLI model** `cross-encoder/nli-deberta-v3-small` is downloaded by Hugging Face on first use.
* Install the package so scripts import `uncertainty` without path tricks:
  `python -m pip install -e .`

## Design decisions

1. **Cross-temperature, question-level design.** Features come from ten T=1.0 samples; the label is
   a separate T=0.1 answer. The two correctness evaluators are different tools:
   `evaluate_low_temp_correctness.py` (target label) and `evaluate_sampled_correctness.py`
   (diagnostic labels for the samples).
2. **Empirical semantic entropy.** Shannon entropy over cluster frequencies from greedy
   first-compatible complete-linkage clustering with bidirectional NLI (threshold 0.5). It is
   order-dependent and is *not* the paper's probability-weighted estimator.
3. **Token entropy** uses raw full-vocabulary softmax at causal predictor positions (`p-1`) for
   answer tokens, selected by character-offset overlap (boundary-crossing tokens included).
4. **Hidden states.** Predictor-state (`p-1`) and post-token (`p`) vectors are different
   representations. Dispersion is mean Euclidean distance to the centroid, not variance.
5. **Evaluation protocols are kept separate.** Direct raw-score ranking and out-of-fold probe
   probabilities must not be mixed. AURAC rejects by descending uncertainty, breaks ties by
   ascending `question_id`, and integrates retained accuracy over coverage with the trapezoidal rule.
6. **Generation protocol.** Prompts are raw few-shot text (`Question: ... Answer:`); no chat template
   is applied, so Qwen3's thinking toggle is not used (`enable_thinking` is recorded as `false` and
   `true` is rejected). Both the T = 0.1 and the T = 1.0 generations are sampled (`do_sample=True`,
   top-p 0.8, top-k 20, at most 64 new tokens). `do_sample=False` or a temperature of 0 would give
   greedy decoding; that was not used.
7. **Code layout.** Reusable logic lives in `src/uncertainty/`; files in `scripts/` are thin
   command-line entry points. The package never imports from `scripts/`.

## Frozen-artifact assumption

Scripts use constants such as `data/generated/trivia_qa_qwen3_20261024_162539*.json`. They
reproduce the reported analysis only when those frozen artifacts exist locally. See
[reproducibility](../reproducibility.md).
