# Reproducibility guide

What can be run from a clean clone, what needs external artifacts, and how to record the
details that were not captured originally.

## 1. Offline checks (any machine, no GPU, no downloads)

```bash
python -m pip install -r requirements-test.txt
python -m pip install -e .
python -m pytest
```

CI runs exactly this (plus a CPU build of PyTorch so the torch-specific tests run) and also builds a
wheel, installs it in a clean environment and imports the package from outside the repository.

## 2. Model-backed stages

Require a CUDA-capable PyTorch, `requirements.txt`, and a local Qwen3-1.7B directory:

```bash
export UNCERTAINTY_MODEL_DIR=/path/to/Qwen3-1.7B
```

All model-loading scripts go through `uncertainty.models`, which raises a clear error when the
variable is unset or the directory lacks `config.json`.

## 3. Frozen artifacts (not committed)

The reported analysis reads these files from `data/generated/` (prefix
`trivia_qa_qwen3_20261024_162539`):

| Artifact | Produced by | Notes |
|---|---|---|
| `<prefix>.json` | `scripts/data/generate_data.py` | 100 questions × (1 low-T + 10 high-T) answers; generation settings are stored in its metadata |
| `<prefix>_clusters.json` | `scripts/semantic/evaluate_semantic_entropy.py` | NLI clusters and entropy |
| `<prefix>_low_temp_correctness.json` | `scripts/evaluation/evaluate_low_temp_correctness.py` | binary target |
| `<prefix>_teacher_forced_scores.json` | `scripts/token_entropy/score_teacher_forced.py` | |
| `<prefix>_token_entropy*.{json,csv}` | `scripts/token_entropy/calculate_token_entropy.py` | |
| `<prefix>_hidden_state*.npz` | `scripts/hidden_states/extract_hidden_state_features.py` | large float16/float32 tensors |

Large JSON/NPZ/model files are intentionally kept out of default Git history. If you publish them
(e.g. a GitHub Release or Hugging Face dataset), record their SHA-256 here.
Small curated reports are tracked under [`results/`](../results/README.md).

## 4. Model provenance

SHA-256 hashes of the local Qwen3-1.7B checkpoint and tokenizer files, plus Python and package
versions, were recorded on 8 November 2026 in
[`results/provenance/model_provenance.json`](../results/provenance/model_provenance.json).

Limits of this record: the upstream Hugging Face commit revision was **not** captured when the
experiments ran (`hf_snapshot_revision` is `null`), the hashes assume the files on disk were not
modified after the experiments, and the package versions describe the environment on the recording
date. To regenerate it for another copy of the model:

```bash
export UNCERTAINTY_MODEL_DIR=/path/to/Qwen3-1.7B
python scripts/utils/record_model_provenance.py
git add results/provenance/model_provenance.json
```

This writes SHA-256 hashes of the config, tokenizer and weight files plus Python and package
versions (no absolute paths) to `results/provenance/model_provenance.json`.

## 5. Environments

| Purpose | File |
|---|---|
| Offline tests / CI | `requirements-test.txt` (CPU, numpy ≥ 2.0) |
| Full model experiments | `requirements.txt` (adds torch, transformers, bitsandbytes, …) |
| Package metadata | `pyproject.toml` (`pip install -e .[models]` for model extras) |

`requirements*.txt` use lower bounds. For an exact record of a working environment, the provenance
script stores installed versions of the key packages.

## 6. End-to-end commands

Run order (each needs the previous stage's artifact): E1 → E2 → E3a → E4 → E5 → E6/E7 → E8 → E9 → E10,
see the [experiment index](experiments/experiment_index.md). Full one-command reproduction is **not**
claimed, because the generated corpus and tensors are not committed and have not been re-run from
scratch in a clean environment.
