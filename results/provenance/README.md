# Provenance

[`model_provenance.json`](model_provenance.json) records SHA-256 hashes and sizes of the local
Qwen3-1.7B config, tokenizer and weight files, plus Python and key package versions. It was generated
on 8 November 2026 on the machine that holds the checkpoint used for the reported results.

Limits:

* `hf_snapshot_revision` is `null`: the upstream Hugging Face commit was not captured when the
  experiments ran. The hashes identify the files on disk, assuming they were not modified afterwards.
* The package versions describe the environment on the recording date, not necessarily every
  historical run.
* Absolute paths are deliberately not stored.

To regenerate for another copy of the model:

```bash
export UNCERTAINTY_MODEL_DIR=/path/to/Qwen3-1.7B
python scripts/utils/record_model_provenance.py
```
