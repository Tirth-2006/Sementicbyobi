# Provenance

`model_provenance.json` is **not yet present**. Generate it on the machine holding the checkpoint
used for the reported results:

```bash
export UNCERTAINTY_MODEL_DIR=/path/to/Qwen3-1.7B
python scripts/utils/record_model_provenance.py
```

It stores SHA-256 hashes of config, tokenizer and weight files and key package versions (no absolute
paths). Until then, the exact checkpoint revision/hash remains unrecorded.
