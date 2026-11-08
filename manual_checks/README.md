# Manual integration checks

These scripts load real models or datasets (Qwen3-1.7B, DeBERTa-NLI, TriviaQA)
and are **not** part of the automated test suite. They are deliberately named
`check_*.py` so that pytest never collects or imports them, and CI never runs
them.

| Script | Needs |
|---|---|
| `check_nli.py` | Downloads `cross-encoder/nli-deberta-v3-small` |
| `check_qwen.py` | CUDA GPU, `UNCERTAINTY_MODEL_DIR` |
| `check_pipeline.py` | TriviaQA download, GPU, `UNCERTAINTY_MODEL_DIR` |

Run from the repository root, e.g.:

```bash
export UNCERTAINTY_MODEL_DIR=/path/to/Qwen3-1.7B
python manual_checks/check_qwen.py
```

The offline unit tests live in `tests/` and run with `python -m pytest`.
