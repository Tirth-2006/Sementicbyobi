# Data

Small, tracked inputs only. Large artifacts are produced locally and are not committed.

| Path | Tracked | Contents |
|---|---|---|
| `examples/sample_entry.json` | yes | One generation record (`bb_6487`) showing the schema |
| `annotations/` | yes | Manual NLI-pair review used to audit semantic clustering |
| `generated/` | partly | Compact reports/metrics kept in Git; everything else is git-ignored |

## Generation record schema

```text
{ "question_id": str,
  "entry": {
    "question": str, "context": null,
    "reference_answers": [str, ...],
    "low_temp_generation":  { "response": str, "temperature": 0.1 },
    "high_temp_generations": { "temperature": 1.0, "num_generations": int,
                               "responses": [ { "index": int, "response": str }, ... ] } } }
```

The full corpus (`generations` keyed by question ID, plus run metadata and the few-shot prompt) is
written by `scripts/data/generate_data.py` to `data/generated/` and is not committed.
See [`docs/reproducibility.md`](../docs/reproducibility.md) for the list of frozen artifacts.
Note that `sample_entry.json` has 3 high-temperature responses (a small example), whereas the
reported experiment used 10 per question.

`data/generated/` is ignored by default (new files there will not be tracked unless allow-listed in
`.gitignore` or force-added). Curated copies of small reports live in [`results/`](../results/README.md).
