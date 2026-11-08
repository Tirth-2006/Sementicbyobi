# Results

Compact, stable reports. Large generated artifacts (answer JSON, NPZ tensors, model weights) stay
out of Git; see [reproducibility](../docs/reproducibility.md).

| Folder | Contents |
|---|---|
| [`baselines/semantic_entropy/`](baselines/semantic_entropy/README.md) | Raw semantic-entropy metrics, CIs, random-rejection reference, AURAC curve |
| [`baselines/token_entropy/`](baselines/token_entropy/README.md) | Raw token-entropy metrics, CIs, paired comparison with semantic entropy |
| [`predictive_probes/`](predictive_probes/README.md) | Initial 235-configuration nested-CV screening summary |
| [`hidden_states/post_token_dispersion/`](hidden_states/post_token_dispersion/README.md) | Post-token dispersion predictive test (`report.md`, `metrics.csv`) |
| [`hidden_states/predictor_vs_post_token/`](hidden_states/predictor_vs_post_token/README.md) | Predictor-state vs post-token comparison (`report.md`, `metrics.csv`) |
| [`provenance/`](provenance/README.md) | Recorded model/environment fingerprint (`model_provenance.json`) |

All results: one 100-question TriviaQA sample, Qwen3-1.7B, 27 correct / 73 incorrect targets.
Exploratory only. Tables in `baselines/` and `predictive_probes/` are transcribed from the project's
saved experiment summaries as reported in the main README; the `hidden_states/` files are copies of
tracked generated reports.
