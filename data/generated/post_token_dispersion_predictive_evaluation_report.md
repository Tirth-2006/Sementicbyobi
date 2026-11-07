# Post-Token Dispersion Predictive Evaluation

| Configuration | AUROC | AURAC | Spearman | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|
| semantic_entropy | 0.838153 | 0.479576 | -0.521014 | 0.730000 | 0.000000 |
| token_entropy | 0.802131 | 0.479362 | -0.464677 | 0.730000 | 0.000000 |
| semantic_plus_token_entropy | 0.821918 | 0.494675 | -0.495109 | 0.770000 | 0.342857 |
| semantic_plus_token_entropy_plus_post_token_dispersion | 0.832065 | 0.509229 | -0.510715 | 0.770000 | 0.488889 |

## Paired difference

Difference is extended configuration minus semantic + token entropy.

```json
{
  "comparison": "semantic_plus_token_entropy_plus_post_token_dispersion minus semantic_plus_token_entropy",
  "resamples": 20000,
  "random_seed": 20261106,
  "confidence_intervals_95": {
    "auroc_incorrect": [
      -0.03353181879182064,
      0.058133980801806866
    ],
    "aurac": [
      -0.027416714605910785,
      0.060537069199043426
    ],
    "spearman_uncertainty_vs_correct": [
      -0.08843133535746817,
      0.05073514223931613
    ],
    "accuracy_at_0_5": [
      -0.07000000000000006,
      0.07000000000000006
    ],
    "precision_at_0_5": [
      -0.4285714285714286,
      0.16666666666666663
    ],
    "recall_at_0_5": [
      -0.03125,
      0.3940909090908959
    ],
    "f1_at_0_5": [
      -0.08571428571428574,
      0.3806528417818734
    ]
  },
  "valid_samples": {
    "auroc_incorrect": 20000,
    "aurac": 20000,
    "spearman_uncertainty_vs_correct": 20000,
    "accuracy_at_0_5": 20000,
    "precision_at_0_5": 20000,
    "recall_at_0_5": 20000,
    "f1_at_0_5": 20000
  },
  "undefined_samples": {
    "auroc_incorrect": 0,
    "aurac": 0,
    "spearman_uncertainty_vs_correct": 0,
    "accuracy_at_0_5": 0,
    "precision_at_0_5": 0,
    "recall_at_0_5": 0,
    "f1_at_0_5": 0
  }
}
```

All preprocessing was fitted within training folds. The four configurations used identical outer and inner split seeds.
