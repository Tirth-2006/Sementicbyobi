# Predictor-State vs Post-Token Dispersion

| Configuration | AUROC | AURAC | Spearman | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| semantic_plus_token_entropy | 0.821918 | 0.494675 | -0.495109 | 0.770000 | 0.750000 | 0.222222 | 0.342857 |
| semantic_plus_token_entropy_plus_predictor_dispersion | 0.839168 | 0.515165 | -0.521639 | 0.790000 | 0.650000 | 0.481481 | 0.553191 |
| semantic_plus_token_entropy_plus_post_token_dispersion | 0.832065 | 0.509229 | -0.510715 | 0.770000 | 0.611111 | 0.407407 | 0.488889 |

## Paired bootstrap intervals

```json
{
  "resamples": 20000,
  "random_seed": 20261106,
  "comparisons": {
    "B_minus_A": {
      "difference": "B minus A",
      "confidence_intervals_95": {
        "auroc_incorrect": [
          -0.028170604655601554,
          0.06949831436132811
        ],
        "aurac": [
          -0.02093701119861045,
          0.06820421142621097
        ],
        "spearman_uncertainty_vs_correct": [
          -0.10586503979258438,
          0.04265719735105537
        ],
        "accuracy_at_0_5": [
          -0.05999999999999994,
          0.09999999999999998
        ],
        "precision_at_0_5": [
          -0.38888888888888884,
          0.20772946859903385
        ],
        "recall_at_0_5": [
          0.038461538461538436,
          0.48
        ],
        "f1_at_0_5": [
          -0.030333670374115274,
          0.4507980507555285
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
    },
    "C_minus_A": {
      "difference": "C minus A",
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
    },
    "C_minus_B": {
      "difference": "C minus B",
      "confidence_intervals_95": {
        "auroc_incorrect": [
          -0.023391812865496967,
          0.007905138339920903
        ],
        "aurac": [
          -0.020819020699156407,
          0.004435506066339868
        ],
        "spearman_uncertainty_vs_correct": [
          -0.011986356819171123,
          0.03588795261787104
        ],
        "accuracy_at_0_5": [
          -0.050000000000000044,
          0.0
        ],
        "precision_at_0_5": [
          -0.1127819548872181,
          0.0
        ],
        "recall_at_0_5": [
          -0.19047619047619047,
          0.0
        ],
        "f1_at_0_5": [
          -0.16666666666666669,
          0.0
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
  }
}
```
