# Token predictive entropy (raw score)

Mean of the ten answer-level mean token entropies (raw full-vocabulary softmax at causal predictor
positions, teacher-forced, EOS not scored).

| Metric | Point estimate | 95% bootstrap interval |
|---|---:|---:|
| Spearman | -0.541147 | [-0.677502, -0.372187] |
| AUROC | 0.851852 | [0.749243, 0.936012] |
| AURAC | 0.537749 | [0.402956, 0.653437] |

## Paired difference, token minus semantic entropy

Same 20,000 question-level resamples (seed `20261031`).

| Metric | Difference | 95% interval |
|---|---:|---:|
| Spearman | -0.012948 | [-0.109464, 0.085115] |
| AUROC | +0.010401 | [-0.054771, 0.074568] |
| AURAC | +0.074870 | [-0.002527, 0.141731] |

All three intervals include zero: token entropy has higher point estimates, but **superiority over
semantic entropy is not established**.

Caveat: 347/1,000 answers contain a prompt/answer boundary-crossing token (retained in the primary
calculation; a sensitivity variant correlates at Spearman 0.706 with it).
