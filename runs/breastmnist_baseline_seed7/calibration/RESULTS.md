# Locked-Checkpoint Evaluation

This run reloads a saved checkpoint and scores one split. Weights are not updated. Do not use the output to choose a new model, threshold, or clinical operating point.

- Checkpoint: `runs/breastmnist_baseline_seed7/model.pt`
- Split: `test` (156 images)
- Selected training epoch recorded in checkpoint: 5

## Discrimination at the default 0.5 threshold

| Metric | Estimate |
|---|---:|
| Accuracy | 0.833 |
| Malignant precision | 0.674 |
| Malignant recall | 0.738 |
| Specificity | 0.868 |
| ROC-AUC | 0.877 |
| PR-AUC | 0.771 |

Confusion counts at threshold 0.5: TP=31, TN=99, FP=15, FN=11.

## Calibration

- Brier score: 0.123 (0 is perfect; 1 is worst for binary labels).
- Expected calibration error (10 equal-width bins): 0.084.

| Bin | Count | Mean confidence | Positive rate | Absolute gap |
|---:|---:|---:|---:|---:|
| 0 | 58 | 0.036 | 0.052 | 0.016 |
| 1 | 29 | 0.144 | 0.172 | 0.028 |
| 2 | 12 | 0.242 | 0.000 | 0.242 |
| 3 | 7 | 0.334 | 0.286 | 0.048 |
| 4 | 4 | 0.439 | 0.250 | 0.189 |
| 5 | 6 | 0.554 | 0.167 | 0.387 |
| 6 | 0 | — | — | — |
| 7 | 8 | 0.764 | 0.500 | 0.264 |
| 8 | 12 | 0.850 | 0.667 | 0.183 |
| 9 | 20 | 0.938 | 0.900 | 0.038 |

Calibration on a tiny public split is descriptive only. Do not choose a clinical threshold or treat well-calibrated scores as safe probabilities.
