# Malignant-Class Weight Validation Experiment

## Protocol

This experiment tests whether weighting malignant examples in the training loss addresses the observed class imbalance. It reuses the baseline architecture and seed, and runs in validation-only mode: the held-out test split was not evaluated.

- Dataset/splits: official BreastMNIST 64px, 546 train / 78 validation / 156 reserved test.
- Model: baseline-sized residual CNN, base channels 8, one block per stage, dropout 0.4 (516,839 parameters).
- Class mapping: malignant is positive.
- Positive class weight: 2.7, approximately the training split's non-malignant-to-malignant ratio (399/147).
- Training: 5 epochs, batch 32, AdamW learning rate 0.0005, weight decay 0.01, seed 7, CPU, 2 threads.
- Selection: minimum weighted validation loss; selected epoch 4 (0.5095).
- Test evaluation: skipped. `test_metrics.json` records this.

Settings and all epoch metrics are stored in `run_config.json`, `history.csv`, and `model.pt`.

## Validation comparison at probability threshold 0.5

| Measure | Unweighted baseline, epoch 5 | Class-weighted, epoch 4 |
|---|---:|---:|
| Accuracy | 0.833 | 0.756 |
| Malignant recall | 0.810 | 0.619 |
| Specificity | 0.842 | 0.807 |
| ROC-AUC | 0.931 | 0.856 |
| Average precision (PR-AUC) | 0.843 | 0.764 |

The weighted loss did not improve these validation metrics at the selected checkpoint. Validation threshold behavior was unstable across epochs: epoch 3 classified 95% of validation images as malignant, while epoch 5 had malignant recall 0.476 and specificity 0.982. The validation split contains only 21 malignant examples.

The weighted validation loss is not directly comparable to the unweighted baseline loss because the objective itself changed. No test-set result was generated for this run, and no threshold was tuned on the test set.

## Interpretation

The imbalance weight changes the optimization objective and increases the cost of malignant-class errors, but this single experiment did not produce a more useful validation operating point. Do not use it for patient care. More tuning on this tiny split would risk fitting the validation cases; the next improvement depends on appropriate, representative, institutionally governed data and a clinician-defined operating objective.
