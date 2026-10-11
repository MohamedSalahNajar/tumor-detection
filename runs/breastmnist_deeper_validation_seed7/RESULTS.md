# Deeper/Wider Model Validation Experiment

## Purpose and protocol

This run tested the hypothesis that adding feature channels and residual units would improve learning on BreastMNIST. It was a validation-only experiment: `--skip-test` was enabled, the official held-out test split was not evaluated, and this run must not be used to claim test or clinical performance.

- Data: official BreastMNIST 64px, 546 training / 78 validation / 156 reserved test images; archive MD5 verified as `742edef2a1fd1524b2efff4bd7ba9364`.
- Model: base channels 16, two residual blocks per encoder/decoder stage, dropout 0.5; 4,027,399 parameters.
- Baseline comparator: base channels 8, one block per stage, dropout 0.4; 516,839 parameters.
- Optimizer: AdamW; learning rate 0.0003, weight decay 0.03; batch 32; 5 epochs; seed 7; CPU, 2 threads.
- Selection: minimum validation loss; selected epoch 4.
- Test evaluation: skipped by design. Do not compare or fill in test metrics for this candidate.

Exact settings, per-epoch history, and weights are in `run_config.json`, `history.csv`, and `model.pt`. `test_metrics.json` records that test evaluation was skipped.

## Validation results

| Measure | Small baseline, epoch 5 | Deeper/wider candidate, epoch 4 |
|---|---:|---:|
| Parameters | 516,839 | 4,027,399 |
| Validation loss | 0.330 | 0.535 |
| Accuracy at 0.5 threshold | 0.833 | 0.731 |
| Malignant recall at 0.5 threshold | 0.810 | 0.905 |
| Specificity at 0.5 threshold | 0.842 | 0.667 |
| ROC-AUC | 0.931 | 0.908 |
| Average precision (PR-AUC) | 0.843 | 0.814 |

The candidate trades higher malignant recall for more false positives and worse values on the other listed validation measures. It predicts more positive cases at the default threshold. Its best validation loss is also materially worse. This single validation split is small (21 malignant validation examples), so estimates are noisy and this is not evidence that the model is better or worse on a broader population.

## Conclusion

The user-requested architectural option is now implemented: feature width and residual blocks per stage are configurable. This first larger model did **not** improve overall validation performance and showed unstable threshold behavior in early epochs. Increasing model size alone is not a solution to limited data. Do not use this model or baseline for diagnosis, triage, treatment, or monitoring.

Next meaningful work is approved, representative data and a clinician-defined task/evaluation plan, not further blind width/depth increases. Any later model selection requires a locked independent external cohort; no test-set score was generated for this candidate.
