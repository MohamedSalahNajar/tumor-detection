# BreastMNIST Seed Variability Results

## Scope

This experiment repeats the same fixed training protocol with three initialization/shuffle seeds on the official BreastMNIST split. It measures sensitivity to training randomness only. It is not cross-validation, an independent external test, or clinical validation: all three models were evaluated on the same 156-image test split.

Archive: `data/medmnist/breastmnist_64.npz`, published MD5 `742edef2a1fd1524b2efff4bd7ba9364` (verified). Source labels were mapped so malignant is positive. Each run trained 5 epochs with batch size 32, learning rate 0.0005, base channels 8, AdamW, and CPU (2 threads); best checkpoint was chosen by validation loss.

## Test results by seed

| Seed | Selected epoch | Accuracy | Malignant precision | Malignant recall | Specificity | ROC-AUC | PR-AUC | Malignant cases missed (of 42) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 4 | 0.814 | 0.676 | 0.595 | 0.895 | 0.834 | 0.689 | 17 |
| 11 | 4 | 0.808 | 0.688 | 0.524 | 0.912 | 0.818 | 0.641 | 20 |
| 23 | 5 | 0.756 | 0.667 | 0.190 | 0.965 | 0.820 | 0.618 | 34 |
| Mean | | 0.793 | 0.677 | 0.437 | 0.924 | 0.824 | 0.649 | 23.7 |
| Seed SD | | 0.032 | 0.010 | 0.216 | 0.037 | 0.009 | 0.037 | |

Per-run test metrics include stratified bootstrap 95% intervals in each `seed_<n>/test_metrics.json`. Aggregate per-seed means, standard deviations, configurations, and all per-seed metrics are in `sweep_summary.json`. Every run has a separate checkpoint, history, and config.

## Interpretation

The most important finding is unstable and inadequate malignant sensitivity: across these three initializations, mean sensitivity is 43.7% with a 21.6 percentage-point standard deviation. At the default probability threshold, the seed-23 checkpoint missed 34 of 42 malignant examples. The runs share the same small test set, so mean and standard deviation summarize initialization variability only; they do not increase the number of independent patients or support a clinical confidence claim.

This model is unsuitable for patient care, diagnosis, triage, treatment, or monitoring. Do not select a threshold or model based on this test-set sweep. Clinical research would need a pre-specified target population/task, governed representative data, patient/site-aware external validation, clinician-led review, and institutional/regulatory oversight.
