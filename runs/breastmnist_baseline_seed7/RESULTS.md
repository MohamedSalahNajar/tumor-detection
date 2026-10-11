# BreastMNIST Baseline Results

## Run identity

- Dataset: official MedMNIST BreastMNIST 64px archive, CC BY 4.0.
- Archive: `data/medmnist/breastmnist_64.npz`.
- Published archive MD5: `742edef2a1fd1524b2efff4bd7ba9364`; verified before loading.
- Split sizes: 546 train, 78 validation, 156 test.
- Positive class: malignant. The loader maps source label `0` (malignant) to target `1`.
- Model: residual CNN encoder with classification head; segmentation decoder disabled for this image-classification task.
- Training: 5 epochs, batch size 32, AdamW learning rate 0.0005, base channels 8, seed 7, CPU with 2 threads.
- Checkpoint selection: lowest validation loss; selected epoch 5, validation loss 0.3303.
- Archive checksum: `742edef2a1fd1524b2efff4bd7ba9364` (verified and included in run metadata).
- Test set: evaluated after checkpoint selection, once during training and once after loading the saved checkpoint to verify reload behavior. The same checkpoint produced the same test metrics both times.

Full settings are in `run_config.json`, epoch history in `history.csv`, serialized model weights in `model.pt`, and test metrics including bootstrap intervals in `test_metrics.json`.

The trainer now validates the cached archive, records its checksum, explicitly seeds the shuffled sampler, and atomically writes history and the best checkpoint after each epoch. These protections were added after this baseline was run; the saved baseline itself used seed 7 on CPU and is unchanged.

## Held-out test metrics

At the default 0.5 probability threshold:

| Metric | Estimate | Stratified bootstrap 95% interval |
|---|---:|---:|
| Accuracy | 0.833 | 0.782 to 0.894 |
| Malignant precision | 0.674 | 0.582 to 0.795 |
| Malignant recall/sensitivity | 0.738 | 0.595 to 0.857 |
| Specificity | 0.868 | 0.807 to 0.930 |
| ROC-AUC | 0.877 | 0.816 to 0.938 |
| Average precision (PR-AUC) | 0.771 | 0.671 to 0.888 |

The test split contained 42 malignant and 114 non-malignant examples. At this threshold, malignant recall corresponds to 31/42 detected malignant examples and 11/42 missed examples. The confusion counts are specific to this tiny benchmark split and this trained checkpoint.

## Interpretation

This is a successful reproducible training-pipeline benchmark, not evidence of clinical utility. BreastMNIST has only 780 low-resolution, preprocessed ultrasound images, one modality, and no segmentation masks. The test cohort is small; uncertainty is substantial. Performance can change with another seed, threshold, population, scanner, or site. The observed missed malignancies alone make this model unsuitable for patient care. It must not be used for diagnosis, triage, treatment, or monitoring.

No threshold was selected for a clinical operating point. A later locked-checkpoint pass added calibration on this same test split without changing weights: Brier score 0.123 and 10-bin expected calibration error 0.084; mid-probability bins are sparse and poorly aligned. See [the calibration report](calibration/RESULTS.md). An exact-image audit found one identical 64px array in train and test with opposite remapped labels; that is not a patient-level leakage check. No external-site validation, subgroup evaluation, prospective study, clinician workflow evaluation, privacy review, or regulatory review was performed.

## Artifacts

- `model.pt`: selected checkpoint, model configuration, seed, and run configuration.
- `run_config.json`: parameters, software versions, device, data split sizes, and positive-class definition.
- `history.csv`: train/validation loss and validation metrics by epoch.
- `test_metrics.json`: held-out test metrics and bootstrap intervals.
- `calibration/`: reloaded-checkpoint Brier score, expected calibration error, and reliability bins.
