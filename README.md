# Tumor Image Research Prototype

See [ROADMAP.md](ROADMAP.md) for current status, owners, decision gates, and next steps.

This repository contains two distinct experimental paths: a deep neural
network that can be trained on the public BreastMNIST classification benchmark,
and a toy geometric-phantom segmentation task. Neither is a clinical system.
No model can guarantee correct tumor detection. Do not use outputs to diagnose,
triage, treat, or monitor a person, and do not load private patient data.

The benchmark model is a residual convolutional encoder-decoder with a binary
classification head. BreastMNIST contains 780 preprocessed breast-ultrasound
images (546 train, 78 validation, 156 test), at low resolution; it is small,
single-domain, and explicitly not intended for clinical use. It includes image
classification labels, not pixel-level tumor masks, so segmentation is not
trained on BreastMNIST. Its original labels are `0=malignant` and
`1=normal/benign`; the loader remaps malignant to the positive class so reported
recall is malignant-class recall. Dataset license: CC BY 4.0. Cite Yang et al.,
"MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical
image classification," Scientific Data, 2023.

The separate synthetic path creates simple geometric phantoms. Those images
are not real medical images and must not be used to estimate medical accuracy.
The legacy `SelfHealingNeuralNetwork` class is only a fail-closed software
wrapper that restores known configuration after an unexpected runtime error;
it is not a neural network and does not control cells or recommend interventions.

## Train on BreastMNIST

Requires Python 3.10+, PyTorch, and the packages in `requirements.txt`. The
official MedMNIST API downloads the dataset to `data/medmnist` on first run.
If the automatic download is unavailable, download the official
[BreastMNIST 64px archive](https://zenodo.org/records/10519652/files/breastmnist_64.npz?download=1)
from Zenodo and place it at `data/medmnist/breastmnist_64.npz`.

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m ai.train --dataset breastmnist --epochs 5 --batch-size 16
```

Training writes the best-validation weights, per-epoch validation metrics, and
final held-out test metrics under `runs/breastmnist` (rename `--output` if
desired). The small model is intentionally modest for this tiny dataset; a
larger network would mainly raise overfitting risk, not establish certainty.
The completed baseline and its limitations are documented in
[the experiment report](runs/breastmnist_baseline_seed7/RESULTS.md).
Three-seed initialization variability is documented in
[the seed-sweep report](runs/breastmnist_seed_sweep/RESULTS.md); malignant
recall varied from 0.190 to 0.595 on the same small test split.
The configurable 4.0M-parameter depth/width experiment is documented in
[its validation-only report](runs/breastmnist_deeper_validation_seed7/RESULTS.md);
it did not improve overall validation metrics over the 0.52M-parameter baseline.
The class-weight experiment is in
[its validation-only report](runs/breastmnist_weighted_validation_seed7/RESULTS.md);
the tested 2.7 malignant weight also did not improve the selected validation metrics.
A five-candidate validation-only architecture search still selected the control
network; see [that report](runs/breastmnist_architecture_search/RESULTS.md).
Exact-image split overlap and locked-checkpoint calibration are in
[the split audit](runs/breastmnist_split_audit/RESULTS.md) and
[the calibration report](runs/breastmnist_baseline_seed7/calibration/RESULTS.md).
Locked-checkpoint false-negative review and validation-fitted temperature scaling are in
[the error analysis](runs/breastmnist_baseline_seed7/error_analysis/RESULTS.md).

To rerun that research-only seed sweep after installing requirements:

```powershell
$env:PYTHONPATH = "src"
python -m ai.seed_sweep --seeds 7 11 23 --output-root runs/breastmnist_seed_sweep
```

To rerun the validation-only architecture search, the exact-image split audit, or
locked-checkpoint calibration (test scoring stays disabled unless you pass
`--allow-test-split` for the already-published baseline):

```powershell
$env:PYTHONPATH = "src"
python -m ai.architecture_search --output-root runs/breastmnist_architecture_search
python -m ai.split_audit --output runs/breastmnist_split_audit
python -m ai.evaluate --checkpoint runs/breastmnist_baseline_seed7/model.pt --split test --allow-test-split --output runs/breastmnist_baseline_seed7/calibration
python -m ai.error_analysis --checkpoint runs/breastmnist_baseline_seed7/model.pt --split test --allow-test-split --output runs/breastmnist_baseline_seed7/error_analysis
```

The training CLI supports `--base-channels`, `--blocks-per-stage`,
`--dropout-rate`, `--learning-rate`, `--weight-decay`, and
`--positive-class-weight`. Here, channels mean learned feature maps, not
biological receptors. Use `--skip-test`
for architecture/hyperparameter experiments so the held-out test split is not
consulted during model iteration. These settings affect model capacity and
regularization; they do not guarantee higher accuracy.

To reproduce the recorded wider/deeper validation-only experiment:

```powershell
$env:PYTHONPATH = "src"
python -m ai.train --dataset breastmnist --epochs 5 --batch-size 32 --learning-rate 0.0003 --weight-decay 0.03 --base-channels 16 --blocks-per-stage 2 --dropout-rate 0.5 --cpu-threads 2 --seed 7 --skip-test --output runs/breastmnist_deeper_validation_seed7
```

Run the regression suite with:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

## Train on geometric phantoms

```powershell
$env:PYTHONPATH = "src"
python -m ai.train --dataset synthetic --samples 1000 --epochs 5
```

## Synthetic candidate demo

```powershell
$env:PYTHONPATH = "src"
python src/demo.py
```

## Before any clinical research

Use of real imaging data would require appropriate institutional review,
consent or another valid legal basis, secure governance, de-identification,
and an approved data-use agreement. A clinical program would also need
multisite, representative datasets; locked external validation; subgroup and
workflow evaluation with qualified clinicians; prospective safety studies;
privacy/security controls; and applicable regulatory review. No market or
competitive analysis, clinical validation, sensor integration, or clinician
user testing has been performed in this scaffold.