# Project Roadmap

## Purpose and limits

Build a research prototype that may help qualified clinicians study medical-image analysis. This is not a diagnostic or treatment system. No model can know with certainty, and the current repository has not been trained on real medical images or clinically validated. The earlier training smoke test used generated geometric images only; its metrics are not medical evidence.

We will not implement interventions that stimulate, suppress, or regrow cells. "Self-healing" in this codebase means software reliability only, such as rejecting invalid input and restoring a known configuration. Any future clinical use requires a separate, qualified medical, ethical, privacy, and regulatory program.

## Current status

- [x] Synthetic grayscale candidate detector and toy image generator exist.
- [x] PyTorch training code and a BreastMNIST adapter exist.
- [x] A one-epoch synthetic smoke run completed and wrote a temporary checkpoint and metrics.
- [x] Unit tests pass, including checksum, artifact-writing, seed-summary, network-contract, checkpoint round-trip, class-weight, split-audit, and calibration guards.
- [x] Current trainer completes a one-epoch synthetic segmentation run and writes parseable history, checkpoint, config, and held-out metrics.
- [x] Official BreastMNIST 64px archive downloaded and MD5-verified.
- [x] Five-epoch benchmark baseline trained, validation-selected, and evaluated on the official held-out test split.
- [x] Model checkpoint reloaded and test metrics reproduced.
- [x] Three-seed variability sweep completed on the same fixed BreastMNIST split.
- [x] Future benchmark runs now checksum cached data and checkpoint/history saves are atomic.
- [x] Network contract tests confirm classification/segmentation output shapes, gradients, decoder bypass, spatial/channel validation, and checkpoint reload.
- [x] Model width/depth, dropout, weight decay, and validation-only mode are configurable; a deeper/wider candidate was compared without using the held-out test split.
- [x] Optional positive-class loss weighting is configurable; a validation-only imbalance experiment was recorded without using the held-out test split.
- [x] A bounded validation-only architecture/loss-weight search compared control, width, depth, combined capacity, and malignant-class weighting; the control remained best by validation loss.
- [x] Exact-image overlap audit for the official BreastMNIST splits, and calibration (Brier score / ECE) for the locked baseline checkpoint.
- [ ] No clinician review, prospective testing, privacy review, or regulatory review has happened.

## Recommended next steps

### 1. Make the public benchmark run reproducibly

**Status: complete for a reproducible public benchmark baseline.**

1. Downloaded `data/medmnist/breastmnist_64.npz` and verified MD5 `742edef2a1fd1524b2efff4bd7ba9364`.
2. Loaded official splits (546/78/156), verified image shape and malignant-positive mapping.
3. Trained 5 epochs; selected by validation loss; evaluated held-out test data.
4. Saved checkpoint, configuration, validation history, and test metrics under `runs/breastmnist_baseline_seed7/`.
5. Reloaded the checkpoint and reproduced test metrics. Adapter and metric tests pass.
6. Hardened future benchmark runs to validate cached archives, record the dataset checksum, explicitly seed the training sampler, and atomically save each epoch's history and best checkpoint.

**Result:** Test accuracy 0.833, malignant sensitivity 0.738, specificity 0.868, ROC-AUC 0.877, and PR-AUC 0.771 on only 156 examples (42 malignant). See [benchmark results](runs/breastmnist_baseline_seed7/RESULTS.md) and machine-readable artifacts in that run folder. These are benchmark results only and do not establish clinical performance; 11 of 42 malignant test examples were missed at the default threshold.

### 1b. Capacity experiment

Made feature width and residual depth configurable, along with dropout, weight decay, and a `--skip-test` switch for validation-only model iteration. Tested a 4.03M-parameter model (16 channels, 2 residual blocks/stage) against the 0.52M-parameter baseline using the same seed and validation split. The deeper candidate's selected validation loss was 0.535 versus 0.330; accuracy 0.731 versus 0.833; specificity 0.667 versus 0.842; ROC-AUC 0.908 versus 0.931; and PR-AUC 0.814 versus 0.843. Recall was higher (0.905 versus 0.810), with lower specificity. The test split was explicitly skipped. Full record: [validation-only capacity experiment](runs/breastmnist_deeper_validation_seed7/RESULTS.md). This does not support claiming the larger model is more accurate overall.

### 2. Decide the research target before scaling the model

**You should decide with a qualified oncologist/radiologist:**

- Which tumor, organ, imaging modality, and intended research task matter first.
- Whether the goal is image-level classification, lesion localization, segmentation, or longitudinal change analysis. These require different labels and model outputs.
- Who the intended users are and what decision the output might inform.
- Which patient populations, scanners, sites, and failure cases must be represented.

A larger network is not the next priority: BreastMNIST has only 780 low-resolution images and classification labels, not segmentation masks. The first larger candidate had worse overall validation results. Scaling the architecture further before defining the task and sourcing enough appropriately labeled data risks overfitting without improving reliability.

The CLI can also weight malignant examples in the training objective. A 2.7 positive-class weight was tested using the baseline-sized network, with test evaluation disabled. Its best validation checkpoint (epoch 4) had accuracy 0.756, recall 0.619, specificity 0.807, ROC-AUC 0.856, and PR-AUC 0.764; this did not improve on the unweighted validation baseline. Threshold metrics shifted substantially by epoch. See [the class-weight report](runs/breastmnist_weighted_validation_seed7/RESULTS.md). The weighting parameter changes the loss and does not guarantee better sensitivity or accuracy.

### 2b. Bounded architecture search

**Status: complete; validation-only.** Five four-epoch candidates reused seed 7 and skipped the test split. Selected by minimum validation loss: the control network (0.52M parameters, validation loss 0.346). Larger width, extra depth, combined capacity, and 2.7 malignant weighting all had higher selected validation loss. Depth-only collapsed to recall 0.000 at its best-loss epoch. Full table: [architecture search](runs/breastmnist_architecture_search/RESULTS.md). This search does not unlock a better model for patient use.

### 2c. Split overlap and locked-checkpoint calibration

**Status: complete as software diagnostics on the public benchmark.** Exact SHA-256 image hashes were compared across the official train/val/test splits; this is not a patient-level leakage audit. One 64px image appears in both train (malignant) and test (non-malignant). The locked seed-7 baseline was reloaded without updating weights: Brier score 0.123, expected calibration error 0.084, still 11 missed malignancies at threshold 0.5. See [split audit](runs/breastmnist_split_audit/RESULTS.md) and [calibration report](runs/breastmnist_baseline_seed7/calibration/RESULTS.md). No new threshold was selected.

### 2a. Training-seed stability check

**Status: complete as a software/research diagnostic only.** Three five-epoch runs (seeds 7, 11, 23) reused the same official split. Test malignant recall was 0.595, 0.524, and 0.190; mean 0.437 with standard deviation 0.216. Seed 23 missed 34 of 42 malignant cases. Full results are in [the seed-sweep report](runs/breastmnist_seed_sweep/RESULTS.md). Since the same small test set is reused, this is not an independent validation cohort. It demonstrates that the baseline's operating behavior is unstable and unsuitable for patient care. Do not tune a threshold using these test results.

### 3. Establish data access and governance

**You / your institution / the clinical collaborators must arrange this; I cannot authorize access or consent on anyone's behalf.**

- Identify a lawful, institutionally approved dataset and confirm its license/data-use terms.
- Obtain ethics/IRB review where required, consent or another valid legal basis, a data-use agreement, and privacy/security approval before handling identifiable or sensitive records.
- Define de-identification, access control, encryption, retention, audit, and breach-response requirements.
- Prefer patient-level and site-aware splits to prevent images from the same person or institution leaking across training and evaluation.
- Include radiologist/oncologist annotation protocols and adjudication for disagreements; patient participation should be overseen by the clinical research team.

Do not send private patient data through this chat or place it in this repository until approved safeguards and a secure workflow are established.

### 4. Build a defensible research model

Once the target task and approved dataset are defined, I can help adapt the pipeline to the data format and labels, establish simple baselines, handle imbalance, track experiments, and report task-appropriate metrics. For classification, include sensitivity/recall, specificity, precision, ROC-AUC and PR-AUC with confidence intervals; for localization/segmentation, use suitable overlap and detection measures. Evaluate calibration, subgroup performance, scanner/site shift, and clinically meaningful error rates. Keep a locked external test set and document limitations.

No single score proves safety. A qualified clinical team must review false negatives, false positives, workflow effects, and the consequences of model errors.

### 5. Only then consider prospective or product work

Prospective studies, workflow integration, real-time imaging interfaces, and any clinical deployment need clinical leadership, institutional authorization, security engineering, applicable regulatory advice, and prospective safety/effectiveness evidence. The prototype must remain clearly marked research-only until those gates are met. Do not connect it to equipment or patient care as an autonomous decision-maker.

## Immediate action list

**I can do next:** after the research target and approved dataset are identified, adapt the pipeline, add patient/site-aware splits when identifiers exist, and support a clinician-led validation protocol. Public-benchmark training, seed-stability, architecture search, exact-image split audit, and locked-checkpoint calibration are complete; this model is not suitable for patient care, and increasing model size is not the next priority.

**You should do next:**

1. Review the [single-run report](runs/breastmnist_baseline_seed7/RESULTS.md) and [three-seed report](runs/breastmnist_seed_sweep/RESULTS.md); in the seed sweep, the worst run missed 34/42 malignant examples.
2. Confirm or change the assumed first target (breast-ultrasound image classification) with a qualified oncologist/radiologist. This assumption only scopes the code work; it is not a clinical recommendation.
3. Identify the clinical collaborator and institution/data-governance route before seeking or using patient images. No patient data should be put in this repository or shared in this chat.

Once you identify the research target and governance route, I can plan the next dataset and evaluation changes around that scope. Do not use this checkpoint for patient care.
