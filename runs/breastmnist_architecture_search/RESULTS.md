# Validation-Only Architecture Search

This bounded search selected candidates by minimum validation loss. The held-out test split was not evaluated. Results are exploratory, based on one seed and one small validation split; they do not establish generalization or clinical utility.

Selected by validation loss: `control`.

| Candidate | Parameters | Best epoch | Validation loss | Accuracy | Malignant recall | Specificity | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 516,839 | 4 | 0.346 | 0.833 | 0.619 | 0.912 | 0.885 | 0.810 |
| double_width_depth | 4,027,399 | 4 | 0.535 | 0.731 | 0.905 | 0.667 | 0.908 | 0.814 |
| width_only | 2,061,447 | 4 | 0.484 | 0.821 | 0.619 | 0.895 | 0.884 | 0.814 |
| depth_only | 1,009,063 | 1 | 0.662 | 0.731 | 0.000 | 1.000 | 0.726 | 0.599 |
| malignant_weight | 516,839 | 4 | 0.509 | 0.756 | 0.619 | 0.807 | 0.856 | 0.764 |

The selected row is only the best result within this small search and validation set. Do not use these results to choose a clinical threshold or model. Keep the locked test set for a pre-specified final evaluation after the research protocol is agreed.
