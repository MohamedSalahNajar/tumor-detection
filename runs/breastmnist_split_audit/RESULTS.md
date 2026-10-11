# BreastMNIST Split Integrity Audit

This check hashes official 64px images and reports identical pixel arrays. It cannot audit same-patient leakage because BreastMNIST does not provide patient or site identifiers.

- Unique images: 779
- Duplicate groups: 1
- Cross-split duplicate groups: 1
- Within-split duplicate groups: 0
- Duplicate groups with conflicting labels: 1

Split sizes: train=546, val=78, test=156

| Digest prefix | Count | Cross-split | Conflicting labels | Splits | Remapped labels |
|---|---:|---|---|---|---|
| `3379cc1edee1` | 2 | True | True | test, train | 0, 1 |

Remapped labels: `1` is malignant and `0` is normal/benign. Conflicting labels on identical pixels mean the same 64px array is treated as both classes across splits. This can come from a true duplicate with inconsistent labels, or from different source images that became identical after 64px preprocessing.

Identical pixels can indicate label leakage between splits, but distinct images from the same person would not be detected without patient identifiers.
