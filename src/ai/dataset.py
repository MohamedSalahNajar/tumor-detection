"""Dataset and transforms for the toy synthetic segmentation task."""

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SyntheticTumorDataset(Dataset):
    """Load generated PNG phantoms; never a substitute for clinical datasets."""

    def __init__(self, root: str | Path, augment: bool = False) -> None:
        self.root = Path(root)
        self.augment = augment
        with (self.root / "manifest.csv").open(newline="", encoding="utf-8") as file:
            self.samples = list(csv.DictReader(file))
        if not self.samples:
            raise ValueError(f"No samples in {self.root / 'manifest.csv'}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = np.asarray(Image.open(self.root / sample["image"]).convert("L"), dtype=np.float32) / 255.0
        mask = np.asarray(Image.open(self.root / sample["mask"]).convert("L"), dtype=np.float32) / 255.0
        if self.augment:
            if np.random.random() < 0.5:
                image, mask = np.fliplr(image), np.fliplr(mask)
            if np.random.random() < 0.5:
                image, mask = np.flipud(image), np.flipud(mask)
        image_tensor = torch.from_numpy(np.ascontiguousarray(image[None])).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask[None])).float()
        label = torch.tensor(int(sample["label"]), dtype=torch.long)
        return image_tensor, mask_tensor, label
