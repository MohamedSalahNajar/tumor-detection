"""Adapter for the public MedMNIST BreastMNIST benchmark."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

BREASTMNIST_64_MD5 = "742edef2a1fd1524b2efff4bd7ba9364"


def verify_archive(path: Path) -> str:
    """Verify and return the published checksum for the official archive."""
    if not path.is_file():
        raise FileNotFoundError(f"BreastMNIST archive not found: {path}")
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    if digest != BREASTMNIST_64_MD5:
        raise ValueError(f"BreastMNIST archive checksum mismatch: {digest}")
    return digest


class BreastMNISTSplit(Dataset):
    """Load one official BreastMNIST split at 64x64 resolution."""

    def __init__(self, split: str, root: str = "data/medmnist", download: bool = True) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        try:
            from medmnist import BreastMNIST
        except ImportError as error:
            raise RuntimeError("Install requirements.txt to use BreastMNIST") from error

        Path(root).mkdir(parents=True, exist_ok=True)
        archive_path = Path(root) / "breastmnist_64.npz"
        if archive_path.exists():
            verify_archive(archive_path)
        self.dataset = BreastMNIST(
            split=split,
            root=root,
            download=download,
            size=64,
            as_rgb=False,
        )
        self.archive_md5 = verify_archive(archive_path)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, label = self.dataset[index]
        pixels = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(np.ascontiguousarray(pixels[None])).float()
        empty_mask = torch.zeros_like(image_tensor)
        raw_label = int(np.asarray(label).reshape(-1)[0])
        label_tensor = torch.tensor(1 - raw_label, dtype=torch.long)
        return image_tensor, empty_mask, label_tensor