"""Generate synthetic grayscale image datasets for software smoke tests only.

The shapes are toy geometric phantoms, not medical scans or biological data.
"""

from pathlib import Path

import numpy as np
from PIL import Image


def generate_dataset(output_dir: str | Path, count: int = 1000, size: int = 128, seed: int = 7) -> None:
    """Write deterministic PNG images and binary masks with a CSV manifest."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    images_dir = output / "images"
    masks_dir = output / "masks"
    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = ["image,mask,label"]

    for index in range(count):
        image = rng.normal(35, 6, (size, size)).clip(0, 255).astype(np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)
        has_lesion = bool(rng.random() < 0.65)
        if has_lesion:
            radius = int(rng.integers(max(4, size // 32), max(6, size // 10)))
            center_row = int(rng.integers(radius, size - radius))
            center_col = int(rng.integers(radius, size - radius))
            yy, xx = np.ogrid[:size, :size]
            inside = (yy - center_row) ** 2 + (xx - center_col) ** 2 <= radius**2
            mask[inside] = 255
            image[inside] = np.clip(image[inside].astype(np.int16) + int(rng.integers(35, 120)), 0, 255)
        image_name = f"sample_{index:06d}.png"
        mask_name = f"mask_{index:06d}.png"
        Image.fromarray(image, mode="L").save(images_dir / image_name)
        Image.fromarray(mask, mode="L").save(masks_dir / mask_name)
        rows.append(f"images/{image_name},masks/{mask_name},{int(has_lesion)}")

    (output / "manifest.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", nargs="?", default="data/synthetic_images")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    generate_dataset(args.output_dir, args.count, args.size, args.seed)
