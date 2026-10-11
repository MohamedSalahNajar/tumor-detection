"""Synthetic grayscale image generator for prototype tests and demos."""

import numpy as np


def make_synthetic_image(
    shape: tuple[int, int] = (128, 128),
    center: tuple[int, int] | None = (64, 64),
    radius: int = 8,
    seed: int = 7,
) -> np.ndarray:
    """Generate a noisy background with an optional bright circular region."""
    if len(shape) != 2 or min(shape) < 8:
        raise ValueError("shape must contain two dimensions of at least 8 pixels")
    if radius < 1:
        raise ValueError("radius must be positive")
    if center is not None and not (0 <= center[0] < shape[0] and 0 <= center[1] < shape[1]):
        raise ValueError("center must lie within the image")

    rng = np.random.default_rng(seed)
    image = rng.normal(loc=0.15, scale=0.025, size=shape).astype(np.float32)
    if center is not None:
        rows, columns = np.ogrid[: shape[0], : shape[1]]
        disk = (rows - center[0]) ** 2 + (columns - center[1]) ** 2 <= radius**2
        image[disk] += 0.65
    return np.clip(image, 0.0, 1.0)