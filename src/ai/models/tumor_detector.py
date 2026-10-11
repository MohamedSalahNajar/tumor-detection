"""Experimental candidate locator for synthetic grayscale images only."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TumorCandidate:
    """A bright connected region in a simulated image, not a diagnosis."""

    bbox: tuple[int, int, int, int]
    area_pixels: int
    peak_intensity: float
    contrast_score: float


class TumorDetector:
    """Find conspicuously bright regions in 2D synthetic images.

    This baseline uses robust global thresholding and connected components.
    It is intended for prototype plumbing and must not be used with patient
    images or to guide clinical decisions.
    """

    def __init__(self, min_area: int = 3, threshold_sigma: float = 4.0) -> None:
        if min_area < 1:
            raise ValueError("min_area must be at least 1")
        if threshold_sigma <= 0:
            raise ValueError("threshold_sigma must be positive")
        self.min_area = min_area
        self.threshold_sigma = threshold_sigma

    def detect(self, image: np.ndarray) -> list[TumorCandidate]:
        """Return candidate regions ordered by descending contrast score."""
        pixels = np.asarray(image, dtype=np.float32)
        if pixels.ndim != 2 or not pixels.size:
            raise ValueError("image must be a non-empty 2D grayscale array")
        if not np.isfinite(pixels).all():
            raise ValueError("image values must all be finite")

        median = float(np.median(pixels))
        mad_sigma = float(np.median(np.abs(pixels - median))) * 1.4826
        spread_floor = float(np.ptp(pixels)) * 0.1
        robust_sigma = max(mad_sigma, spread_floor, 1e-6)
        threshold = median + self.threshold_sigma * robust_sigma
        mask = pixels > threshold
        height, width = pixels.shape
        visited = np.zeros(mask.shape, dtype=bool)
        candidates: list[TumorCandidate] = []

        for row, column in zip(*np.nonzero(mask & ~visited)):
            if visited[row, column]:
                continue
            stack = [(int(row), int(column))]
            visited[row, column] = True
            component: list[tuple[int, int]] = []
            while stack:
                current_row, current_column = stack.pop()
                component.append((current_row, current_column))
                for next_row in range(max(0, current_row - 1), min(height, current_row + 2)):
                    for next_column in range(max(0, current_column - 1), min(width, current_column + 2)):
                        if mask[next_row, next_column] and not visited[next_row, next_column]:
                            visited[next_row, next_column] = True
                            stack.append((next_row, next_column))

            if len(component) < self.min_area:
                continue
            rows, columns = zip(*component)
            peak = max(float(pixels[r, c]) for r, c in component)
            candidates.append(
                TumorCandidate(
                    bbox=(min(columns), min(rows), max(columns) + 1, max(rows) + 1),
                    area_pixels=len(component),
                    peak_intensity=peak,
                    contrast_score=(peak - median) / robust_sigma,
                )
            )

        return sorted(candidates, key=lambda candidate: candidate.contrast_score, reverse=True)