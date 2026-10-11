"""Fail-closed runtime wrapper for the synthetic detector prototype."""

from dataclasses import dataclass

import numpy as np

from .tumor_detector import TumorCandidate, TumorDetector


@dataclass(frozen=True)
class AnalysisResult:
    status: str
    candidates: tuple[TumorCandidate, ...]
    message: str
    recovered: bool = False


class SelfHealingNeuralNetwork:
    """Monitor detector execution and fail closed; this is not a neural net.

    Recovery recreates the same known detector configuration after an
    unexpected runtime error. It does not learn from patient data, alter
    model weights, or recommend treatment.
    """

    def __init__(self, detector: TumorDetector | None = None) -> None:
        self._detector = detector or TumorDetector()
        self._known_config = {
            "min_area": self._detector.min_area,
            "threshold_sigma": self._detector.threshold_sigma,
        }

    def analyze(self, image: np.ndarray) -> AnalysisResult:
        try:
            candidates = self._detector.detect(image)
        except (TypeError, ValueError) as error:
            return AnalysisResult(
                status="review_required",
                candidates=(),
                message=f"Input rejected: {error}",
            )
        except Exception:
            self._detector = TumorDetector(**self._known_config)
            return AnalysisResult(
                status="review_required",
                candidates=(),
                message="Detector runtime failure; known configuration restored.",
                recovered=True,
            )

        return AnalysisResult(
            status="candidate_found" if candidates else "no_candidate",
            candidates=tuple(candidates),
            message="Synthetic-image result only; not for clinical use.",
        )