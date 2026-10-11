"""Run a synthetic-only end-to-end detector demonstration."""

import json

from ai.models import SelfHealingNeuralNetwork
from core.imaging.synthetic import make_synthetic_image


def main() -> None:
    image = make_synthetic_image()
    result = SelfHealingNeuralNetwork().analyze(image)
    print(
        json.dumps(
            {
                "status": result.status,
                "message": result.message,
                "candidates": [
                    {
                        "bbox_xyxy": candidate.bbox,
                        "area_pixels": candidate.area_pixels,
                        "contrast_score": round(candidate.contrast_score, 2),
                    }
                    for candidate in result.candidates
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()