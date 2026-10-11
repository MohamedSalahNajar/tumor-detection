"""Probability calibration summaries for binary classification scores."""

import numpy as np


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logit_array = np.asarray(logits, dtype=np.float64)
    if logit_array.ndim != 2 or logit_array.shape[1] < 2:
        raise ValueError("logits must have shape (n_examples, n_classes)")
    scaled = logit_array / temperature
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exponential = np.exp(scaled)
    return exponential / np.sum(exponential, axis=1, keepdims=True)


def negative_log_likelihood(logits: np.ndarray, labels: list[int], temperature: float) -> float:
    probabilities = apply_temperature(logits, temperature)
    label_array = np.asarray(labels, dtype=np.int64)
    if probabilities.shape[0] != label_array.shape[0]:
        raise ValueError("logits and labels must have the same length")
    selected = probabilities[np.arange(label_array.size), label_array]
    return float(-np.mean(np.log(np.clip(selected, 1e-12, 1.0))))


def fit_temperature(logits: np.ndarray, labels: list[int]) -> float:
    """Choose a positive scalar temperature by grid search on negative log likelihood."""
    temperatures = np.concatenate((np.array([1.0]), np.logspace(-1, 1, 41)))
    unique_temperatures = np.unique(np.round(temperatures, decimals=8))
    return float(min(unique_temperatures, key=lambda temperature: negative_log_likelihood(logits, labels, float(temperature))))


def brier_score(labels: list[int], probabilities: list[float]) -> float:
    label_array = np.asarray(labels, dtype=np.float64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if label_array.size == 0:
        raise ValueError("labels and probabilities must be non-empty")
    return float(np.mean((probability_array - label_array) ** 2))


def reliability_bins(
    labels: list[int],
    probabilities: list[float],
    bin_count: int = 10,
) -> list[dict]:
    if bin_count < 2:
        raise ValueError("bin_count must be at least 2")
    label_array = np.asarray(labels, dtype=np.float64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if label_array.shape != probability_array.shape:
        raise ValueError("labels and probabilities must have the same length")
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    bins = []
    for index in range(bin_count):
        left = edges[index]
        right = edges[index + 1]
        if index == bin_count - 1:
            selected = (probability_array >= left) & (probability_array <= right)
        else:
            selected = (probability_array >= left) & (probability_array < right)
        count = int(np.sum(selected))
        if count == 0:
            bins.append({
                "index": index,
                "left": float(left),
                "right": float(right),
                "count": 0,
                "mean_confidence": None,
                "positive_rate": None,
                "absolute_gap": None,
            })
            continue
        mean_confidence = float(np.mean(probability_array[selected]))
        positive_rate = float(np.mean(label_array[selected]))
        bins.append({
            "index": index,
            "left": float(left),
            "right": float(right),
            "count": count,
            "mean_confidence": mean_confidence,
            "positive_rate": positive_rate,
            "absolute_gap": abs(mean_confidence - positive_rate),
        })
    return bins


def expected_calibration_error(bins: list[dict], sample_count: int) -> float:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    total = 0.0
    for bin_record in bins:
        if bin_record["count"] == 0:
            continue
        total += bin_record["count"] * bin_record["absolute_gap"]
    return float(total / sample_count)


def calibration_report(
    labels: list[int],
    probabilities: list[float],
    bin_count: int = 10,
) -> dict:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and probabilities must be non-empty and the same length")
    bins = reliability_bins(labels, probabilities, bin_count=bin_count)
    return {
        "sample_count": len(labels),
        "positive_count": int(sum(labels)),
        "bin_count": bin_count,
        "brier_score": brier_score(labels, probabilities),
        "expected_calibration_error": expected_calibration_error(bins, len(labels)),
        "bins": bins,
        "warning": (
            "Calibration on a tiny public split is descriptive only. Do not choose a clinical "
            "threshold or treat well-calibrated scores as safe probabilities."
        ),
    }


def confusion_counts(labels: list[int], probabilities: list[float], threshold: float = 0.5) -> dict:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1, exclusive")
    predictions = [int(probability >= threshold) for probability in probabilities]
    tp = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
    tn = sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    return {
        "threshold": threshold,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
    }
