"""Evaluate a saved checkpoint without retraining or changing weights."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ai.breastmnist import BreastMNISTSplit
from ai.calibration import calibration_report, confusion_counts
from ai.models.deep_tumor_net import DeepTumorNet
from ai.train import binary_metrics


def collect_outputs(model, loader, device: torch.device) -> tuple[list[int], list[float], list[list[float]]]:
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    logits: list[list[float]] = []
    with torch.no_grad():
        for images, _masks, targets in loader:
            outputs = model(images.to(device), return_segmentation=False)["classification"]
            labels.extend(int(value) for value in targets.tolist())
            probabilities.extend(outputs.softmax(1)[:, 1].cpu().tolist())
            logits.extend(outputs.cpu().tolist())
    return labels, probabilities, logits


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DeepTumorNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def render_report(report: dict) -> str:
    calibration = report["calibration"]
    confusion = report["confusion_at_default_threshold"]
    metrics = report["metrics"]
    lines = [
        "# Locked-Checkpoint Evaluation",
        "",
        "This run reloads a saved checkpoint and scores one split. Weights are not updated. Do not use the output to choose a new model, threshold, or clinical operating point.",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Split: `{report['split']}` ({report['sample_count']} images)",
        f"- Selected training epoch recorded in checkpoint: {report['selected_epoch']}",
        "",
        "## Discrimination at the default 0.5 threshold",
        "",
        "| Metric | Estimate |",
        "|---|---:|",
        f"| Accuracy | {metrics['accuracy']:.3f} |",
        f"| Malignant precision | {metrics['precision']:.3f} |",
        f"| Malignant recall | {metrics['recall']:.3f} |",
        f"| Specificity | {metrics['specificity']:.3f} |",
        f"| ROC-AUC | {metrics['roc_auc']:.3f} |",
        f"| PR-AUC | {metrics['pr_auc']:.3f} |",
        "",
        f"Confusion counts at threshold {confusion['threshold']}: "
        f"TP={confusion['true_positives']}, TN={confusion['true_negatives']}, "
        f"FP={confusion['false_positives']}, FN={confusion['false_negatives']}.",
        "",
        "## Calibration",
        "",
        f"- Brier score: {calibration['brier_score']:.3f} (0 is perfect; 1 is worst for binary labels).",
        f"- Expected calibration error ({calibration['bin_count']} equal-width bins): "
        f"{calibration['expected_calibration_error']:.3f}.",
        "",
        "| Bin | Count | Mean confidence | Positive rate | Absolute gap |",
        "|---:|---:|---:|---:|---:|",
    ]
    for bin_record in calibration["bins"]:
        if bin_record["count"] == 0:
            lines.append(f"| {bin_record['index']} | 0 | — | — | — |")
            continue
        lines.append(
            f"| {bin_record['index']} | {bin_record['count']} | {bin_record['mean_confidence']:.3f} | "
            f"{bin_record['positive_rate']:.3f} | {bin_record['absolute_gap']:.3f} |"
        )
    lines.extend(["", calibration["warning"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default="data/medmnist")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--allow-test-split", action="store_true", help="Required to score the locked test split")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.split == "test" and not args.allow_test_split:
        parser.error("refusing to score the locked test split without --allow-test-split")

    device = torch.device("cpu")
    model, checkpoint = load_model(Path(args.checkpoint), device)
    dataset = BreastMNISTSplit("val" if args.split == "val" else "test", root=args.root, download=False)
    labels, probabilities, _logits = collect_outputs(model, DataLoader(dataset, batch_size=args.batch_size), device)
    report = {
        "purpose": "Reload a locked checkpoint and summarize discrimination plus calibration.",
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "sample_count": len(labels),
        "selected_epoch": checkpoint.get("selected_epoch"),
        "metrics": binary_metrics(labels, probabilities),
        "confusion_at_default_threshold": confusion_counts(labels, probabilities),
        "calibration": calibration_report(labels, probabilities),
        "weights_updated": False,
        "threshold_selected": False,
    }
    output_directory = Path(args.output)
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_directory / "RESULTS.md").write_text(render_report(report), encoding="utf-8")
    print(json.dumps({
        "split": args.split,
        "brier_score": report["calibration"]["brier_score"],
        "expected_calibration_error": report["calibration"]["expected_calibration_error"],
        "false_negatives": report["confusion_at_default_threshold"]["false_negatives"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
