"""Inspect locked-checkpoint errors without updating weights or choosing a threshold."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from ai.breastmnist import BreastMNISTSplit
from ai.calibration import apply_temperature, brier_score, calibration_report, fit_temperature
from ai.evaluate import collect_outputs, load_model


def bucket_examples(labels: list[int], probabilities: list[float], threshold: float = 0.5) -> list[dict]:
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have the same length")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1, exclusive")
    records = []
    for index, (label, probability) in enumerate(zip(labels, probabilities)):
        prediction = int(probability >= threshold)
        if label == 1 and prediction == 1:
            outcome = "true_positive"
        elif label == 0 and prediction == 0:
            outcome = "true_negative"
        elif label == 0 and prediction == 1:
            outcome = "false_positive"
        else:
            outcome = "false_negative"
        records.append({
            "index": index,
            "label": int(label),
            "probability": float(probability),
            "prediction": prediction,
            "outcome": outcome,
        })
    return records


def tensor_to_image(image: torch.Tensor) -> Image.Image:
    array = np.clip(image.detach().cpu().numpy().squeeze() * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="L")


def save_gallery(dataset, records: list[dict], path: Path, columns: int = 4, cell: int = 64) -> None:
    if not records:
        return
    rows = math.ceil(len(records) / columns)
    canvas = Image.new("L", (columns * (cell + 2) + 2, rows * (cell + 2) + 2), 0)
    for position, record in enumerate(records):
        image, _mask, _label = dataset[record["index"]]
        tile = tensor_to_image(image).resize((cell, cell))
        row, column = divmod(position, columns)
        canvas.paste(tile, (2 + column * (cell + 2), 2 + row * (cell + 2)))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def score_conflicts(examples: list[dict], audit: dict, split: str) -> list[dict]:
    scored = []
    by_index = {record["index"]: record for record in examples}
    for group in audit.get("duplicates", []):
        for member in group["members"]:
            if member["split"] != split:
                continue
            example = by_index.get(member["index"])
            if example is None:
                continue
            scored.append({
                **member,
                "digest": group["digest"],
                "conflicting_labels": group.get("conflicting_labels", False),
                "model_probability": example["probability"],
                "model_outcome": example["outcome"],
            })
    return scored


def render_report(report: dict) -> str:
    counts = report["outcome_counts"]
    temperature = report["temperature_scaling"]
    lines = [
        "# Locked-Checkpoint Error Analysis",
        "",
        "This run reloads a saved checkpoint. Weights are not updated, and no new clinical threshold is selected. False negatives are missed malignancies at the default 0.5 cutoff.",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Split: `{report['split']}` ({report['sample_count']} images)",
        f"- True positives: {counts['true_positive']}",
        f"- True negatives: {counts['true_negative']}",
        f"- False positives: {counts['false_positive']}",
        f"- False negatives: {counts['false_negative']}",
        "",
        "## Missed malignancies",
        "",
        "| Index | Malignant probability |",
        "|---:|---:|",
    ]
    for record in report["false_negatives"]:
        lines.append(f"| {record['index']} | {record['probability']:.3f} |")
    if not report["false_negatives"]:
        lines.append("| — | — |")
    lines.extend([
        "",
        "![Missed malignancies](false_negatives.png)" if report["false_negative_gallery"] else "",
        "",
        "## False alarms",
        "",
        "| Index | Malignant probability |",
        "|---:|---:|",
    ])
    for record in report["false_positives"]:
        lines.append(f"| {record['index']} | {record['probability']:.3f} |")
    if not report["false_positives"]:
        lines.append("| — | — |")
    lines.extend([
        "",
        "![False alarms](false_positives.png)" if report["false_positive_gallery"] else "",
        "",
        "## Conflicting split-overlap cases on this split",
        "",
    ])
    if report["conflict_scores"]:
        lines.extend([
            "| Split index | Published remapped label | Model probability | Outcome |",
            "|---:|---:|---:|---|",
        ])
        for record in report["conflict_scores"]:
            lines.append(
                f"| {record['index']} | {record['label']} | {record['model_probability']:.3f} | {record['model_outcome']} |"
            )
        lines.append("")
    else:
        lines.extend(["No exact-image overlap members from the split audit were present on this split.", ""])
    lines.extend([
        "## Temperature scaling (fit on validation only)",
        "",
        f"- Fitted temperature: {temperature['temperature']:.3f}",
        f"- Fit split: `{temperature['fit_split']}`",
        f"- Unscaled Brier / ECE: {temperature['unscaled']['brier_score']:.3f} / {temperature['unscaled']['expected_calibration_error']:.3f}",
        f"- Scaled Brier / ECE: {temperature['scaled']['brier_score']:.3f} / {temperature['scaled']['expected_calibration_error']:.3f}",
        "",
        temperature["warning"],
        "",
        "These galleries and scores are for research review of a public 64px benchmark. They are not a clinical case review.",
        "",
    ])
    return "\n".join(line for line in lines if line is not None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default="data/medmnist")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--split-audit", default="runs/breastmnist_split_audit/split_audit.json")
    parser.add_argument("--allow-test-split", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.split == "test" and not args.allow_test_split:
        parser.error("refusing to score the locked test split without --allow-test-split")

    device = torch.device("cpu")
    model, checkpoint = load_model(Path(args.checkpoint), device)
    dataset = BreastMNISTSplit("val" if args.split == "val" else "test", root=args.root, download=False)
    labels, probabilities, logits = collect_outputs(model, DataLoader(dataset, batch_size=args.batch_size), device)
    examples = bucket_examples(labels, probabilities)
    false_negatives = sorted(
        (record for record in examples if record["outcome"] == "false_negative"),
        key=lambda record: record["probability"],
    )
    false_positives = sorted(
        (record for record in examples if record["outcome"] == "false_positive"),
        key=lambda record: -record["probability"],
    )

    output_directory = Path(args.output)
    output_directory.mkdir(parents=True, exist_ok=True)
    false_negative_gallery = output_directory / "false_negatives.png"
    false_positive_gallery = output_directory / "false_positives.png"
    save_gallery(dataset, false_negatives, false_negative_gallery)
    save_gallery(dataset, false_positives, false_positive_gallery)

    audit = {}
    audit_path = Path(args.split_audit)
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))

    validation = BreastMNISTSplit("val", root=args.root, download=False)
    val_labels, _val_probabilities, val_logits = collect_outputs(
        model, DataLoader(validation, batch_size=args.batch_size), device
    )
    temperature = fit_temperature(np.asarray(val_logits, dtype=np.float64), val_labels)
    scaled_probabilities = apply_temperature(np.asarray(logits, dtype=np.float64), temperature)[:, 1].tolist()

    report = {
        "purpose": "Locked-checkpoint error review and validation-fitted temperature scaling.",
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "sample_count": len(examples),
        "selected_epoch": checkpoint.get("selected_epoch"),
        "weights_updated": False,
        "threshold_selected": False,
        "outcome_counts": {
            outcome: sum(record["outcome"] == outcome for record in examples)
            for outcome in ("true_positive", "true_negative", "false_positive", "false_negative")
        },
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "false_negative_gallery": str(false_negative_gallery) if false_negatives else None,
        "false_positive_gallery": str(false_positive_gallery) if false_positives else None,
        "conflict_scores": score_conflicts(examples, audit, args.split),
        "temperature_scaling": {
            "temperature": temperature,
            "fit_split": "val",
            "evaluated_on_fit_split": args.split == "val",
            "unscaled": {
                "brier_score": brier_score(labels, probabilities),
                "expected_calibration_error": calibration_report(labels, probabilities)["expected_calibration_error"],
            },
            "scaled": {
                "brier_score": brier_score(labels, scaled_probabilities),
                "expected_calibration_error": calibration_report(labels, scaled_probabilities)["expected_calibration_error"],
            },
            "warning": (
                "Temperature was fit on the validation split only. It rescales probabilities for a "
                "calibration diagnostic and is not a new model, threshold, or clinical operating point."
            ),
        },
    }
    (output_directory / "error_analysis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_directory / "RESULTS.md").write_text(render_report(report), encoding="utf-8")
    print(json.dumps({
        "split": args.split,
        "false_negatives": report["outcome_counts"]["false_negative"],
        "false_positives": report["outcome_counts"]["false_positive"],
        "temperature": temperature,
        "conflict_cases": len(report["conflict_scores"]),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
