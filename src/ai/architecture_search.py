"""Run a bounded validation-only architecture and loss-weight search."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch

from ai.models.deep_tumor_net import DeepTumorNet


CANDIDATES = [
    {
        "name": "control",
        "base_channels": 8,
        "blocks_per_stage": 1,
        "dropout_rate": 0.4,
        "learning_rate": 5e-4,
        "weight_decay": 1e-2,
        "positive_class_weight": 1.0,
    },
    {
        "name": "double_width_depth",
        "base_channels": 16,
        "blocks_per_stage": 2,
        "dropout_rate": 0.5,
        "learning_rate": 3e-4,
        "weight_decay": 3e-2,
        "positive_class_weight": 1.0,
    },
    {
        "name": "width_only",
        "base_channels": 16,
        "blocks_per_stage": 1,
        "dropout_rate": 0.4,
        "learning_rate": 5e-4,
        "weight_decay": 1e-2,
        "positive_class_weight": 1.0,
    },
    {
        "name": "depth_only",
        "base_channels": 8,
        "blocks_per_stage": 2,
        "dropout_rate": 0.5,
        "learning_rate": 3e-4,
        "weight_decay": 3e-2,
        "positive_class_weight": 1.0,
    },
    {
        "name": "malignant_weight",
        "base_channels": 8,
        "blocks_per_stage": 1,
        "dropout_rate": 0.4,
        "learning_rate": 5e-4,
        "weight_decay": 1e-2,
        "positive_class_weight": 2.7,
    },
]


def collect_candidate(name: str, settings: dict, output_directory: Path) -> dict:
    history_path = output_directory / "history.csv"
    with history_path.open(newline="", encoding="utf-8") as file:
        history = list(csv.DictReader(file))
    if not history:
        raise ValueError(f"Candidate {name} has no validation history")

    best_epoch = min(history, key=lambda row: float(row["validation_loss"]))
    checkpoint_path = output_directory / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = DeepTumorNet(**checkpoint["model_config"])
    return {
        "name": name,
        "settings": settings,
        "selected_epoch": int(best_epoch["epoch"]),
        "validation_loss": float(best_epoch["validation_loss"]),
        "validation_metrics": {
            key.removeprefix("validation_"): float(value)
            for key, value in best_epoch.items()
            if key in {
                "validation_accuracy",
                "validation_precision",
                "validation_recall",
                "validation_specificity",
                "validation_roc_auc",
                "validation_pr_auc",
            }
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "run_directory": str(output_directory),
        "test_evaluation": "skipped",
    }


def render_report(report: dict) -> str:
    lines = [
        "# Validation-Only Architecture Search",
        "",
        "This bounded search selected candidates by minimum validation loss. The held-out test split was not evaluated. Results are exploratory, based on one seed and one small validation split; they do not establish generalization or clinical utility.",
        "",
        f"Selected by validation loss: `{report['selected_candidate']}`.",
        "",
        "| Candidate | Parameters | Best epoch | Validation loss | Accuracy | Malignant recall | Specificity | ROC-AUC | PR-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in report["candidates"]:
        metrics = candidate["validation_metrics"]
        lines.append(
            f"| {candidate['name']} | {candidate['parameter_count']:,} | {candidate['selected_epoch']} | "
            f"{candidate['validation_loss']:.3f} | {metrics['accuracy']:.3f} | "
            f"{metrics['recall']:.3f} | {metrics['specificity']:.3f} | "
            f"{metrics['roc_auc']:.3f} | {metrics['pr_auc']:.3f} |"
        )
    lines.extend([
        "",
        "The selected row is only the best result within this small search and validation set. Do not use these results to choose a clinical threshold or model. Keep the locked test set for a pre-specified final evaluation after the research protocol is agreed.",
        "",
    ])
    return "\n".join(lines)


def select_best_candidate(results: list[dict]) -> dict:
    if not results:
        raise ValueError("at least one completed candidate is required")
    return min(results, key=lambda candidate: candidate["validation_loss"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="runs/breastmnist_architecture_search")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.seed < 0 or args.cpu_threads < 1:
        parser.error("epochs, batch-size, and cpu-threads must be positive; seed must be non-negative")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for candidate in CANDIDATES:
        output_directory = output_root / candidate["name"]
        if (output_directory / "history.csv").exists():
            results.append(collect_candidate(candidate["name"], candidate, output_directory))
            continue
        command = [
            sys.executable, "-u", "-m", "ai.train",
            "--dataset", "breastmnist",
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--seed", str(args.seed),
            "--cpu-threads", str(args.cpu_threads),
            "--skip-test",
            "--output", str(output_directory),
        ]
        for flag, key in (
            ("--base-channels", "base_channels"),
            ("--blocks-per-stage", "blocks_per_stage"),
            ("--dropout-rate", "dropout_rate"),
            ("--learning-rate", "learning_rate"),
            ("--weight-decay", "weight_decay"),
            ("--positive-class-weight", "positive_class_weight"),
        ):
            command.extend((flag, str(candidate[key])))
        subprocess.run(command, check=True)
        results.append(collect_candidate(candidate["name"], candidate, output_directory))

    selected = select_best_candidate(results)
    report = {
        "purpose": "Bounded validation-only architecture/hyperparameter comparison",
        "selection_metric": "minimum validation loss",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "shared_test_split_evaluated": False,
        "selected_candidate": selected["name"],
        "candidates": results,
    }
    (output_root / "search_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_root / "RESULTS.md").write_text(render_report(report), encoding="utf-8")
    print(json.dumps({"selected_candidate": selected["name"], "validation_loss": selected["validation_loss"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()