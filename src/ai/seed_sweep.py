"""Repeat the fixed BreastMNIST baseline across initialization seeds."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def summarize_runs(runs: list[dict]) -> dict:
    metric_names = ("accuracy", "precision", "recall", "specificity", "roc_auc", "pr_auc")
    summary = {}
    for name in metric_names:
        values = [run["test_metrics"][name] for run in runs]
        summary[name] = {
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "per_seed": {str(run["seed"]): run["test_metrics"][name] for run in runs},
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23])
    parser.add_argument("--output-root", default="runs/breastmnist_seed_sweep")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be unique non-negative integers")
    if args.epochs < 1 or args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("epochs, batch-size, and cpu-threads must be positive")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in args.seeds:
        run_dir = output_root / f"seed_{seed}"
        metrics_path = run_dir / "test_metrics.json"
        if not metrics_path.exists():
            command = [
                sys.executable,
                "-u",
                "-m",
                "ai.train",
                "--dataset",
                "breastmnist",
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--learning-rate",
                str(args.learning_rate),
                "--base-channels",
                str(args.base_channels),
                "--cpu-threads",
                str(args.cpu_threads),
                "--seed",
                str(seed),
                "--output",
                str(run_dir),
            ]
            subprocess.run(command, check=True)
        test_result = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
        runs.append({
            "seed": seed,
            "selected_epoch": checkpoint["selected_epoch"],
            "test_loss": test_result["test_loss"],
            "test_metrics": test_result["test_metrics"],
            "run_config": run_config,
        })

    report = {
        "purpose": "Training-seed variability on one fixed BreastMNIST split; not external validation.",
        "seeds": args.seeds,
        "shared_test_split_size": 156,
        "runs": runs,
        "summary": summarize_runs(runs),
        "warning": "These runs reuse the same tiny test set and do not establish clinical validity or independent generalization.",
    }
    (output_root / "sweep_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()