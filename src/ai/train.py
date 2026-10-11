"""Train/evaluate the research model on BreastMNIST or toy images."""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

from ai.data_generator import generate_dataset
from ai.dataset import SyntheticTumorDataset
from ai.models.deep_tumor_net import DeepTumorNet
from ai.breastmnist import BreastMNISTSplit


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def binary_metrics(
    labels: list[int], probabilities: list[float], bootstrap_samples: int = 500, seed: int = 7
) -> dict[str, object]:
    label_array = np.asarray(labels, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    predictions = (probability_array >= 0.5).astype(np.int64)
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    metrics = {
        "accuracy": (tp + tn) / max(len(labels), 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
    }
    if np.unique(label_array).size < 2:
        metrics.update({"roc_auc": None, "pr_auc": None, "confidence_intervals_95": {}})
        return metrics

    metrics["roc_auc"] = float(roc_auc_score(label_array, probability_array))
    metrics["pr_auc"] = float(average_precision_score(label_array, probability_array))
    positive_indices = np.flatnonzero(label_array == 1)
    negative_indices = np.flatnonzero(label_array == 0)
    random = np.random.default_rng(seed)
    bootstrap_values: dict[str, list[float]] = {
        key: [] for key in ("accuracy", "precision", "recall", "specificity", "roc_auc", "pr_auc")
    }
    for _ in range(bootstrap_samples):
        sampled = np.concatenate((
            random.choice(positive_indices, len(positive_indices), replace=True),
            random.choice(negative_indices, len(negative_indices), replace=True),
        ))
        sampled_labels = label_array[sampled]
        sampled_probabilities = probability_array[sampled]
        sampled_predictions = (sampled_probabilities >= 0.5).astype(np.int64)
        sampled_tp = int(np.sum((sampled_labels == 1) & (sampled_predictions == 1)))
        sampled_tn = int(np.sum((sampled_labels == 0) & (sampled_predictions == 0)))
        sampled_fp = int(np.sum((sampled_labels == 0) & (sampled_predictions == 1)))
        sampled_fn = int(np.sum((sampled_labels == 1) & (sampled_predictions == 0)))
        values = {
            "accuracy": (sampled_tp + sampled_tn) / len(sampled),
            "precision": sampled_tp / max(sampled_tp + sampled_fp, 1),
            "recall": sampled_tp / max(sampled_tp + sampled_fn, 1),
            "specificity": sampled_tn / max(sampled_tn + sampled_fp, 1),
            "roc_auc": float(roc_auc_score(sampled_labels, sampled_probabilities)),
            "pr_auc": float(average_precision_score(sampled_labels, sampled_probabilities)),
        }
        for key, value in values.items():
            bootstrap_values[key].append(value)
    metrics["confidence_intervals_95"] = {
        key: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        for key, values in bootstrap_values.items()
    }
    return metrics


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    training: bool,
    segmentation_weight: float,
    positive_class_weight: float = 1.0,
) -> tuple[float, dict[str, float]]:
    model.train(training)
    total_loss = 0.0
    labels: list[int] = []
    probabilities: list[float] = []
    seen = 0
    for images, masks, targets in loader:
        images, masks, targets = images.to(device), masks.to(device), targets.to(device)
        with torch.set_grad_enabled(training):
            outputs = model(images, return_segmentation=segmentation_weight > 0)
            class_weights = torch.tensor([1.0, positive_class_weight], device=device)
            classification_loss = F.cross_entropy(
                outputs["classification"],
                targets,
                weight=class_weights,
            )
            loss = classification_loss
            if segmentation_weight > 0:
                segmentation_loss = F.binary_cross_entropy_with_logits(outputs["segmentation"], masks)
                loss = loss + segmentation_weight * segmentation_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch_size = images.size(0)
        total_loss += float(loss.detach()) * batch_size
        labels.extend(targets.detach().cpu().tolist())
        probabilities.extend(outputs["classification"].softmax(1)[:, 1].detach().cpu().tolist())
        seen += batch_size
    return total_loss / max(seen, 1), binary_metrics(labels, probabilities)


def write_history(history: list[dict], output_path: Path) -> None:
    temporary_path = output_path.with_suffix(".csv.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "validation_loss",
                "validation_accuracy",
                "validation_precision",
                "validation_recall",
                "validation_specificity",
                "validation_roc_auc",
                "validation_pr_auc",
                "validation_confidence_intervals_95",
            ],
        )
        writer.writeheader()
        for record in history:
            writer.writerow({
                "epoch": record["epoch"],
                "train_loss": record["train_loss"],
                "validation_loss": record["validation_loss"],
                **{f"validation_{key}": value for key, value in record["validation_metrics"].items()},
                "validation_confidence_intervals_95": json.dumps(
                    record["validation_metrics"]["confidence_intervals_95"]
                ),
            })
    temporary_path.replace(output_path)


def save_checkpoint(model, output_path: Path, run_config: dict, epoch: int) -> None:
    temporary_path = output_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "in_channels": model.in_channels,
                "num_classes": model.num_classes,
                "base_channels": model.base_channels,
                "blocks_per_stage": model.blocks_per_stage,
                "dropout_rate": model.dropout_rate,
            },
            "run_config": run_config,
            "selected_epoch": epoch,
        },
        temporary_path,
    )
    temporary_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/synthetic_images")
    parser.add_argument("--dataset", choices=("synthetic", "breastmnist"), default="breastmnist")
    parser.add_argument("--output", default="runs/breastmnist")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--blocks-per-stage", type=int, default=1)
    parser.add_argument("--dropout-rate", type=float, default=0.4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--positive-class-weight", type=float, default=1.0)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--skip-test", action="store_true", help="Do not evaluate the held-out test split")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.samples < 10:
        parser.error("epochs/batch-size must be positive and samples must be at least 10")
    if args.base_channels < 2 or args.blocks_per_stage < 1:
        parser.error("base-channels must be at least 2 and blocks-per-stage must be positive")
    if not 0.0 <= args.dropout_rate < 1.0 or args.weight_decay < 0 or args.positive_class_weight <= 0:
        parser.error("dropout-rate must be in [0, 1), weight-decay non-negative, and positive-class-weight positive")

    seed_everything(args.seed)
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    if args.dataset == "breastmnist":
        data_root = Path("data/medmnist")
        training = BreastMNISTSplit("train", root="data/medmnist")
        validation = BreastMNISTSplit("val", root="data/medmnist")
        testing = BreastMNISTSplit("test", root="data/medmnist")
        segmentation_weight = 0.0
    else:
        data_root = Path(args.data)
        if not (data_root / "manifest.csv").exists():
            generate_dataset(data_root, count=args.samples, seed=args.seed)
        dataset = SyntheticTumorDataset(data_root)
        indices = np.random.default_rng(args.seed).permutation(len(dataset)).tolist()
        test_size = max(1, int(len(indices) * 0.1))
        validation_size = max(1, int(len(indices) * 0.1))
        testing = Subset(dataset, indices[:test_size])
        validation = Subset(dataset, indices[test_size : test_size + validation_size])
        training = Subset(dataset, indices[test_size + validation_size :])
        segmentation_weight = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_generator = torch.Generator().manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, generator=loader_generator)
    validation_loader = DataLoader(validation, batch_size=args.batch_size)
    test_loader = None if args.skip_test else DataLoader(testing, batch_size=args.batch_size)
    model = DeepTumorNet(
        base_channels=args.base_channels,
        blocks_per_stage=args.blocks_per_stage,
        dropout_rate=args.dropout_rate,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_sizes = {"train": len(training), "validation": len(validation), "test": len(testing)}
    run_config = {
        **vars(args),
        "data": str(data_root),
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "split_sizes": split_sizes,
        "positive_class": "malignant" if args.dataset == "breastmnist" else "synthetic lesion present",
        "deterministic_cudnn": True,
        "test_evaluation_enabled": not args.skip_test,
    }
    if args.dataset == "breastmnist":
        with (data_root / "breastmnist_64.npz").open("rb") as archive:
            run_config["dataset_archive_md5"] = hashlib.file_digest(archive, "md5").hexdigest()
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    history = []
    best_validation_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model, train_loader, optimizer, device, True, segmentation_weight, args.positive_class_weight
        )
        val_loss, val_metrics = run_epoch(
            model, validation_loader, optimizer, device, False, segmentation_weight, args.positive_class_weight
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_metrics": train_metrics,
            "validation_loss": val_loss,
            "validation_metrics": val_metrics,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        write_history(history, output_dir / "history.csv")
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            save_checkpoint(model, output_dir / "model.pt", run_config, epoch)

    if best_state is not None:
        model.load_state_dict(best_state)
    if test_loader is not None:
        test_loss, test_metrics = run_epoch(
            model, test_loader, optimizer, device, False, segmentation_weight, args.positive_class_weight
        )
        test_result = {"test_loss": test_loss, "test_metrics": test_metrics}
        print(json.dumps(test_result), flush=True)
    else:
        test_result = {"status": "skipped", "reason": "--skip-test was set; use validation metrics for experiments"}

    (output_dir / "test_metrics.json").write_text(json.dumps(test_result, indent=2) + "\n", encoding="utf-8")
    (output_dir / "limitations.txt").write_text(
        f"Dataset: {args.dataset}. Metrics do not indicate clinical accuracy. "
        "Not for patient care, diagnosis, triage, treatment, or monitoring.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
