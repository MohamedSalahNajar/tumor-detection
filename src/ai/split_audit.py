"""Audit official BreastMNIST splits for exact image overlap.

This cannot detect same-patient leakage without patient identifiers. It only
reports identical pixel arrays within and across the published splits.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ai.breastmnist import BreastMNISTSplit


def image_digest(image: torch.Tensor) -> str:
    pixels = image.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(pixels).hexdigest()


def tensor_to_image(image: torch.Tensor) -> Image.Image:
    array = np.clip(image.detach().cpu().numpy().squeeze() * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="L")


def export_duplicate_images(root: str, duplicates: list[dict], output_directory: Path) -> list[str]:
    saved = []
    datasets: dict[str, Dataset] = {}
    for group_index, group in enumerate(duplicates):
        for member in group["members"]:
            split = member["split"]
            if split not in datasets:
                datasets[split] = BreastMNISTSplit(split, root=root, download=False)
            image, _mask, _label = datasets[split][member["index"]]
            name = f"duplicate_{group_index}_{split}_{member['index']}_label{member['label']}.png"
            tensor_to_image(image).save(output_directory / name)
            saved.append(name)
    return saved


def collect_split_records(dataset: Dataset, split: str) -> list[dict]:
    records = []
    for index in range(len(dataset)):
        image, _mask, label = dataset[index]
        records.append({
            "split": split,
            "index": index,
            "label": int(label),
            "digest": image_digest(image),
        })
    return records


def find_duplicate_groups(records: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["digest"]].append(record)
    duplicates = []
    for digest, members in grouped.items():
        if len(members) < 2:
            continue
        splits = sorted({member["split"] for member in members})
        labels = sorted({member["label"] for member in members})
        duplicates.append({
            "digest": digest,
            "count": len(members),
            "cross_split": len(splits) > 1,
            "conflicting_labels": len(labels) > 1,
            "splits": splits,
            "labels": labels,
            "members": members,
        })
    duplicates.sort(key=lambda group: (-int(group["cross_split"]), -group["count"], group["digest"]))
    return duplicates


def audit_records(records: list[dict]) -> dict:
    split_sizes: dict[str, int] = defaultdict(int)
    for record in records:
        split_sizes[record["split"]] += 1
    duplicates = find_duplicate_groups(records)
    cross_split = [group for group in duplicates if group["cross_split"]]
    within_split = [group for group in duplicates if not group["cross_split"]]
    return {
        "purpose": "Exact-image overlap audit for published BreastMNIST splits; not a patient-level leak check.",
        "split_sizes": dict(split_sizes),
        "unique_images": len({record["digest"] for record in records}),
        "duplicate_groups": len(duplicates),
        "cross_split_duplicate_groups": len(cross_split),
        "within_split_duplicate_groups": len(within_split),
        "duplicates": duplicates,
        "warning": (
            "Identical pixels can indicate label leakage between splits, but distinct images "
            "from the same person would not be detected without patient identifiers."
        ),
    }


def render_report(report: dict) -> str:
    lines = [
        "# BreastMNIST Split Integrity Audit",
        "",
        "This check hashes official 64px images and reports identical pixel arrays. It cannot audit same-patient leakage because BreastMNIST does not provide patient or site identifiers.",
        "",
        f"- Unique images: {report['unique_images']}",
        f"- Duplicate groups: {report['duplicate_groups']}",
        f"- Cross-split duplicate groups: {report['cross_split_duplicate_groups']}",
        f"- Within-split duplicate groups: {report['within_split_duplicate_groups']}",
        f"- Duplicate groups with conflicting labels: {sum(1 for group in report['duplicates'] if group.get('conflicting_labels'))}",
        "",
        "Split sizes: " + ", ".join(f"{name}={size}" for name, size in report["split_sizes"].items()),
        "",
    ]
    if report["duplicates"]:
        if report.get("exported_images"):
            lines.extend([
                "Exported images: " + ", ".join(f"`{name}`" for name in report["exported_images"]),
                "",
            ])
        lines.extend([
            "| Digest prefix | Count | Cross-split | Conflicting labels | Splits | Remapped labels |",
            "|---|---:|---|---|---|---|",
        ])
        for group in report["duplicates"][:20]:
            lines.append(
                f"| `{group['digest'][:12]}` | {group['count']} | {group['cross_split']} | "
                f"{group.get('conflicting_labels', False)} | {', '.join(group['splits'])} | "
                f"{', '.join(str(label) for label in group['labels'])} |"
            )
        lines.append("")
        if any(group.get("conflicting_labels") for group in report["duplicates"]):
            lines.extend([
                "Remapped labels: `1` is malignant and `0` is normal/benign. Conflicting labels on identical pixels mean the same 64px array is treated as both classes across splits. This can come from a true duplicate with inconsistent labels, or from different source images that became identical after 64px preprocessing.",
                "",
            ])
    else:
        lines.extend(["No identical pixel arrays were found within or across the published splits.", ""])
    lines.append(report["warning"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/medmnist")
    parser.add_argument("--output", default="runs/breastmnist_split_audit")
    args = parser.parse_args()

    records = []
    for split in ("train", "val", "test"):
        records.extend(collect_split_records(BreastMNISTSplit(split, root=args.root, download=False), split))
    report = audit_records(records)
    output_directory = Path(args.output)
    output_directory.mkdir(parents=True, exist_ok=True)
    report["exported_images"] = export_duplicate_images(args.root, report["duplicates"], output_directory)
    (output_directory / "split_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_directory / "RESULTS.md").write_text(render_report(report), encoding="utf-8")
    print(json.dumps({
        "unique_images": report["unique_images"],
        "cross_split_duplicate_groups": report["cross_split_duplicate_groups"],
        "within_split_duplicate_groups": report["within_split_duplicate_groups"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
