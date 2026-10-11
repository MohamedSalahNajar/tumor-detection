import unittest
import hashlib
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset

from ai.breastmnist import BreastMNISTSplit, verify_archive
from ai.architecture_search import render_report, select_best_candidate
from ai.calibration import apply_temperature, brier_score, calibration_report, confusion_counts, fit_temperature
from ai.error_analysis import bucket_examples
from ai.split_audit import audit_records
from ai.models import SelfHealingNeuralNetwork, TumorDetector
from ai.models.deep_tumor_net import DeepTumorNet
from ai.seed_sweep import summarize_runs
from ai.train import binary_metrics, run_epoch, save_checkpoint, write_history
from core.imaging.synthetic import make_synthetic_image


class TumorDetectorTests(unittest.TestCase):
    def test_finds_synthetic_region_with_expected_bounds(self):
        image = make_synthetic_image(shape=(64, 64), center=(31, 35), radius=5)
        candidates = TumorDetector().detect(image)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].bbox, (30, 26, 41, 37))

    def test_no_lesion_scene_returns_no_candidates(self):
        image = make_synthetic_image(shape=(64, 64), center=None)

        self.assertEqual(TumorDetector().detect(image), [])

    def test_rejects_non_grayscale_input(self):
        with self.assertRaises(ValueError):
            TumorDetector().detect(np.zeros((12, 12, 3)))

    def test_wrapper_fails_closed_on_invalid_input(self):
        result = SelfHealingNeuralNetwork().analyze(np.zeros((0, 12)))

        self.assertEqual(result.status, "review_required")
        self.assertEqual(result.candidates, ())
        self.assertFalse(result.recovered)

    def test_wrapper_restores_known_detector_after_runtime_failure(self):
        class FailingDetector(TumorDetector):
            def detect(self, image):
                raise RuntimeError("simulated internal failure")

        model = SelfHealingNeuralNetwork(FailingDetector())
        result = model.analyze(np.zeros((12, 12)))

        self.assertEqual(result.status, "review_required")
        self.assertTrue(result.recovered)
        self.assertIs(type(model._detector), TumorDetector)


class DeepTumorNetTests(unittest.TestCase):
    def test_more_blocks_and_channels_increase_model_capacity(self):
        baseline = DeepTumorNet(base_channels=4, blocks_per_stage=1)
        larger = DeepTumorNet(base_channels=8, blocks_per_stage=2)
        baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
        larger_parameters = sum(parameter.numel() for parameter in larger.parameters())

        self.assertGreater(larger_parameters, baseline_parameters)
        self.assertEqual(larger.blocks_per_stage, 2)

    def test_classification_and_segmentation_shapes_and_gradients(self):
        model = DeepTumorNet(base_channels=2)
        images = torch.rand(2, 1, 64, 80)
        outputs = model(images)

        self.assertEqual(outputs["classification"].shape, (2, 2))
        self.assertEqual(outputs["segmentation"].shape, (2, 1, 64, 80))
        (outputs["classification"].sum() + outputs["segmentation"].sum()).backward()
        self.assertIsNotNone(model.stem[0].weight.grad)
        self.assertIsNotNone(model.segmentation_head.weight.grad)

    def test_classification_only_skips_segmentation_head(self):
        model = DeepTumorNet(base_channels=2).eval()
        outputs = model(torch.rand(2, 1, 64, 64), return_segmentation=False)

        self.assertEqual(set(outputs), {"classification"})
        self.assertEqual(outputs["classification"].shape, (2, 2))
        outputs["classification"].sum().backward()
        self.assertIsNotNone(model.stem[0].weight.grad)
        self.assertIsNone(model.segmentation_head.weight.grad)

    def test_rejects_spatial_dimensions_not_divisible_by_sixteen(self):
        model = DeepTumorNet(base_channels=2)

        with self.assertRaisesRegex(ValueError, "dimensions must be divisible by 16"):
            model(torch.rand(2, 1, 63, 64))

    def test_rejects_incompatible_input_channels(self):
        model = DeepTumorNet(base_channels=2)

        with self.assertRaisesRegex(ValueError, "expected 1 input channels"):
            model(torch.rand(2, 3, 64, 64))

    def test_rejects_too_small_base_width(self):
        with self.assertRaisesRegex(ValueError, "base_channels must be at least 2"):
            DeepTumorNet(base_channels=1)

    def test_checkpoint_round_trip_preserves_weights_and_metadata(self):
        model = DeepTumorNet(base_channels=2).eval()
        images = torch.rand(2, 1, 64, 64)
        expected = model(images, return_segmentation=False)["classification"]

        with TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "model.pt"
            save_checkpoint(model, checkpoint_path, {"seed": 19}, epoch=3)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            restored = DeepTumorNet(**checkpoint["model_config"]).eval()
            restored.load_state_dict(checkpoint["model_state_dict"])
            actual = restored(images, return_segmentation=False)["classification"]

        self.assertEqual(checkpoint["selected_epoch"], 3)
        self.assertEqual(checkpoint["run_config"], {"seed": 19})
        torch.testing.assert_close(actual, expected)


class BreastMNISTAdapterTests(unittest.TestCase):
    def test_creates_root_and_maps_malignant_to_positive(self):
        class FakeBreastMNIST(Dataset):
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                (Path(kwargs["root"]) / "breastmnist_64.npz").write_bytes(b"mock archive")

            def __len__(self):
                return 2

            def __getitem__(self, index):
                label = 0 if index == 0 else 1
                return Image.fromarray(np.zeros((64, 64), dtype=np.uint8)), np.array([label])

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "new" / "medmnist"
            mock_digest = hashlib.md5(b"mock archive").hexdigest()
            with patch("medmnist.BreastMNIST", FakeBreastMNIST), patch(
                "ai.breastmnist.BREASTMNIST_64_MD5", mock_digest
            ):
                dataset = BreastMNISTSplit("train", root=str(root), download=False)

            self.assertTrue(root.is_dir())
            self.assertEqual(dataset[0][2], torch.tensor(1))
            self.assertEqual(dataset[1][2], torch.tensor(0))
            self.assertEqual(dataset[0][0].shape, (1, 64, 64))
            self.assertEqual(dataset.archive_md5, mock_digest)

    def test_rejects_bad_archive_checksum(self):
        with TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / "breastmnist_64.npz"
            archive.write_bytes(b"corrupt archive")

            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_archive(archive)

    def test_metrics_include_discrimination_and_bootstrap_intervals(self):
        metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], bootstrap_samples=20)

        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["pr_auc"], 1.0)
        self.assertEqual(set(metrics["confidence_intervals_95"]), {
            "accuracy", "precision", "recall", "specificity", "roc_auc", "pr_auc"
        })

    def test_history_is_parseable_and_written_incrementally(self):
        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "history.csv"
            record = {
                "epoch": 1,
                "train_loss": 0.5,
                "validation_loss": 0.4,
                "validation_metrics": {
                    "accuracy": 0.8,
                    "precision": 0.7,
                    "recall": 0.6,
                    "specificity": 0.9,
                    "roc_auc": 0.85,
                    "pr_auc": 0.75,
                    "confidence_intervals_95": {"roc_auc": [0.7, 0.95]},
                },
            }

            write_history([record], output)

            with output.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["validation_roc_auc"], "0.85")
            self.assertEqual(json.loads(rows[0]["validation_confidence_intervals_95"]), {"roc_auc": [0.7, 0.95]})

    def test_seed_summary_reports_mean_and_variability(self):
        runs = [
            {"seed": seed, "test_metrics": {name: value for name, value in zip(
                ("accuracy", "precision", "recall", "specificity", "roc_auc", "pr_auc"), metrics
            )}}
            for seed, metrics in ((7, (0.8, 0.7, 0.6, 0.9, 0.85, 0.75)), (11, (0.6, 0.5, 0.4, 0.8, 0.65, 0.55)))
        ]

        summary = summarize_runs(runs)

        self.assertAlmostEqual(summary["accuracy"]["mean"], 0.7)
        self.assertAlmostEqual(summary["accuracy"]["standard_deviation"], np.std([0.8, 0.6], ddof=1))
        self.assertEqual(summary["recall"]["per_seed"], {"7": 0.6, "11": 0.4})

    def test_architecture_search_selects_lowest_validation_loss(self):
        candidates = [
            {"name": "wider", "validation_loss": 0.7, "parameter_count": 500, "selected_epoch": 2,
             "validation_metrics": {"accuracy": 0.7, "recall": 0.9, "specificity": 0.5, "roc_auc": 0.8, "pr_auc": 0.7}},
            {"name": "baseline", "validation_loss": 0.4, "parameter_count": 100, "selected_epoch": 3,
             "validation_metrics": {"accuracy": 0.8, "recall": 0.8, "specificity": 0.8, "roc_auc": 0.9, "pr_auc": 0.8}},
        ]
        report = {"selected_candidate": "baseline", "candidates": candidates}

        selected = select_best_candidate(candidates)
        markdown = render_report(report)

        self.assertEqual(selected["name"], "baseline")
        self.assertIn("held-out test split was not evaluated", markdown)
        self.assertIn("| baseline |", markdown)

    def test_positive_class_weight_changes_training_objective(self):
        class FixedLogitModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.tensor([[2.0, 0.0]]))

            def forward(self, images, return_segmentation=False):
                return {"classification": self.logits.expand(images.shape[0], -1)}

        images = torch.zeros((4, 1, 16, 16))
        masks = torch.zeros_like(images)
        labels = torch.tensor([0, 0, 1, 1])
        loader = DataLoader(TensorDataset(images, masks, labels), batch_size=4)

        unweighted_loss, _ = run_epoch(
            FixedLogitModel(), loader, None, torch.device("cpu"), False, 0.0
        )
        weighted_loss, _ = run_epoch(
            FixedLogitModel(), loader, None, torch.device("cpu"), False, 0.0, positive_class_weight=2.7
        )

        self.assertNotAlmostEqual(unweighted_loss, weighted_loss)


class SplitAuditTests(unittest.TestCase):
    def test_detects_cross_split_identical_images(self):
        report = audit_records([
            {"split": "train", "index": 0, "label": 1, "digest": "aaa"},
            {"split": "test", "index": 0, "label": 1, "digest": "aaa"},
            {"split": "val", "index": 1, "label": 0, "digest": "bbb"},
        ])

        self.assertEqual(report["unique_images"], 2)
        self.assertEqual(report["cross_split_duplicate_groups"], 1)
        self.assertEqual(report["within_split_duplicate_groups"], 0)
        self.assertFalse(report["duplicates"][0]["conflicting_labels"])

    def test_flags_identical_pixels_with_opposite_labels(self):
        report = audit_records([
            {"split": "train", "index": 115, "label": 1, "digest": "same"},
            {"split": "test", "index": 76, "label": 0, "digest": "same"},
        ])

        self.assertTrue(report["duplicates"][0]["conflicting_labels"])


class CalibrationTests(unittest.TestCase):
    def test_perfect_scores_have_zero_brier_and_ece(self):
        report = calibration_report([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0], bin_count=4)

        self.assertEqual(brier_score([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0]), 0.0)
        self.assertEqual(report["expected_calibration_error"], 0.0)
        self.assertEqual(confusion_counts([0, 1, 1, 0], [0.1, 0.9, 0.2, 0.8])["false_negatives"], 1)

    def test_confident_wrong_predictions_increase_ece(self):
        report = calibration_report([0, 0, 1, 1], [0.95, 0.95, 0.05, 0.05], bin_count=4)

        self.assertGreater(report["expected_calibration_error"], 0.8)

    def test_higher_temperature_reduces_confidence(self):
        logits = np.array([[0.0, 4.0]])

        self.assertGreater(apply_temperature(logits, 1.0)[0, 1], apply_temperature(logits, 4.0)[0, 1])
        self.assertGreater(fit_temperature(np.array([[8.0, -8.0], [-8.0, 8.0]]), [0, 1]), 0)


class ErrorAnalysisTests(unittest.TestCase):
    def test_buckets_false_negatives_ahead_of_false_positives(self):
        records = bucket_examples([1, 0, 1, 0], [0.2, 0.8, 0.9, 0.1])
        outcomes = [record["outcome"] for record in records]

        self.assertEqual(outcomes, ["false_negative", "false_positive", "true_positive", "true_negative"])
        self.assertEqual(records[0]["index"], 0)


if __name__ == "__main__":
    unittest.main()