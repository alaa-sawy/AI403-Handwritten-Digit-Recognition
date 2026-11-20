"""M5 demo adapter tests: no dataset access, tuning, or pretrained downloads."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from src.ctc_utils import greedy_decode
from src.inference import (
    MODEL_CHECKPOINTS, checkpoint_for, load_checkpoint_model, predict_image,
)
from src.models.registry import build_model


class TestM5Integration(unittest.TestCase):
    def test_all_seven_conditions_and_three_seeds(self):
        self.assertEqual(len(MODEL_CHECKPOINTS), 7)
        paths = set()
        for name, (model, _) in MODEL_CHECKPOINTS.items():
            for seed in (7, 21, 42):
                path = checkpoint_for(name, seed)
                self.assertEqual(path.name, "best.pt")
                self.assertIn(f"seed_{seed}", str(path))
                paths.add(str(path))
        self.assertEqual(len(paths), 21)
        self.assertEqual(MODEL_CHECKPOINTS["Modern Augmentation OFF"][0], "modern")

    def test_invalid_selection(self):
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            checkpoint_for("unknown", 7)
        with self.assertRaisesRegex(ValueError, "Seed"):
            checkpoint_for("Classic Pretrained", 999)

    def test_missing_checkpoint(self):
        with self.assertRaises(FileNotFoundError):
            load_checkpoint_model("baseline", Path(tempfile.gettempdir()) / "missing_m5_ai403.pt")

    def test_ctc_leading_zeros_and_repeated_digits(self):
        ids = [1, 0, 1, 6, 3, 1]  # 00520
        logits = torch.full((1, len(ids), 11), -9.0)
        for i, token in enumerate(ids):
            logits[0, i, token] = 9.0
        self.assertEqual(greedy_decode(logits, torch.tensor([len(ids)])), ["00520"])

    def test_preprocessing_and_real_forward_with_synthetic_weights(self):
        model = build_model("baseline", dropout=0.1).eval()
        im = Image.new("RGB", (120, 30), "white")
        prediction, processed, width = predict_image(im, model, "cpu")
        self.assertEqual(processed.mode, "L")
        self.assertEqual(processed.height, 48)
        self.assertEqual(width, 192)
        self.assertTrue(isinstance(prediction, str))
        self.assertTrue(not prediction or prediction.isascii() and prediction.isdigit())

    def test_checkpoint_round_trip_baseline(self):
        model = build_model("baseline", dropout=0.2)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "best.pt"
            torch.save({"config": {"model": "baseline", "dropout": 0.2, "seed": 7},
                        "model_state_dict": model.state_dict()}, path)
            loaded, checkpoint, device = load_checkpoint_model("baseline", path, "cpu")
            self.assertEqual(str(device), "cpu")
            self.assertFalse(loaded.training)
            self.assertEqual(checkpoint["config"]["seed"], 7)
            self.assertTrue(torch.equal(loaded.classifier.weight, model.classifier.weight))
            with self.assertRaisesRegex(ValueError, "Wrong model"):
                load_checkpoint_model("modern", path, "cpu")

    def test_strict_state_dict_rejects_incompatible_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "best.pt"
            torch.save({"config": {"model": "baseline", "dropout": 0.1, "seed": 7},
                        "model_state_dict": {"fake": torch.ones(1)}}, path)
            with self.assertRaises(RuntimeError):
                load_checkpoint_model("baseline", path, "cpu")

    def test_classic_pretrained_restoration_does_not_fetch_imagenet(self):
        # Final weights are architecture-compatible; do not request ResNet's default weights.
        from src.models import classic
        with patch.object(classic, "resnet18", wraps=classic.resnet18) as mock_builder:
            model = build_model("classic_scratch", dropout=0.1)
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "best.pt"
                torch.save({"config": {"model": "classic_pretrained", "dropout": 0.1,
                                       "seed": 7}, "model_state_dict": model.state_dict()}, path)
                restored, _, _ = load_checkpoint_model("classic_pretrained", path, "cpu")
                self.assertFalse(restored.training)
            self.assertTrue(all(call.kwargs.get("weights") is None for call in mock_builder.call_args_list))

    def test_modern_augmentation_mismatch_is_detected(self):
        model = build_model("baseline", dropout=0.1)  # Flag checked before architecture construction.
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "best.pt"
            torch.save({"config": {"model": "modern", "dropout": 0.1,
                                   "seed": 7, "augmentation": False},
                        "model_state_dict": model.state_dict()}, path)
            with patch("src.inference.checkpoint_for", return_value=path):
                with self.assertRaisesRegex(ValueError, "augmentation"):
                    load_checkpoint_model("modern", path, "cpu", model_label="Modern Augmentation ON")


if __name__ == "__main__":
    unittest.main()
