"""M1 - Part 5.1: tests for the fixed ORAND train/val Dataset.

Run from the AI403 project folder:
    python -m unittest discover -s tests -p "test_data.py" -v
"""

import unittest

import torch

from src.datasets.dataset import DigitStringDataset


class TestDigitStringDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = DigitStringDataset("train")
        cls.val = DigitStringDataset("val")

    def test_split_sizes(self):
        self.assertEqual(len(self.train), 4507)
        self.assertEqual(len(self.val), 501)

    def test_train_val_do_not_overlap(self):
        # A sample is identified by its source and filename.
        train_names = {(s["source"], s["filename"]) for s in self.train.samples}
        val_names = {(s["source"], s["filename"]) for s in self.val.samples}
        self.assertFalse(train_names & val_names)

        # Exact duplicate images must not appear on both sides.
        train_hashes = {s["sha256"] for s in self.train.samples}
        val_hashes = {s["sha256"] for s in self.val.samples}
        self.assertFalse(train_hashes & val_hashes)

    def test_all_images_and_labels(self):
        checked = 0

        for dataset in (self.train, self.val):
            for index in range(len(dataset)):
                sample = dataset[index]
                image = sample["image"]
                label = sample["label"]
                width = sample["width"]
                filename = sample["filename"]

                self.assertTrue(
                    label.isascii() and label.isdigit(),
                    f"Invalid label: {filename} -> {label}",
                )
                self.assertTrue(2 <= len(label) <= 8, filename)

                self.assertEqual(image.ndim, 3, filename)
                self.assertEqual(image.shape[0], 1, filename)
                self.assertEqual(image.shape[1], 48, filename)
                self.assertEqual(image.shape[2], width, filename)
                self.assertGreater(width, 0, filename)
                self.assertEqual(image.dtype, torch.float32, filename)
                self.assertTrue(torch.isfinite(image).all().item(), filename)
                self.assertGreaterEqual(float(image.min()), 0.0, filename)
                self.assertLessEqual(float(image.max()), 1.0, filename)

                checked += 1
                if checked % 1000 == 0:
                    print(f"Checked {checked} images...")

        self.assertEqual(checked, 5008)
        print(f"PASS: all {checked} train/val images passed checks.")

    def test_validation_sample_is_repeatable(self):
        first = self.val[0]
        second = self.val[0]
        self.assertEqual(first["label"], second["label"])
        self.assertTrue(torch.equal(first["image"], second["image"]))


if __name__ == "__main__":
    unittest.main()
