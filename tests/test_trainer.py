"""Fast, local unit tests. No dataset extraction and no training runs needed.

From AI403 root:
    python -m unittest discover -s tests -p "test_trainer.py" -v
"""

import unittest
import torch
from torch import nn

from src.ctc_utils import (BLANK_ID, collate_batch, ctc_forward,
                           encode_label, min_ctc_steps, greedy_decode)
from src.evaluate import aligned_operations, metrics_from_predictions
from src.models.registry import build_model


class TestSharedTrainer(unittest.TestCase):
    def test_digit_encoding_and_repeated_ctc(self):
        self.assertEqual(encode_label("0052"), [1, 1, 6, 3])
        self.assertEqual(min_ctc_steps("5528"), 5)
        self.assertEqual(BLANK_ID, 0)

    def test_padding_preserves_widths(self):
        samples = [
            {"image": torch.ones(1, 48, 80), "width": 80, "label": "52", "filename": "a_car_1.png"},
            {"image": torch.ones(1, 48, 117), "width": 117, "label": "5528", "filename": "b_car_2.png"},
        ]
        batch = collate_batch(samples)
        self.assertEqual(tuple(batch["images"].shape), (2, 1, 48, 117))
        self.assertEqual(batch["widths"].tolist(), [80, 117])
        self.assertEqual(batch["targets"].tolist(), [6, 3, 6, 6, 3, 9])
        self.assertEqual(batch["target_lengths"].tolist(), [2, 4])
        self.assertTrue(torch.all(batch["images"][0, :, :, 80:] == 1))

    def test_model_contract_ctc_loss_and_backward(self):
        model = build_model("baseline", dropout=0.3)
        samples = [
            {"image": torch.rand(1, 48, 80), "width": 80, "label": "52", "filename": "a_car_1.png"},
            {"image": torch.rand(1, 48, 117), "width": 117, "label": "5528", "filename": "b_car_2.png"},
        ]
        batch = collate_batch(samples)
        criterion = nn.CTCLoss(blank=0, zero_infinity=False)
        loss, logits, lengths = ctc_forward(model, batch, criterion, torch.device("cpu"))
        self.assertEqual(tuple(logits.shape), (2, 117, 11))
        self.assertEqual(lengths.tolist(), [80, 117])
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(model.classifier.weight.grad)
        self.assertTrue(torch.isfinite(model.classifier.weight.grad).all().item())
        self.assertEqual(model.dropout.p, 0.3)

    def test_ctc_rejects_impossible_length(self):
        class ShortModel(nn.Module):
            def forward(self, images, widths):
                return torch.zeros(images.size(0), 4, 11, requires_grad=True), torch.tensor([4])
        batch = collate_batch([{"image": torch.ones(1, 48, 80), "width": 80,
                               "label": "5528", "filename": "x.png"}])
        with self.assertRaisesRegex(ValueError, "CTC time steps"):
            ctc_forward(ShortModel(), batch, nn.CTCLoss(blank=0), torch.device("cpu"))

    def test_greedy_ctc_collapse(self):
        # blank=0; digit 5 = 6; digit 2 = 3; digit 8 = 9.
        ids = [6, 6, 0, 6, 3, 9, 0]
        logits = torch.full((1, len(ids), 11), -10.0)
        for t, value in enumerate(ids):
            logits[0, t, value] = 10.0
        self.assertEqual(greedy_decode(logits, torch.tensor([len(ids)])), ["5528"])

    def test_per_digit_metrics_and_alignment(self):
        operations = aligned_operations("2250", "2350")
        self.assertEqual(sum(op == "substitution" for op, a, b in operations), 1)
        summary, per_digit, confusions, rows = metrics_from_predictions([
            {"filename": "a_car_1.png", "label": "2250", "prediction": "2350"}
        ])
        self.assertEqual(summary["cer"], 0.25)
        self.assertEqual(summary["exact_match"], 0.0)
        self.assertEqual(confusions[("2", "3")], 1)
        d2 = next(row for row in per_digit if row["digit"] == "2")
        self.assertEqual(d2["support"], 2)
        self.assertEqual(d2["fn"], 1)
        self.assertEqual(rows[0]["source"], "CAR-A")


if __name__ == "__main__":
    unittest.main()
