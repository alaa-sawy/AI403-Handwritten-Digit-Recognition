import unittest

import torch
import torch.nn as nn

from src.models.advanced import AdvancedCTC


class TestAdvancedCTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.images = torch.randn(2, 1, 48, 117)
        cls.widths = [80, 117]

    def test_forward_shapes(self):
        """Test output shapes for both architectures."""

        for multiscale in [True, False]:
            model = AdvancedCTC(
                dropout=0.1,
                multiscale=multiscale,
            )
            model.eval()

            with torch.no_grad():
                logits, output_lengths = model(
                    self.images,
                    self.widths,
                )

            self.assertEqual(
                tuple(logits.shape),
                (2, 117, 11),
            )

            self.assertEqual(
                output_lengths.tolist(),
                [80, 117],
            )

    def test_ctc_loss_and_backward(self):
        """Test CTC loss with repeated digits and gradients."""

        model = AdvancedCTC(dropout=0.1)
        model.train()

        logits, output_lengths = model(
            self.images,
            self.widths,
        )

        targets = torch.tensor(
            [5, 5, 2, 8, 1, 3, 3, 4],
            dtype=torch.long,
        )

        target_lengths = torch.tensor([4, 4])

        log_probs = logits.log_softmax(2).transpose(0, 1)

        criterion = nn.CTCLoss(
            blank=0,
            zero_infinity=True,
        )

        loss = criterion(
            log_probs,
            targets,
            output_lengths,
            target_lengths,
        )

        self.assertTrue(torch.isfinite(loss).item())

        loss.backward()

        modules = [
            model.stage1,
            model.stage2,
            model.stage3,
            model.bigru,
            model.attention,
            model.classifier,
        ]

        for module in modules:
            gradients = [
                parameter.grad
                for parameter in module.parameters()
                if parameter.requires_grad
            ]

            self.assertTrue(
                any(
                    gradient is not None
                    and torch.isfinite(gradient).all()
                    and gradient.abs().sum() > 0
                    for gradient in gradients
                ),
                f"No valid gradients in {module.__class__.__name__}",
            )

    def test_both_architectures(self):
        """Test multiscale and no-multiscale configurations."""

        for multiscale in [True, False]:
            model = AdvancedCTC(
                dropout=0.1,
                multiscale=multiscale,
            )

            self.assertEqual(model.multiscale, multiscale)

            logits, lengths = model(
                self.images,
                self.widths,
            )

            self.assertEqual(logits.shape[-1], 11)
            self.assertEqual(lengths.tolist(), self.widths)

    def test_invalid_dropout(self):
        """Reject invalid dropout values."""

        for dropout in [-0.1, 1.0, 1.5]:
            with self.assertRaises(ValueError):
                AdvancedCTC(dropout=dropout)

    def test_invalid_inputs(self):
        """Reject invalid image dimensions and widths."""

        model = AdvancedCTC(dropout=0.1)
        model.eval()

        with self.assertRaises(ValueError):
            model(torch.randn(2, 48, 117), [80, 117])

        with self.assertRaises(ValueError):
            model(torch.randn(2, 3, 48, 117), [80, 117])

        with self.assertRaises(ValueError):
            model(torch.randn(2, 1, 32, 117), [80, 117])

        with self.assertRaises(ValueError):
            model(self.images, [80])

        with self.assertRaises(ValueError):
            model(self.images, [0, 117])

        with self.assertRaises(ValueError):
            model(self.images, [80, 118])


if __name__ == "__main__":
    unittest.main()