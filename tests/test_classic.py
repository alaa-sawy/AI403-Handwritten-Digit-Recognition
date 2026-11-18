import unittest

import torch
from torchvision.models import resnet18, ResNet18_Weights

from src.models.classic import ClassicCTC
from src.models.registry import build_model
from torch import nn
from src.ctc_utils import ctc_forward
from src.train import train_one_epoch


class TestClassicCTC(unittest.TestCase):

    def setUp(self):
        self.model = ClassicCTC(
            dropout=0.1,
            pretrained=False,
        )
        self.model.eval()

        self.images = torch.randn(2, 1, 48, 117)
        self.widths = torch.tensor([80, 117])

    def test_forward_shapes(self):
        with torch.no_grad():
            logits, output_lengths = self.model(
                self.images,
                self.widths,
            )

        self.assertEqual(tuple(logits.shape), (2, 117, 11))
        self.assertEqual(output_lengths.tolist(), [80, 117])

    def test_backward(self):
        self.model.train()

        logits, _ = self.model(self.images, self.widths)
        loss = logits.mean()
        loss.backward()

        self.assertIsNotNone(
            self.model.classifier.weight.grad
        )
        self.assertIsNotNone(
            self.model.gru.weight_ih_l0.grad
        )
        self.assertIsNotNone(
            self.model.backbone.layer4[1].conv2.weight.grad
        )

    def test_scratch_registry(self):
        model = build_model(
            "classic_scratch",
            dropout=0.1,
        )

        self.assertIsInstance(model, ClassicCTC)

    def test_pretrained_registry(self):
        model = build_model(
            "classic_pretrained",
            dropout=0.1,
        )

        self.assertIsInstance(model, ClassicCTC)

    def test_pretrained_forward(self):
        model = build_model(
            "classic_pretrained",
            dropout=0.1,
        )
        model.eval()

        with torch.no_grad():
            logits, output_lengths = model(
                self.images,
                self.widths,
            )

        self.assertEqual(tuple(logits.shape), (2, 117, 11))
        self.assertEqual(output_lengths.tolist(), [80, 117])

    def test_invalid_dropout(self):
        with self.assertRaises(ValueError):
            ClassicCTC(dropout=1.5, pretrained=False)

    def test_pretrained_weights_loaded(self):
        model = build_model(
            "classic_pretrained",
            dropout=0.1,
        )

        original = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        self.assertTrue(
            torch.equal(
                model.backbone.layer1[0].conv1.weight,
                original.layer1[0].conv1.weight,
            )
        )

    def test_pretrained_weights_loaded(self):
        model = build_model(
            "classic_pretrained",
            dropout=0.1,
        )

        original = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        self.assertTrue(
            torch.equal(
                model.backbone.layer1[0].conv1.weight,
                original.layer1[0].conv1.weight,
            )
        )
    def test_ctc_forward_integration(self):
        batch = {
            "images": torch.randn(2, 1, 48, 117),
            "widths": torch.tensor([80, 117]),
            "labels": ["12", "34"],
            "targets": torch.tensor([2, 3, 4, 5]),
            "target_lengths": torch.tensor([2, 2]),
        }

        loss_fn = nn.CTCLoss(blank=0)

        loss, logits, output_lengths = ctc_forward(
            self.model,
            batch,
            loss_fn,
            torch.device("cpu"),
        )

        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(tuple(logits.shape), (2, 117, 11))
        self.assertEqual(output_lengths.tolist(), [80, 117])

        loss.backward()

        self.assertIsNotNone(
            self.model.classifier.weight.grad
        )
    
    def test_shared_trainer_one_epoch(self):
        batch = {
            "images": torch.randn(2, 1, 48, 117),
            "widths": torch.tensor([80, 117]),
            "labels": ["12", "34"],
            "targets": torch.tensor([2, 3, 4, 5]),
            "target_lengths": torch.tensor([2, 2]),
        }

        loss_fn = nn.CTCLoss(blank=0)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.0003,
        )

        config = {
            "gradient_clip": 5.0,
        }

        original_weights = self.model.classifier.weight.detach().clone()

        avg_loss = train_one_epoch(
            model=self.model,
            loader=[batch],
            loss_fn=loss_fn,
            optimizer=optimizer,
            config=config,
            device=torch.device("cpu"),
        )

        self.assertTrue(torch.isfinite(torch.tensor(avg_loss)).item())

        self.assertFalse(
            torch.equal(
                original_weights,
                self.model.classifier.weight.detach(),
            )
        )
if __name__ == "__main__":
    unittest.main()