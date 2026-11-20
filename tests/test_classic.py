"""M2 architecture and integration tests; run from the AI403 project root:

    python -m unittest discover -s tests -p "test_classic.py" -v

First-time pretrained tests require torchvision to download ResNet-18 weights.
"""

import unittest

import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

from src.ctc_utils import ctc_forward
from src.models.classic import ClassicCTC
from src.models.registry import build_model
from src.train import train_one_epoch


class TestClassicCTC(unittest.TestCase):
    def setUp(self):
        self.model = ClassicCTC(dropout=0.1, pretrained=False)
        self.model.eval()
        self.images = torch.randn(2, 1, 48, 117)
        self.widths = torch.tensor([80, 117])

    def test_forward_shapes_and_valid_widths(self):
        with torch.no_grad():
            logits, lengths = self.model(self.images, self.widths)
        self.assertEqual(tuple(logits.shape), (2, 117, 11))
        self.assertEqual(lengths.tolist(), [80, 117])

    def test_backward(self):
        self.model.train()
        logits, _ = self.model(self.images, self.widths)
        logits.mean().backward()
        self.assertIsNotNone(self.model.classifier.weight.grad)
        self.assertIsNotNone(self.model.gru.weight_ih_l0.grad)
        self.assertIsNotNone(self.model.backbone.layer4[1].conv2.weight.grad)

    def test_unused_resnet_classifier_is_removed(self):
        self.assertIsInstance(self.model.backbone.fc, nn.Identity)
        self.assertFalse(any(name.startswith("backbone.fc.") for name, _ in self.model.named_parameters()))

    def test_resnet_preserves_horizontal_length(self):
        # The actual CNN map's width must agree with the CTC length contract.
        x = self.images
        b = self.model.backbone
        with torch.no_grad():
            x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
            x = b.layer1(x)
            x = b.layer2(x)
            x = b.layer3(x)
            x = b.layer4(x)
        self.assertEqual(x.shape[3], self.images.shape[3])
        self.assertEqual(x.shape[1], 512)

    def test_scratch_registry(self):
        self.assertIsInstance(build_model("classic_scratch", dropout=0.1), ClassicCTC)

    def test_pretrained_registry(self):
        self.assertIsInstance(build_model("classic_pretrained", dropout=0.1), ClassicCTC)

    def test_pretrained_forward(self):
        model = build_model("classic_pretrained", dropout=0.1)
        model.eval()
        with torch.no_grad():
            logits, lengths = model(self.images, self.widths)
        self.assertEqual(tuple(logits.shape), (2, 117, 11))
        self.assertEqual(lengths.tolist(), [80, 117])

    def test_invalid_dropout(self):
        with self.assertRaises(ValueError):
            ClassicCTC(dropout=1.5, pretrained=False)

    def test_pretrained_weights_loaded_and_grayscale_adapted(self):
        model = build_model("classic_pretrained", dropout=0.1)
        original = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.assertIsInstance(model.backbone.fc, nn.Identity)
        self.assertTrue(torch.equal(
            model.backbone.layer1[0].conv1.weight,
            original.layer1[0].conv1.weight,
        ))
        self.assertTrue(torch.allclose(
            model.backbone.conv1.weight,
            original.conv1.weight.mean(dim=1, keepdim=True),
            rtol=0,
            atol=0,
        ))
        self.assertEqual(model.backbone.conv1.in_channels, 1)

    def test_ctc_forward_integration(self):
        batch = {
            "images": torch.randn(2, 1, 48, 117),
            "widths": torch.tensor([80, 117]),
            "labels": ["12", "34"],
            "targets": torch.tensor([2, 3, 4, 5]),
            "target_lengths": torch.tensor([2, 2]),
        }
        loss, logits, lengths = ctc_forward(
            self.model, batch, nn.CTCLoss(blank=0), torch.device("cpu")
        )
        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(tuple(logits.shape), (2, 117, 11))
        self.assertEqual(lengths.tolist(), [80, 117])
        loss.backward()
        self.assertIsNotNone(self.model.classifier.weight.grad)

    def test_shared_trainer_one_epoch(self):
        batch = {
            "images": torch.randn(2, 1, 48, 117),
            "widths": torch.tensor([80, 117]),
            "labels": ["12", "34"],
            "targets": torch.tensor([2, 3, 4, 5]),
            "target_lengths": torch.tensor([2, 2]),
        }
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003)
        original_weights = self.model.classifier.weight.detach().clone()
        avg_loss = train_one_epoch(
            model=self.model,
            loader=[batch],
            loss_fn=nn.CTCLoss(blank=0),
            optimizer=optimizer,
            config={"gradient_clip": 5.0},
            device=torch.device("cpu"),
        )
        self.assertTrue(torch.isfinite(torch.tensor(avg_loss)).item())
        self.assertFalse(torch.equal(
            original_weights, self.model.classifier.weight.detach()
        ))


if __name__ == "__main__":
    unittest.main()
