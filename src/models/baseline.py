"""M1 - Part 7.1: Small CNN baseline with CTC-ready outputs.

Run from the AI403 project root:
    python -m src.models.baseline
"""

import torch
from torch import nn


NUM_CLASSES = 11  # CTC blank (0) + digits 0..9 (IDs 1..10)


class CNNCTCBaseline(nn.Module):
    def __init__(self, dropout: float = 0.1):
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 and 1")
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(128, NUM_CLASSES)

    def forward(self, images, input_widths):
        """Return logits [B, T, 11] and valid sequence lengths [B]."""
        # Images: [batch, 1, 48, padded_width]
        features = self.features(images)     # [B, 128, 6, W]
        features = features.mean(dim=2)      # [B, 128, W]
        features = features.permute(0, 2, 1) # [B, W, 128]

        logits = self.classifier(self.dropout(features))   # [B, W, 11]

        # Pooling reduces height only; the time axis keeps the original width.
        output_lengths = input_widths.long()
        return logits, output_lengths


if __name__ == "__main__":
    from src.datasets.dataset import DigitStringDataset

    dataset = DigitStringDataset("train", augment=False)
    sample = dataset[0]

    # Add the batch dimension: [1, 48, W] -> [1, 1, 48, W]
    images = sample["image"].unsqueeze(0)
    widths = torch.tensor([sample["width"]], dtype=torch.long)

    model = CNNCTCBaseline()
    model.eval()

    with torch.no_grad():
        logits, output_lengths = model(images, widths)

    print("Label:", sample["label"])
    print("Input shape:", images.shape)
    print("Logits shape:", logits.shape)
    print("Output lengths:", output_lengths.tolist())
    print("Number of classes:", logits.shape[-1])

    assert logits.shape == (1, sample["width"], NUM_CLASSES)
    assert output_lengths.item() == sample["width"]

    print("\nPASS: Baseline CNN produces CTC-ready outputs!")
    print("Note: the model has not been trained yet.")
