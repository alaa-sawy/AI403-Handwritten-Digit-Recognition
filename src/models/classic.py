"""M2 classic digit-string recognizer: ResNet-18 + BiGRU + CTC.

The CNN is adapted to grayscale images of height 48 while preserving the
horizontal time axis. The RGB classification head is intentionally removed:
we use the ResNet only as a visual feature extractor.

Model contract, shared with the other AI403 architectures:
    forward(images: [B, 1, 48, W], valid_widths: [B])
        -> (logits: [B, W, 11], output_lengths: [B])
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.models import resnet18, ResNet18_Weights


NUM_CLASSES = 11  # blank=0; digits 0..9 use class IDs 1..10


class ClassicCTC(nn.Module):
    """Identical architecture for scratch and pretrained weight initialization."""

    def __init__(self, dropout=0.1, pretrained=False):
        super().__init__()

        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 and 1")

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        # torchvision's ImageNet weights start with an RGB convolution.
        # Adapt it to a single grayscale channel. Keep the prior mean-channel
        # initialization for reproducibility with the original M2 design.
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=(2, 1),
            padding=3,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        backbone.conv1 = new_conv

        # FIX: the ImageNet 1000-class head is not called in forward(). Keeping
        # it would count/save 513,000 parameters that never receive gradients.
        # We make that fact explicit by removing the unused classification head.
        backbone.fc = nn.Identity()

        # Reduce height, but keep the horizontal sequence length unchanged.
        backbone.maxpool.stride = (2, 1)
        for layer in (backbone.layer2, backbone.layer3, backbone.layer4):
            layer[0].conv1.stride = (2, 1)
            layer[0].downsample[0].stride = (2, 1)

        self.backbone = backbone
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(512, NUM_CLASSES)

    def forward(self, images, valid_widths):
        # ResNet without the original global-average-pooling/classification
        # pathway: preserve one visual descriptor per horizontal position.
        x = self.backbone.conv1(images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # [B, 512, H, W] -> [B, W, 512]
        x = x.mean(dim=2).permute(0, 2, 1)

        # Every horizontal convolution/pool uses stride 1 and symmetric
        # padding, so the valid output length equals the input image width.
        output_lengths = valid_widths.long()

        # This prevents the GRU and CTC from processing *extra padded steps*.
        # It does not completely undo CNN boundary effects from batch padding.
        packed = pack_padded_sequence(
            x,
            output_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.gru(packed)
        x, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=x.size(1),
        )

        logits = self.classifier(self.dropout(x))
        return logits, output_lengths
