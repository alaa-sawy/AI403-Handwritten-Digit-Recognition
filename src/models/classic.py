"""Classic model: ResNet18 + BiGRU + CTC."""

import torch
from torch import nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)
from torchvision.models import resnet18, ResNet18_Weights


NUM_CLASSES = 11


class ClassicCTC(nn.Module):

    def __init__(self, dropout=0.1, pretrained=False):
        super().__init__()

        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 and 1")

        # 1. Create ResNet18
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        # 2. Change input from RGB (3 channels) to grayscale (1 channel)
        old_conv = backbone.conv1

        new_conv = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=(2, 1),
            padding=3,
            bias=False,
        )

        if pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(
                    old_conv.weight.mean(dim=1, keepdim=True)
                )

        backbone.conv1 = new_conv

        # 3. Preserve image width while reducing image height
        backbone.maxpool.stride = (2, 1)

        for layer in [
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        ]:
            layer[0].conv1.stride = (2, 1)
            layer[0].downsample[0].stride = (2, 1)

        self.backbone = backbone

        # 4. Bidirectional GRU
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # 5. Final classification layer
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(512, NUM_CLASSES)

    def forward(self, images, valid_widths):

        # ResNet feature extraction
        x = self.backbone.conv1(images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # [B, 512, H, W] -> [B, W, 512]
        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)

        # Width is preserved by the modified ResNet
        output_lengths = valid_widths.long()

        # Pack sequences to ignore padded positions in the GRU
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

        # Convert features into 11 class scores
        logits = self.classifier(self.dropout(x))

        return logits, output_lengths