"""M4 - Advanced CNN + BiGRU + Attention + CTC architecture.

Supports two variants:
    advanced              -> Multi-scale CNN + BiGRU + Attention
    advanced_no_multiscale -> Single-scale CNN + BiGRU + Attention

Shared interface:
    model(images, valid_widths) -> (logits, output_lengths)

Input:
    images: [B, 1, 48, W]
    valid_widths: [B]

Output:
    logits: [B, T, 11]
    output_lengths: [B]
"""

import torch
from torch import nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)


NUM_CLASSES = 11
BLANK_ID = 0


class ConvBlock(nn.Module):
    """Convolution + BatchNorm + ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 dilation=1):
        super().__init__()

        padding = dilation * (kernel_size // 2)

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MultiScaleBlock(nn.Module):
    """Parallel convolution branches with different receptive fields."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        branch_channels = out_channels // 2

        self.branch_3x3 = ConvBlock(
            in_channels,
            branch_channels,
            kernel_size=3,
        )

        self.branch_5x5 = ConvBlock(
            in_channels,
            branch_channels,
            kernel_size=5,
        )

        self.project = nn.Sequential(
            nn.Conv2d(
                branch_channels * 2,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        small_features = self.branch_3x3(x)
        large_features = self.branch_5x5(x)

        combined = torch.cat(
            [small_features, large_features],
            dim=1,
        )

        return self.project(combined)


class AdvancedCTC(nn.Module):
    """Advanced handwritten digit-string recognition model."""

    def __init__(self, dropout: float = 0.3,
                 multiscale: bool = True):
        super().__init__()

        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 and 1")

        self.multiscale = multiscale

        # Stage 1: extract low-level visual features.
        self.stage1 = nn.Sequential(
            ConvBlock(1, 32, kernel_size=3),
            ConvBlock(32, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

        # Stage 2: extract deeper visual features.
        self.stage2 = nn.Sequential(
            ConvBlock(32, 64, kernel_size=3),
            ConvBlock(64, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

        # Stage 3: multi-scale or single-scale feature extraction.
        if multiscale:
            self.stage3 = MultiScaleBlock(
                in_channels=64,
                out_channels=128,
            )
        else:
            self.stage3 = ConvBlock(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
            )

        self.height_pool = nn.MaxPool2d(
            kernel_size=(2, 1)
        )

        self.feature_dropout = nn.Dropout2d(dropout)

        # CNN output: [B, 128, 6, W]
        # Convert to sequence: [B, W, 128]
        self.bigru = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # BiGRU output dimension = 128 * 2 = 256.
        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        self.attention_norm = nn.LayerNorm(256)

        self.feed_forward = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.Dropout(dropout),
        )

        self.output_norm = nn.LayerNorm(256)

        self.classifier = nn.Linear(
            256,
            NUM_CLASSES,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, images, valid_widths):
        """Return CTC logits and valid output lengths."""

        if images.ndim != 4:
            raise ValueError(
                f"Expected images [B,1,H,W], got {tuple(images.shape)}"
            )

        batch_size, channels, height, padded_width = images.shape

        if channels != 1:
            raise ValueError("Input images must have one channel")

        if height != 48:
            raise ValueError(
                f"Expected image height 48, got {height}"
            )

        widths = torch.as_tensor(
            valid_widths,
            dtype=torch.long,
            device="cpu",
        )

        if widths.numel() != batch_size:
            raise ValueError(
                "valid_widths must contain one value per image"
            )

        if torch.any(widths <= 0) or torch.any(widths > padded_width):
            raise ValueError("Invalid input widths")

        # CNN feature extraction.
        features = self.stage1(images)
        features = self.stage2(features)
        features = self.stage3(features)
        features = self.height_pool(features)

        features = self.feature_dropout(features)

        # Average over height, preserving the horizontal time axis.
        features = features.mean(dim=2)

        # [B, 128, W] -> [B, W, 128]
        features = features.permute(0, 2, 1).contiguous()

        # Width is preserved by all CNN operations.
        output_lengths = widths

        # Pack sequences so padded positions do not affect BiGRU.
        packed_features = pack_padded_sequence(
            features,
            output_lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, _ = self.bigru(packed_features)

        sequence, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=padded_width,
        )

        # Create attention padding mask.
        positions = torch.arange(
            padded_width,
            device=sequence.device,
        )

        padding_mask = positions.unsqueeze(0) >= output_lengths.to(
            sequence.device
        ).unsqueeze(1)

        # Self-attention.
        attended, _ = self.attention(
            sequence,
            sequence,
            sequence,
            key_padding_mask=padding_mask,
            need_weights=False,
        )

        sequence = self.attention_norm(
            sequence + attended
        )

        # Feed-forward residual block.
        feed_forward_output = self.feed_forward(sequence)

        sequence = self.output_norm(
            sequence + feed_forward_output
        )

        sequence = self.dropout(sequence)

        # CTC classification.
        logits = self.classifier(sequence)

        return logits, output_lengths