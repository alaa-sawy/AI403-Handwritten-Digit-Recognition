"""M4 - Advanced CNN + BiGRU + Custom Attention + CTC.

Supported variants:
    advanced
    advanced_no_multiscale
    advanced_no_attention

Input:
    images: [B, 1, 48, W]
    valid_widths: [B]

Output:
    logits: [B, T, 11]
    output_lengths: [B]
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)


NUM_CLASSES = 11
BLANK_ID = 0


class ConvBlock(nn.Module):
    """Convolution + BatchNorm + ReLU."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
    ):
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
    """Custom parallel multi-scale convolution block."""

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


class CustomMultiHeadAttention(nn.Module):
    """Custom Multi-Head Self-Attention.

    Parameter names and shapes are compatible with
    torch.nn.MultiheadAttention for checkpoint loading.
    """

    def __init__(
        self,
        embed_dim=256,
        num_heads=4,
        dropout=0.3,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        # Keep the same parameter names and shapes
        # as nn.MultiheadAttention.
        self.in_proj_weight = nn.Parameter(
            torch.empty(3 * embed_dim, embed_dim)
        )

        self.in_proj_bias = nn.Parameter(
            torch.empty(3 * embed_dim)
        )

        self.out_proj = nn.Linear(
            embed_dim,
            embed_dim,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=False,
    ):
        batch_size, sequence_length, _ = query.shape

        qkv = F.linear(
            query,
            self.in_proj_weight,
            self.in_proj_bias,
        )

        q, k, v = qkv.chunk(3, dim=-1)

        q = q.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        ) / math.sqrt(self.head_dim)

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]
            scores = scores.masked_fill(
                mask,
                float("-inf"),
            )

        attention_weights = torch.softmax(
            scores,
            dim=-1,
        )

        attention_weights = F.dropout(
            attention_weights,
            p=self.dropout,
            training=self.training,
        )

        attended = torch.matmul(
            attention_weights,
            v,
        )

        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .reshape(
                batch_size,
                sequence_length,
                self.embed_dim,
            )
        )

        attended = self.out_proj(attended)

        if need_weights:
            return attended, attention_weights.mean(dim=1)

        return attended, None


class AdvancedCTC(nn.Module):
    """Advanced handwritten digit-string recognition model."""

    def __init__(
        self,
        dropout: float = 0.3,
        multiscale: bool = True,
        attention: bool = True,
    ):
        super().__init__()

        if not 0 <= dropout < 1:
            raise ValueError(
                "dropout must be between 0 and 1"
            )

        self.multiscale = multiscale
        self.use_attention = attention

        # Stage 1
        self.stage1 = nn.Sequential(
            ConvBlock(1, 32, kernel_size=3),
            ConvBlock(32, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            ConvBlock(32, 64, kernel_size=3),
            ConvBlock(64, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

        # Stage 3
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

        # BiGRU
        self.bigru = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Optional Custom Attention
        if self.use_attention:
            self.attention = CustomMultiHeadAttention(
                embed_dim=256,
                num_heads=4,
                dropout=dropout,
            )

            self.attention_norm = nn.LayerNorm(256)
        else:
            self.attention = None
            self.attention_norm = None

        # Feed Forward
        self.feed_forward = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.Dropout(dropout),
        )

        self.output_norm = nn.LayerNorm(256)

        # CTC Classifier
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
            raise ValueError(
                "Input images must have one channel"
            )

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

        # CNN feature extraction
        features = self.stage1(images)
        features = self.stage2(features)
        features = self.stage3(features)
        features = self.height_pool(features)

        features = self.feature_dropout(features)

        # Convert CNN features into sequences
        features = features.mean(dim=2)
        features = features.permute(0, 2, 1).contiguous()

        output_lengths = widths

        # BiGRU with packed sequences
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

        # Attention padding mask
        positions = torch.arange(
            padded_width,
            device=sequence.device,
        )

        padding_mask = (
            positions.unsqueeze(0)
            >= output_lengths.to(sequence.device).unsqueeze(1)
        )

        # Optional Attention
        if self.use_attention:
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

        # Feed Forward residual block
        feed_forward_output = self.feed_forward(sequence)

        sequence = self.output_norm(
            sequence + feed_forward_output
        )

        sequence = self.dropout(sequence)

        # CTC classification
        logits = self.classifier(sequence)

        return logits, output_lengths