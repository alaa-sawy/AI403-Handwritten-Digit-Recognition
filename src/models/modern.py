import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ModernCTC(nn.Module):
    def __init__(self, num_classes=11, hidden_dim=256, dropout=0.1):
        super().__init__()

        # 1. CNN Feature Extractor (متوافقة مع ارتفاع صور 48 وعرض متغير)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # H/2, W/2

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # H/4, W/4

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # استخدام Adaptive Avg Pooling لتثبيت الارتفاع على 1 والحفاظ على العرض أوتوماتيكياً
            nn.AdaptiveAvgPool2d((1, None))
        )

        # Linear projection layer (128 channels -> hidden_dim)
        self.projector = nn.Linear(128, hidden_dim)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)

        # 3. Transformer Encoder (Batch First)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 4. CTC Head (Raw Logits without Softmax)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, images, valid_widths):
        if images.ndim != 4 or images.size(1) != 1 or images.size(2) != 48:
            raise ValueError("images must have shape [B, 1, 48, W]")
        valid_widths = torch.as_tensor(valid_widths, dtype=torch.long, device=images.device)
        if valid_widths.shape != (images.size(0),):
            raise ValueError("valid_widths must have shape [B]")
        if torch.any(valid_widths < 4) or torch.any(valid_widths > images.size(-1)):
            raise ValueError("valid widths must be between 4 and padded image width")

        # The collator pads on the right with white pixels (value 1).
        # Zero invalid input columns to match the CNN's implicit zero padding
        # at the right edge of an individual, unpadded image. This prevents
        # convolution features from seeing arbitrary extra white columns.
        input_indices = torch.arange(images.size(-1), device=images.device)
        x = images.masked_fill(
            input_indices[None, None, None, :] >= valid_widths[:, None, None, None],
            0.0,
        )

        # Mask feature columns after EACH width-downsampling block. Without
        # this, the CNN can propagate features from padded columns back into
        # valid positions in later convolutional layers, even if attention
        # itself masks the padding correctly.
        output_lengths = valid_widths.clone()
        for layer in self.cnn:
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                # Both MaxPool2d layers use kernel=2, stride=2.
                output_lengths = output_lengths // 2
                output_indices = torch.arange(x.size(-1), device=x.device)
                x = x.masked_fill(
                    output_indices[None, None, None, :] >= output_lengths[:, None, None, None],
                    0.0,
                )
        features = x

        # إعادة ترتيب الأبعاد للـ Sequence -> [Batch, W_out, 128]
        features = features.squeeze(2)
        features = features.permute(0, 2, 1)

        # إسقاط الأبعاد للـ hidden_dim -> [Batch, W_out, hidden_dim]
        x = self.projector(features)

        # إضافة الـ Positional Encoding
        x = self.pos_encoder(x)

        # تمرير البيانات خلال الـ Transformer
        # Transformer must not attend to right-padding time steps.
        # Two 2x width max-pools reduce each valid width to floor(width / 4).
        seq_len = x.size(1)
        if torch.any(output_lengths < 1) or torch.any(output_lengths > seq_len):
            raise ValueError("Incompatible post-CNN sequence lengths")
        time_indices = torch.arange(seq_len, device=x.device).unsqueeze(0)
        padding_mask = time_indices >= output_lengths.unsqueeze(1)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # حساب الـ Raw Logits للـ CTC -> [Batch, W_out, num_classes]
        logits = self.fc_out(x)

        # حساب أطوال التسلسل الصحيحة بعد الـ Downsampling (بما إننا عملنا MaxPool مرتين في العرض، فالعرض بيتقسم على 4)
        # The same lengths were used to construct the attention padding mask.

        return logits, output_lengths