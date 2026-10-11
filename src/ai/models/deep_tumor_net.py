"""Deep convolutional network for tumor detection and segmentation.

This model uses an encoder-decoder architecture with residual blocks and
attention mechanisms for accurate tumor localization and classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual convolutional block with batch normalization."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class AttentionGate(nn.Module):
    """Spatial attention gate for decoder skip connections."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class DeepTumorNet(nn.Module):
    """Residual encoder-decoder with classification and segmentation heads."""

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int, blocks: int, stride: int = 1) -> nn.Module:
        layers = [ResidualBlock(in_channels, out_channels, stride=stride)]
        layers.extend(ResidualBlock(out_channels, out_channels) for _ in range(blocks - 1))
        return layers[0] if blocks == 1 else nn.Sequential(*layers)

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 32,
        blocks_per_stage: int = 1,
        dropout_rate: float = 0.4,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be at least 1")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if base_channels < 2:
            raise ValueError("base_channels must be at least 2")
        if blocks_per_stage < 1:
            raise ValueError("blocks_per_stage must be at least 1")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in the range [0, 1)")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.blocks_per_stage = blocks_per_stage
        self.dropout_rate = dropout_rate
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.enc1 = self._make_stage(base_channels, base_channels, blocks_per_stage)
        self.enc2 = self._make_stage(base_channels, base_channels * 2, blocks_per_stage, stride=2)
        self.enc3 = self._make_stage(base_channels * 2, base_channels * 4, blocks_per_stage, stride=2)
        self.enc4 = self._make_stage(base_channels * 4, base_channels * 8, blocks_per_stage, stride=2)
        self.bottleneck = self._make_stage(base_channels * 8, base_channels * 16, blocks_per_stage, stride=2)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.att4 = AttentionGate(base_channels * 8)
        self.dec4 = self._make_stage(base_channels * 16, base_channels * 8, blocks_per_stage)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.att3 = AttentionGate(base_channels * 4)
        self.dec3 = self._make_stage(base_channels * 8, base_channels * 4, blocks_per_stage)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.att2 = AttentionGate(base_channels * 2)
        self.dec2 = self._make_stage(base_channels * 4, base_channels * 2, blocks_per_stage)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.att1 = AttentionGate(base_channels)
        self.dec1 = self._make_stage(base_channels * 2, base_channels, blocks_per_stage)

        self.segmentation_head = nn.Conv2d(base_channels, 1, 1)
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 16, base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(base_channels * 4, num_classes),
        )

    def forward(self, x: torch.Tensor, return_segmentation: bool = True) -> dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError("input must have shape (batch, channels, height, width)")
        if x.shape[0] < 1:
            raise ValueError("input batch must not be empty")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {x.shape[1]}")
        if x.shape[2] % 16 != 0 or x.shape[3] % 16 != 0:
            raise ValueError("input height and width dimensions must be divisible by 16")

        x1 = self.enc1(self.stem(x))
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.bottleneck(x4)
        classification = self.classification_head(x5)

        if not return_segmentation:
            return {"classification": classification}

        d4 = self.dec4(torch.cat((self.att4(x4), self.up4(x5)), dim=1))
        d3 = self.dec3(torch.cat((self.att3(x3), self.up3(d4)), dim=1))
        d2 = self.dec2(torch.cat((self.att2(x2), self.up2(d3)), dim=1))
        d1 = self.dec1(torch.cat((self.att1(x1), self.up1(d2)), dim=1))
        segmentation = self.segmentation_head(d1)
        return {"classification": classification, "segmentation": segmentation}
