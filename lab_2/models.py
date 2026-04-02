"""Convolutional Autoencoder with configurable bottleneck size."""

import torch
import torch.nn as nn

from config import DEFAULT_BOTTLENECK


class ConvAutoencoder(nn.Module):
    """Symmetric convolutional autoencoder for anomaly detection.

    Architecture (for 256x256 input):
        Encoder: 256x256x3 -> 128x128x32 -> 64x64x64 -> 32x32x128 -> 16x16x256 -> 16x16xB
        Decoder: 16x16xB -> 16x16x256 -> 32x32x128 -> 64x64x64 -> 128x128x32 -> 256x256x3

    No skip connections -- the decoder must reconstruct solely from the
    compressed bottleneck representation, which is essential for anomaly
    detection (artifacts cannot pass through the bottleneck).
    """

    def __init__(self, bottleneck_channels: int = DEFAULT_BOTTLENECK):
        super().__init__()
        self.bottleneck_channels = bottleneck_channels

        self.encoder = nn.Sequential(
            # 256x256x3 -> 128x128x32
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 128x128x32 -> 64x64x64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 64x64x64 -> 32x32x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 32x32x128 -> 16x16x256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 16x16x256 -> 16x16xB (bottleneck)
            nn.Conv2d(256, bottleneck_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            # 16x16xB -> 16x16x256
            nn.Conv2d(bottleneck_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 16x16x256 -> 32x32x128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 32x32x128 -> 64x64x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 64x64x64 -> 128x128x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 128x128x32 -> 256x256x3
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
