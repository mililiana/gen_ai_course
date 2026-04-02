"""Perceptual loss using VGG-16 features for autoencoder training."""

import torch
import torch.nn as nn
import torchvision.models as models


# ImageNet normalization (VGG expects this)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class VGGFeatureExtractor(nn.Module):
    """Extract intermediate features from VGG-16 for perceptual loss."""

    # Layer name -> index in vgg16.features Sequential
    LAYER_MAP = {
        "relu1_2": 3,
        "relu2_2": 8,
        "relu3_3": 15,
        "relu4_3": 22,
    }

    def __init__(self, layer_names: list[str] | None = None):
        super().__init__()
        if layer_names is None:
            layer_names = list(self.LAYER_MAP.keys())

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        max_idx = max(self.LAYER_MAP[name] for name in layer_names) + 1
        self.features = vgg.features[:max_idx]

        # Freeze all parameters
        for param in self.features.parameters():
            param.requires_grad = False

        self.layer_names = layer_names
        self.layer_indices = {name: self.LAYER_MAP[name] for name in layer_names}

        # Register normalization buffers
        self.register_buffer("mean", IMAGENET_MEAN)
        self.register_buffer("std", IMAGENET_STD)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert [0,1] range to ImageNet-normalized."""
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract features at specified layers.

        Args:
            x: Tensor in [0, 1] range, shape (B, 3, H, W).

        Returns:
            Dict mapping layer_name -> feature tensor.
        """
        x = self._normalize(x)
        features = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            for name, idx in self.layer_indices.items():
                if i == idx:
                    features[name] = x
        return features


class PerceptualLoss(nn.Module):
    """Combined MSE + VGG perceptual loss.

    L = (1 - alpha) * MSE(input, output) + alpha * sum(MSE(vgg_i(input), vgg_i(output)))
    """

    def __init__(
        self,
        alpha: float = 0.5,
        layer_names: list[str] | None = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.vgg = VGGFeatureExtractor(layer_names)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Pixel-level MSE
        mse_loss = self.mse(output, target)

        # Perceptual loss
        output_features = self.vgg(output)
        target_features = self.vgg(target)

        perceptual_loss = torch.tensor(0.0, device=output.device)
        for name in output_features:
            perceptual_loss = perceptual_loss + self.mse(
                output_features[name], target_features[name]
            )
        perceptual_loss = perceptual_loss / len(output_features)

        return (1 - self.alpha) * mse_loss + self.alpha * perceptual_loss
