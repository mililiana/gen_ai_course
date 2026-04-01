"""Model definitions for artifact detection."""

import timm
import torch.nn as nn

from config import EFFICIENTNET_MODEL, VIT_MODEL, NUM_CLASSES


def get_efficientnet(pretrained: bool = True) -> nn.Module:
    """EfficientNet-B0 for binary classification (Approach 1).

    Good at capturing texture and local patterns (bad hands, mask edges, etc.).
    Input: 384x384.
    """
    model = timm.create_model(
        EFFICIENTNET_MODEL,
        pretrained=pretrained,
        num_classes=NUM_CLASSES,
    )
    return model


def get_vit(pretrained: bool = True) -> nn.Module:
    """ViT-Base/16 for binary classification (Approach 2).

    Pretrained on ImageNet-21K, strong at capturing global/semantic artifacts
    (text overlays, unusual compositions, eye direction).
    Input: 224x224.
    """
    model = timm.create_model(
        VIT_MODEL,
        pretrained=pretrained,
        num_classes=NUM_CLASSES,
    )
    return model


def freeze_backbone(model: nn.Module, model_type: str):
    """Freeze all parameters except the classification head."""
    if model_type == "efficientnet":
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    elif model_type == "vit":
        for name, param in model.named_parameters():
            if "head" not in name:
                param.requires_grad = False


def unfreeze_all(model: nn.Module):
    """Unfreeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = True


def get_param_groups(model: nn.Module, model_type: str, backbone_lr: float, head_lr: float):
    """Create parameter groups with differential learning rates."""
    if model_type == "efficientnet":
        head_params = [p for n, p in model.named_parameters() if "classifier" in n]
        backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    elif model_type == "vit":
        head_params = [p for n, p in model.named_parameters() if "head" in n]
        backbone_params = [p for n, p in model.named_parameters() if "head" not in n]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ]
