"""EfficientNetV2-S based defect classifier."""
import torch
import torch.nn as nn
import timm


def create_defect_classifier(
    num_classes: int,
    pretrained: bool = True,
    drop_rate: float = 0.3,
    drop_path_rate: float = 0.2,
) -> nn.Module:
    """Create EfficientNetV2-S with custom classifier head.

    EfficientNetV2-S key features:
    - Fused-MBConv blocks (faster than V1's MBConv)
    - Progressive learning strategy
    - 21M parameters (vs 5.3M in B0)
    - Optimal image size: 384x384

    Reference: Tan & Le, ICML 2021
    """
    model = timm.create_model(
        'tf_efficientnetv2_s.in1k',  # Pretrained on ImageNet-1k
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )

    return model


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'total_mb': total * 4 / (1024 * 1024),
    }
