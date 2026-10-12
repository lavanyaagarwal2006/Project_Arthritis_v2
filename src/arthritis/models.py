"""Model zoo. Every model outputs one logit (erosion present)."""
from __future__ import annotations

import torch
import torch.nn as nn


class ScratchCNN(nn.Module):
    """Small 4-block CNN trained from scratch: shows how much ImageNet pretraining adds."""

    def __init__(self, in_ch: int = 1):
        super().__init__()
        def block(i, o, p):
            return nn.Sequential(nn.Conv2d(i, o, 3), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Dropout(p))
        self.features = nn.Sequential(block(in_ch, 32, 0.3), block(32, 64, 0.3), block(64, 128, 0.3), block(128, 256, 0.4))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(256 * 16, 256),
                                  nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(256, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


class TimmBinary(nn.Module):
    def __init__(self, name: str, pretrained: bool = True, drop_rate: float = 0.2):
        super().__init__()
        import timm
        self.net = timm.create_model(name, pretrained=pretrained, num_classes=1, drop_rate=drop_rate)

    def forward(self, x):
        return self.net(x).squeeze(1)


def build_model(mcfg: dict) -> tuple[nn.Module, int, bool]:
    """Returns (model, input channels, uses ImageNet normalisation)."""
    name = mcfg["name"]
    if name == "scratch_cnn":
        return ScratchCNN(in_ch=1), 1, False
    pretrained = mcfg.get("pretrained", True)
    return TimmBinary(name, pretrained, mcfg.get("drop_rate", 0.2)), 3, pretrained


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Last conv-like block, used by Grad-CAM (Phase 1b)."""
    if isinstance(model, ScratchCNN):
        return model.features[-1][0]
    net = model.net
    for attr in ("features", "stages", "layer4"):
        if hasattr(net, attr):
            m = getattr(net, attr)
            return m[-1] if isinstance(m, (nn.Sequential, nn.ModuleList)) else m
    raise ValueError("set the Grad-CAM layer manually for this backbone")
