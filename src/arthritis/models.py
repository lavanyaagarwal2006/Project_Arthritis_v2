"""Model zoo. Every model maps (image, joint index) -> (erosion logits [B, K_be], JSN logits [B, K_jsn] or None).

K_be = 1 for a binary head; K_be = (#grade levels - 1) for an ordinal head, where logit k models
P(grade > level_k). Logit 0 is always P(erosion present), i.e. the primary endpoint.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ScratchCNN(nn.Module):
    """Small 4-block CNN trained from scratch: shows how much ImageNet pretraining adds."""

    def __init__(self, in_ch: int = 1, n_be: int = 1, n_jsn: int = 0):
        super().__init__()

        def block(i, o, p):
            return nn.Sequential(nn.Conv2d(i, o, 3), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Dropout(p))
        self.features = nn.Sequential(block(in_ch, 32, 0.3), block(32, 64, 0.3), block(64, 128, 0.3), block(128, 256, 0.4))
        self.trunk = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(256 * 16, 256),
                                   nn.ReLU(inplace=True), nn.Dropout(0.5))
        self.head_be = nn.Linear(256, n_be)
        self.head_jsn = nn.Linear(256, n_jsn) if n_jsn else None

    def forward(self, x, joint_idx=None):
        f = self.trunk(self.features(x))
        return self.head_be(f), (self.head_jsn(f) if self.head_jsn is not None else None)


class JointNet(nn.Module):
    """timm backbone -> pooled features (+ optional joint-identity embedding) -> erosion head (+ JSN head)."""

    def __init__(self, name: str, pretrained: bool = True, drop_rate: float = 0.2,
                 n_be: int = 1, n_jsn: int = 0, n_joints: int = 0, emb_dim: int = 0):
        super().__init__()
        import timm
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, drop_rate=drop_rate)  # timm applies drop_rate before its (identity) head
        d = self.backbone.num_features
        self.emb = nn.Embedding(n_joints, emb_dim) if emb_dim else None
        d_in = d + (emb_dim if emb_dim else 0)
        self.head_be = nn.Linear(d_in, n_be)
        self.head_jsn = nn.Linear(d_in, n_jsn) if n_jsn else None

    def forward(self, x, joint_idx=None):
        f = self.backbone(x)
        if self.emb is not None:
            f = torch.cat([f, self.emb(joint_idx)], dim=1)
        return self.head_be(f), (self.head_jsn(f) if self.head_jsn is not None else None)


def build_model(cfg: dict) -> tuple[nn.Module, int, bool, int, int]:
    """Returns (model, input channels, uses ImageNet normalisation, K_be, K_jsn)."""
    from .data import JOINT_VOCAB
    mcfg = cfg["model"]
    ordinal = mcfg.get("head", "binary") == "ordinal"
    k_be = len(cfg["levels"]["BE"]) - 1 if ordinal else 1
    k_jsn = (len(cfg["levels"]["JSN"]) - 1) if cfg["train"].get("multitask_jsn") else 0
    if mcfg["name"] == "scratch_cnn":
        return ScratchCNN(1, k_be, k_jsn), 1, False, k_be, k_jsn
    pretrained = mcfg.get("pretrained", True)
    m = JointNet(mcfg["name"], pretrained, mcfg.get("drop_rate", 0.2), k_be, k_jsn,
                 len(JOINT_VOCAB), int(mcfg.get("joint_embedding", 0) or 0))
    return m, 3, pretrained, k_be, k_jsn


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Last conv-like block, used by Grad-CAM."""
    if isinstance(model, ScratchCNN):
        return model.features[-1][0]
    net = model.backbone
    for attr in ("features", "stages", "layer4"):
        if hasattr(net, attr):
            m = getattr(net, attr)
            return m[-1] if isinstance(m, (nn.Sequential, nn.ModuleList)) else m
    raise ValueError("set the Grad-CAM layer manually for this backbone")
