"""轻量二分类 CNN (纯 CNN, 无手工特征分支).

模型契约:
    输入 : (B, 2, 36, 36)  float32, 通道0=Y/255, 通道1=有效像素掩码
    输出 : (B,) logits -> sigmoid -> P(dot)

骨干: 4 层卷积 + SE 注意力, 不使用池化降采样以保留微小目标空间信息;
GlobalAvgPool + GlobalMaxPool 拼接后接单二分类头。参数量约 7.6 万,
CPU 单 ROI 推理 < 0.1ms 量级, 适合嵌入式机载部署。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, ch: int, r: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, max(ch // r, 4)), nn.ReLU(inplace=True),
            nn.Linear(max(ch // r, 4), ch), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x.mean(dim=(2, 3)))[:, :, None, None]
        return x * w


class ConvBN(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class DroneNet(nn.Module):
    """二分类: 输入 (B, 2, 36, 36) -> logits (B,)."""

    def __init__(self, in_ch: int = 2):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBN(in_ch, 16),             # 36 -> 34
            ConvBN(16, 32),                # 34 -> 32
            SEBlock(32),
            ConvBN(32, 64),                # 32 -> 30
            SEBlock(64),
            ConvBN(64, 64),                # 30 -> 28
        )
        self.fc = nn.Sequential(nn.Linear(128, 96), nn.ReLU(inplace=True))
        self.head = nn.Linear(96, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)                       # (B, 64, 28, 28)
        gap = f.mean(dim=(2, 3))
        gmp = f.amax(dim=(2, 3))
        z = torch.cat([gap, gmp], dim=1)           # (B, 128)
        return self.head(self.fc(z)).squeeze(1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """返回 P(dot) 概率。"""
        return torch.sigmoid(self.forward(x))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class BinaryFocalLoss(nn.Module):
    """二分类 Focal Loss (gamma=2, 缓解正样本稀少).

    正样本 (dot) 在实拍中占比往往更低; Focal 通过下调易分样本权重,
    让模型聚焦难分样本与少数正类。pos_weight 用于处理正负类不平衡。
    """

    def __init__(self, gamma: float = 2.0,
                 pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight)
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        return (((1.0 - pt) ** self.gamma) * bce).mean()
