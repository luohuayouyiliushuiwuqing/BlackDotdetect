"""轻量二分类 CNN (设计方案 4.1, 改造为"是否无人机"二分类).

骨干: 4 层卷积 + SE 注意力, 不使用池化降采样以保留微小目标空间信息;
融合 8 维手工结构特征后接单个二分类头:
    head: sigmoid -> P(无人机)
输入为 36x36x2 (Y + 掩码), 目标居中、外圈 8px 为背景。参数量约 7.6 万,
CPU 单 ROI 推理 < 0.1ms 量级, 适合嵌入式机载部署。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import NUM_HANDCRAFTED


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
    """二分类: 输入 (B, 2, 36, 36) + 手工特征 (B, 8) -> P(无人机)."""

    def __init__(self, in_ch: int = 2, n_feat: int = NUM_HANDCRAFTED,
                 use_handcrafted: bool = True):
        super().__init__()
        self.use_handcrafted = use_handcrafted
        self.backbone = nn.Sequential(
            ConvBN(in_ch, 16),            # 36 -> 34
            ConvBN(16, 32),               # 34 -> 32
            SEBlock(32),
            ConvBN(32, 64),               # 32 -> 30
            SEBlock(64),
            ConvBN(64, 64),               # 30 -> 28
        )
        fused = 128 + (n_feat if use_handcrafted else 0)
        self.fc = nn.Sequential(nn.Linear(fused, 96), nn.ReLU(inplace=True))
        self.head = nn.Linear(96, 1)

    def forward(self, x: torch.Tensor, feat: torch.Tensor | None = None
                ) -> torch.Tensor:
        f = self.backbone(x)                       # (B, 64, 28, 28)
        gap = f.mean(dim=(2, 3))
        gmp = f.amax(dim=(2, 3))
        z = torch.cat([gap, gmp], dim=1)           # (B, 128)
        if self.use_handcrafted:
            if feat is None:
                raise ValueError("use_handcrafted=True 时必须传入手工特征")
            z = torch.cat([z, feat], dim=1)
        z = self.fc(z)
        return self.head(z).squeeze(1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, feat: torch.Tensor | None = None
                ) -> dict[str, torch.Tensor]:
        """返回无人机概率与综合置信度。"""
        p = torch.sigmoid(self.forward(x, feat))
        return {"is_drone": p, "confidence": p}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class BinaryFocalLoss(nn.Module):
    """二分类 Focal Loss (gamma=2, 缓解正样本稀少).

    正样本 (无人机) 在合成数据中约占 1/4, 实拍中占比可能更低; Focal 通过
    下调易分样本权重, 让模型聚焦难分样本与少数正类。
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
