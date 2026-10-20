"""双头轻量 CNN (设计方案 4.1).

骨干: 4 层卷积 + SE 注意力, 不使用池化降采样以保留微小目标空间信息;
融合 8 维手工结构特征后接两个输出头:
    head1: black_exist (sigmoid, 黑色元素存在判定)
    head2: class_probs (softmax, 无人机/天空边缘/云层边缘/背景 4 分类)
参数量约 7 万, CPU 单 ROI 推理 < 0.1ms 量级。
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


class DualHeadNet(nn.Module):
    """输入 (B, 2, 20, 20) + 手工特征 (B, 8)."""

    def __init__(self, in_ch: int = 2, num_cls: int = 4,
                 n_feat: int = NUM_HANDCRAFTED, use_handcrafted: bool = True):
        super().__init__()
        self.use_handcrafted = use_handcrafted
        self.backbone = nn.Sequential(
            ConvBN(in_ch, 16),            # 20 -> 18
            ConvBN(16, 32),               # 18 -> 16
            SEBlock(32),
            ConvBN(32, 64),               # 16 -> 14
            SEBlock(64),
            ConvBN(64, 64),               # 14 -> 12
        )
        fused = 128 + (n_feat if use_handcrafted else 0)
        self.fc = nn.Sequential(nn.Linear(fused, 96), nn.ReLU(inplace=True))
        self.head_black = nn.Linear(96, 1)
        self.head_cls = nn.Linear(96, num_cls)

    def forward(self, x: torch.Tensor, feat: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.backbone(x)                       # (B, 64, 12, 12)
        gap = f.mean(dim=(2, 3))
        gmp = f.amax(dim=(2, 3))
        z = torch.cat([gap, gmp], dim=1)           # (B, 128)
        if self.use_handcrafted:
            if feat is None:
                raise ValueError("use_handcrafted=True 时必须传入手工特征")
            z = torch.cat([z, feat], dim=1)
        z = self.fc(z)
        return self.head_black(z).squeeze(1), self.head_cls(z)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, feat: torch.Tensor | None = None
                ) -> dict[str, torch.Tensor]:
        """返回 black_exist 概率、4 类概率、综合置信度。"""
        logit_b, logit_c = self.forward(x, feat)
        pb = torch.sigmoid(logit_b)
        pc = torch.softmax(logit_c, dim=1)
        return {"black_exist": pb, "class_probs": pc,
                "confidence": pb * pc.max(dim=1).values}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class FocalLoss(nn.Module):
    """多分类 Focal Loss (gamma=2, 缓解正样本稀少)."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()
