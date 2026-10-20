"""数据集模块: 在线合成 Dataset 与实拍 ROI 文件夹 Dataset.

标签约定 (与 synthesis.py 一致):
    0 = 无人机, 1 = 天空边缘, 2 = 云层边缘, 3 = 纯背景
第一级标签 black_exist: cls==0/1/2 为 1 (边缘类在退化后同样存在暗像素),
cls==3 为 0。可按业务需要调整边缘类是否计入"存在"。

样本张量: 20x20x2 (Y 归一化 [0,1] + 有效像素掩码), 附带 8 维手工特征。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .features import handcrafted_features
from .synthesis import CLASS_NAMES, ROI, synth_sample


def encode_roi(roi_gray: np.ndarray, out: int = ROI) -> tuple[np.ndarray, np.ndarray]:
    """将任意尺寸 Y ROI 编码为 (2, out, out) 张量 + 8 维手工特征.

    不足 out 的部分右侧/下侧填 0, 掩码通道标记有效像素 (设计方案 2.1)。
    """
    h, w = roi_gray.shape
    canvas = np.zeros((out, out), np.float32)
    mask = np.zeros((out, out), np.float32)
    hh, ww = min(h, out), min(w, out)
    canvas[:hh, :ww] = roi_gray[:hh, :ww] / 255.0
    mask[:hh, :ww] = 1.0
    x = np.stack([canvas, mask])                       # (2, out, out)
    feat = handcrafted_features(roi_gray.astype(np.float32))
    return x, feat


def augment(rng: np.random.Generator, img: np.ndarray) -> np.ndarray:
    """数据增强 (设计方案 5.4): 亮度/平移/小旋转/翻转/噪声. 不做颜色抖动."""
    img = img * rng.uniform(0.7, 1.3)                  # 亮度缩放
    g = rng.uniform(0.8, 1.25)                         # gamma
    img = 255.0 * (np.clip(img, 0, 255) / 255.0) ** g
    h, w = img.shape
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)
    ang = rng.uniform(-15, 15)
    dx, dy = rng.uniform(-3, 3, 2)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    M[:, 2] += [dx, dy]
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    img += rng.normal(0, rng.uniform(0, 3), img.shape)
    return np.clip(img, 0, 255).astype(np.float32)


class SyntheticROIDataset(Dataset):
    """在线合成数据集 (每次 __getitem__ 重新采样, 等价无限数据)."""

    def __init__(self, n_per_class: int, seed: int = 0, train: bool = True):
        self.n_per_class = n_per_class
        self.train = train
        self.seed = seed
        self.n = n_per_class * 4

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        cls = idx % 4
        img = synth_sample(rng, cls)
        if self.train:
            img = augment(rng, img)
        x, feat = encode_roi(img)
        black = 1.0 if cls != 3 else 0.0
        return (torch.from_numpy(x), torch.from_numpy(feat),
                torch.tensor(cls, dtype=torch.long),
                torch.tensor(black, dtype=torch.float32))


class FolderROIDataset(Dataset):
    """实拍 ROI 数据集.

    目录结构: root/<class_name>/*.png, class_name ∈ CLASS_NAMES。
    图像读取后转灰度, resize 到 20x20 (INTER_AREA)。
    """

    def __init__(self, root: str | Path, train: bool = True, seed: int = 0):
        self.train = train
        self.rng = np.random.default_rng(seed)
        self.samples: list[tuple[Path, int]] = []
        root = Path(root)
        for cls, name in enumerate(CLASS_NAMES):
            for p in sorted((root / name).glob("*")):
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                    self.samples.append((p, cls))
        if not self.samples:
            raise FileNotFoundError(f"未在 {root} 找到任何 ROI 图像")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, cls = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取 {path}")
        img = cv2.resize(img, (ROI, ROI), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)
        if self.train:
            img = augment(self.rng, img)
        x, feat = encode_roi(img)
        black = 1.0 if cls != 3 else 0.0
        return (torch.from_numpy(x), torch.from_numpy(feat),
                torch.tensor(cls, dtype=torch.long),
                torch.tensor(black, dtype=torch.float32))
