"""数据集模块: 在线合成 Dataset 与实拍 ROI 文件夹 Dataset.

任务: 二分类 —— 是否为无人机 (plane/drone).
标签约定 (与 synthesis.py 一致):
    0 = 无人机 (正类, label=1), 1 = 天空边缘, 2 = 云层边缘, 3 = 纯背景
    后三类均为负类 (label=0); 它们是与无人机最易混淆的"暗边缘/暗区"样本。

样本张量: 36x36x2 (Y 归一化 [0,1] + 有效像素掩码)。
目标应大致居中, 外圈 8px 为背景, 与合成管线保持一致。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .synthesis import ROI, synth_sample


def encode_roi(roi_gray: np.ndarray, out: int = ROI) -> np.ndarray:
    """将任意尺寸 Y ROI 编码为 (2, out, out) 张量.

    尺度处理原则 (方案 B): **绝不放大**
      - ROI > out : 等比缩小到最长边 = out (INTER_AREA, 保留全部内容, 不裁掉周围),
                    再居中填充到 out×out, 填充区掩码置 0
      - ROI < out : 居中放入 out 画布, 四周填 0 且掩码置 0 (保持像素尺度, 不放大)
    小目标因此永远不会被拉伸放大; 掩码通道告知网络哪些区域是填充。
    """
    roi_gray = np.asarray(roi_gray)
    h, w = roi_gray.shape

    # 超界: 等比缩小(保留全部内容); 非方形缩小后仍需居中填充
    if max(h, w) > out:
        scale = out / float(max(h, w))
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        patch = cv2.resize(roi_gray, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        patch = roi_gray

    ph, pw = patch.shape[:2]
    ch, cw = min(ph, out), min(pw, out)
    canvas = np.zeros((out, out), np.float32)
    mask = np.zeros((out, out), np.float32)
    sy, sx = (ph - ch) // 2, (pw - cw) // 2        # 源: 居中
    oy, ox = (out - ch) // 2, (out - cw) // 2      # 目标: 居中放置
    canvas[oy:oy + ch, ox:ox + cw] = patch[sy:sy + ch, sx:sx + cw] / 255.0
    mask[oy:oy + ch, ox:ox + cw] = 1.0

    return np.stack([canvas, mask])                # (2, out, out)


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
    """在线合成数据集 (每次 __getitem__ 重新采样, 等价无限数据).

    crop_prob / min_valid: 模拟实拍中"ROI 尺寸不一"的实际情况——
    以 crop_prob 的概率把 36x36 画布裁成 [min_valid, 36] 的居中窗口作为
    "本次裁剪到的 ROI", 再由 encode_roi 居中放回 36x36 并置掩码。
    目标绝对像素尺寸不变, 但训练时能见到"有效区域小于画布"的样本,
    与实拍分布对齐 (实测实拍仅 4.2% 是 36x36)。
    """

    def __init__(self, n_per_class: int, seed: int = 0, train: bool = True,
                 crop_prob: float = 0.6, min_valid: int = 12):
        self.n_per_class = n_per_class
        self.train = train
        self.seed = seed
        self.crop_prob = crop_prob
        self.min_valid = min_valid
        self.n = n_per_class * 4

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        cls = idx % 4
        valid = ROI
        if self.crop_prob > 0 and rng.random() < self.crop_prob:
            valid = int(rng.integers(self.min_valid, ROI + 1))
        img = synth_sample(rng, cls, valid_size=valid)
        if self.train:
            img = augment(rng, img)
        x = encode_roi(img)
        is_drone = 1.0 if cls == 0 else 0.0
        return (torch.from_numpy(x), torch.tensor(is_drone, dtype=torch.float32))


# 正类子目录名 (不区分大小写): 位于这些目录下的图像视为"飞机/无人机"正样本
POSITIVE_NAMES = {"drone", "plane", "target", "uav", "positive"}


class FolderROIDataset(Dataset):
    """实拍 ROI 数据集 (二分类: 飞机/无人机=正, 其余=负).

    目录结构(两种均可, 负样本无需细分为多类):
      ① 极简二分类: root/drone/*.png 为正, 其余任意子目录(如 non_drone/)为负
      ② 兼容旧四类: root/{drone,sky_edge,cloud_edge,background}/*.png,
         仅 drone 为正, 其余三类自动并入负类
    标签规则: 以图像所在子目录名是否 ∈ POSITIVE_NAMES 判定正负。
    图像读取后转灰度 (Y); **不缩放**——由 encode_roi 居中裁剪/居中放置到 36x36,
    保持目标真实像素尺度 (方案 B)。构图要求: 目标大致居中。
    """

    def __init__(self, root: str | Path, train: bool = True, seed: int = 0):
        self.train = train
        self.rng = np.random.default_rng(seed)
        self.samples: list[tuple[Path, int]] = []
        root = Path(root)
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts:
                is_pos = 1 if p.parent.name.lower() in POSITIVE_NAMES else 0
                self.samples.append((p, is_pos))
        if not self.samples:
            raise FileNotFoundError(f"未在 {root} 找到任何 ROI 图像")
        n_pos = sum(lbl for _, lbl in self.samples)
        print(f"[FolderROIDataset] {root}: {len(self.samples)} 张 "
              f"(正样本 {n_pos}, 负样本 {len(self.samples) - n_pos})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, is_drone = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取 {path}")
        img = img.astype(np.float32)   # 不 resize: 由 encode_roi 居中处理, 保持尺度
        if self.train:
            img = augment(self.rng, img)
        x = encode_roi(img)
        return (torch.from_numpy(x), torch.tensor(float(is_drone), dtype=torch.float32))
