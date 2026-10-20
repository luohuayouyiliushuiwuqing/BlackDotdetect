"""合成数据管线 (设计方案 5.3, 适配二分类"是否无人机"任务).

生成四类 ROI 样本 (Y 分量, 36x36, 0~255 float32), 直接作为模型输入:
    0 = 无人机小目标 (紧凑暗斑, 居中)   -> 正类 (label=1)
    1 = 天空边缘   (两亮度区分界条带)   -> 负类 (label=0)
    2 = 云层边缘   (弥散软过渡暗区)     -> 负类 (label=0)
    3 = 纯背景     (均匀天空 / 云内部)   -> 负类 (label=0)

关键原则: 所有类别走同一退化管线 (缩放/模糊/噪声/亮度漂移),
避免模型学到"模糊程度"捷径; 输出 36x36 含外圈 8px 背景环带,
供 handcrafted_features 的相对暗判定使用。
"""

from __future__ import annotations

import cv2
import numpy as np

ROI = 36          # 模型输入 ROI 边长 (含上下文环带, 目标居中, 外圈 8px 为背景)
CTX = 36          # 生成画布边长 (与 ROI 一致, 直接作为模型输入)
CLASS_NAMES = ["drone", "sky_edge", "cloud_edge", "background"]


# ---------------- 背景生成 ----------------
def _smooth_noise(rng: np.random.Generator, size: int, sigma: float) -> np.ndarray:
    n = rng.normal(0, 1, (size, size)).astype(np.float32)
    return cv2.GaussianBlur(n, (0, 0), sigma)


def make_sky_background(rng: np.random.Generator, size: int = CTX) -> np.ndarray:
    """均匀/缓变天空背景, 亮度 120~230."""
    base = rng.uniform(120, 230)
    grad = np.linspace(rng.uniform(-20, 20), rng.uniform(-20, 20), size,
                       dtype=np.float32)[:, None]
    img = base + grad + _smooth_noise(rng, size, rng.uniform(2, 5)) * rng.uniform(1, 4)
    return img.astype(np.float32)


def make_cloud_texture(rng: np.random.Generator, size: int = CTX) -> np.ndarray:
    """云层纹理 (低频噪声), 亮度 140~240."""
    base = rng.uniform(140, 240)
    tex = _smooth_noise(rng, size, rng.uniform(1.5, 4.0)) * rng.uniform(15, 45)
    return (base + tex).astype(np.float32)


# ---------------- 目标/边缘合成 ----------------
def _drone_silhouette(rng: np.random.Generator, hi: int = 64) -> np.ndarray:
    """高分辨率无人机剪影 (hi x hi, 0/1). 可由实拍抠图/3D 渲染替换."""
    img = np.zeros((hi, hi), np.uint8)
    c = hi // 2
    bw = int(hi * rng.uniform(0.22, 0.34))   # 机身半宽
    bh = int(bw * rng.uniform(0.5, 0.8))
    cv2.ellipse(img, (c, c), (bw, bh), 0, 0, 360, 1, -1)
    arm = int(hi * rng.uniform(0.30, 0.45))  # 旋翼臂
    t = max(1, hi // 32)
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        cv2.line(img, (c, c), (c + dx * arm, c + dy * arm), 1, t)
        r = max(1, int(hi * rng.uniform(0.04, 0.08)))
        cv2.circle(img, (c + dx * arm, c + dy * arm), r, 1, -1)
    ang = rng.uniform(0, 360)
    M = cv2.getRotationMatrix2D((c, c), ang, 1.0)
    return cv2.warpAffine(img, M, (hi, hi))


def _degrade(rng: np.random.Generator, img: np.ndarray,
             target: int | None = None) -> np.ndarray:
    """统一退化管线: 大气 PSF 模糊 + 传感器噪声 (所有类别共用).

    target: 目标边长(仅正样本传入)。size-aware 模糊: 小目标用小模糊核,
    避免 2x2 这类极小目标被高斯模糊链完全消散而学不到; 大目标与负样本
    保持原 σ∈[0.3,0.9] 全范围, 不改变它们的退化分布。
    """
    if target is not None:
        # 模糊核上限随目标边长线性缩放并截断到原范围:
        #   2x2 → σ≤0.4, 3x3 → σ≤0.6, 4x4 → σ≤0.8, ≥5x5 → σ≤0.9 (与原来一致)
        sigma_hi = float(np.clip(0.2 * target, 0.3, 0.9))
        sigma = rng.uniform(0.3, sigma_hi)
    else:
        sigma = rng.uniform(0.3, 0.9)  # 负样本/边缘: 无目标, 保持全范围
    img = cv2.GaussianBlur(img, (0, 0), sigma)
    img = img + rng.normal(0, rng.uniform(1.0, 6.0), img.shape).astype(np.float32)
    ps = rng.uniform(0.0, 0.02)  # 少量椒盐噪声
    if ps > 0.005:
        n_flip = int(img.size * ps)
        idx = rng.choice(img.size, n_flip, replace=False)
        flat = img.reshape(-1)
        flat[idx] = rng.choice([0, 255], n_flip)
    return img


def _paste_drone(rng: np.random.Generator, bg: np.ndarray
                 ) -> tuple[np.ndarray, int]:
    """在背景上融合退化后的无人机暗斑 (相对亮度系数 alpha 变暗).

    返回 (融合后画布, 目标边长) —— 目标边长供 size-aware 退化使用。
    """
    size = bg.shape[0]
    sil = _drone_silhouette(rng)
    # 目标最终像素尺寸 2x2 ~ 18x18, 亚像素相位偏移
    target = int(rng.integers(2, 19))
    shift = rng.uniform(-0.5, 0.5, 2)
    M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    sil_small = cv2.resize(sil.astype(np.float32), (target, target),
                           interpolation=cv2.INTER_AREA)
    sil_small = cv2.warpAffine(sil_small, M, (target, target))
    tau = rng.uniform(0.6, 1.0)              # 大气透过率 → 对比度衰减
    # 极小目标(<5px)提高变暗深度下限, 配合 size-aware 模糊保证目标芯仍可达最小对比度
    alpha_lo = 0.45 if target < 5 else 0.25
    alpha = rng.uniform(alpha_lo, 0.65) * tau
    x0 = (size - target) // 2 + int(rng.integers(-2, 3))
    y0 = (size - target) // 2 + int(rng.integers(-2, 3))
    x0, y0 = np.clip([x0, y0], 0, size - target)
    roi = bg[y0:y0 + target, x0:x0 + target]
    bg[y0:y0 + target, x0:x0 + target] = roi * (1.0 - alpha * sil_small)
    return bg, target


def _make_sky_edge(rng: np.random.Generator, size: int) -> np.ndarray:
    """天空边缘: 两亮度区直线分界 (地平线/建筑轮廓)."""
    hi, lo = rng.uniform(150, 235), rng.uniform(60, 140)
    if rng.random() < 0.5:
        hi, lo = lo, hi
    img = np.full((size, size), hi, np.float32)
    ang = rng.uniform(0, np.pi)
    off = rng.uniform(-0.35, 0.35) * size
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    d = (xx - size / 2) * np.cos(ang) + (yy - size / 2) * np.sin(ang) - off
    img[d > 0] = lo
    img += _smooth_noise(rng, size, 3.0) * rng.uniform(1, 4)
    return img


def _make_cloud_edge(rng: np.random.Generator, size: int) -> np.ndarray:
    """云层边缘: 云纹理与天空的弥散软过渡."""
    sky = make_sky_background(rng, size)
    cloud = make_cloud_texture(rng, size)
    mask = _smooth_noise(rng, size, rng.uniform(2.0, 4.0))
    mask = cv2.GaussianBlur(mask, (0, 0), rng.uniform(2.5, 5.0))
    mask = (mask - mask.min()) / (np.ptp(mask) + 1e-6)
    mask = np.clip((mask - rng.uniform(0.3, 0.6)) * rng.uniform(2, 5), 0, 1)
    return sky * (1 - mask) + cloud * mask


# ---------------- 采样入口 ----------------
def synth_sample(rng: np.random.Generator, cls: int,
                 roi: int = ROI, ctx: int = CTX) -> np.ndarray:
    """生成单个 ROI 样本, 返回 (ctx, ctx) float32 0~255 的完整画布.

    无人机目标居中(含 ±2px 抖动), 外圈 8px 为背景环带; 边缘/背景类则铺满整幅。
    直接作为 36x36 模型输入, 不再中心裁剪 —— 保留背景上下文以稳定相对暗判定。
    """
    if cls == 0:
        canvas = make_sky_background(rng, ctx) if rng.random() < 0.6 \
            else _make_cloud_edge(rng, ctx)
        canvas, target = _paste_drone(rng, canvas)
        canvas = _degrade(rng, canvas, target=target)   # size-aware 模糊
    elif cls == 1:
        canvas = _degrade(rng, _make_sky_edge(rng, ctx))
    elif cls == 2:
        canvas = _degrade(rng, _make_cloud_edge(rng, ctx))
    else:
        canvas = make_sky_background(rng, ctx) if rng.random() < 0.6 \
            else make_cloud_texture(rng, ctx)
        canvas = _degrade(rng, canvas)
    # 直接返回完整 36x36 画布: 目标居中(无人机)或分界(边缘), 外圈 8px 为背景环带,
    # 供 handcrafted_features 的 bg_stats 估计局部背景 -> 相对暗判定更稳定。
    return np.clip(canvas, 0, 255).astype(np.float32)


def synth_dataset(n_per_class: int, roi: int = ROI, seed: int = 0
                  ) -> tuple[np.ndarray, np.ndarray]:
    """生成均衡合成数据集, 返回 X (N, roi, roi) float32, y (N,) 二分类(无人机=1).

    内部仍按 4 类语义生成以保证负样本多样性 (天空/云边缘), 但标签统一映射为
    二分类: drone=1, 其余=0。
    """
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for cls in range(4):
        for _ in range(n_per_class):
            xs.append(synth_sample(rng, cls, roi))
            ys.append(1 if cls == 0 else 0)
    return np.stack(xs), np.array(ys, dtype=np.int64)
