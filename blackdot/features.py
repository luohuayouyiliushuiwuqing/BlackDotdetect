"""黑色元素判定与手工结构特征提取.

对应设计方案第 3 节:
- 像素级暗像素判定:  Y < max(mu_bg - k*sigma_bg, T_abs)
- 区域级存在性判定:  暗像素占比 / 连通域面积 / 紧凑度 / 闭合对比度
- 8 维手工结构特征, 用于与 CNN 特征融合
"""

from __future__ import annotations

import cv2
import numpy as np

# ---- 阈值常量 (设计方案 3.2 / 3.3, 初始值, 需在验证集标定) ----
K_SIGMA = 2.5          # 相对暗判定系数 k
T_ABS = 90.0           # 绝对亮度下限保护 (0~255 域)
MIN_DARK_RATIO = 0.02  # 暗像素占比阈值 (20x20 ROI 中最小目标仅 ~8 像素)
MIN_BLOB_AREA = 6      # 最大暗连通域最小面积 (像素)
MIN_COMPACTNESS = 0.35  # 紧凑度阈值 4*pi*A/P^2
MIN_CONTRAST = 0.12    # 最小闭合对比度 (模糊退化后目标芯对比度约 0.1~0.4)

NUM_HANDCRAFTED = 8    # 手工特征维度
CORE = 20              # 目标区边长: 36x36 画布中心 20x20 为目标区, 外圈 8px 为背景环带


def bg_stats(roi: np.ndarray, ring: int = 8) -> tuple[float, float]:
    """由 ROI 外 ring 像素环带估计局部背景均值/标准差.

    roi 为 Y 分量灰度块 (H, W), 0~255。若 roi 已含上下文(如 36x36, 中心 20x20
    为目标区), 取四角与环带作为背景; 若无法取环带则退化为全图统计。
    """
    h, w = roi.shape
    if h > 2 * ring + 4 and w > 2 * ring + 4:
        ring_mask = np.ones((h, w), dtype=bool)
        ring_mask[ring:-ring, ring:-ring] = False
        bg = roi[ring_mask]
    else:
        bg = roi.reshape(-1)
    return float(bg.mean()), float(bg.std() + 1e-6)


def core_region(roi: np.ndarray, ring: int = 8) -> np.ndarray:
    """截取中心目标区 (36x36 -> 中心 20x20); 无上下文环带时返回原图.

    阈值常量按 20x20 目标区标定, 暗像素统计必须只在目标区内进行,
    背景统计量则来自外圈环带 (bg_stats), 否则占比阈值会随画布尺寸失效。
    """
    h, w = roi.shape
    if h > 2 * ring + 4 and w > 2 * ring + 4:
        return roi[ring:h - ring, ring:w - ring]
    return roi


def dark_mask(roi: np.ndarray, mu_bg: float, sigma_bg: float,
              k: float = K_SIGMA, t_abs: float = T_ABS) -> np.ndarray:
    """像素级暗像素掩码 (uint8 0/1)."""
    t_dark = max(mu_bg - k * sigma_bg, t_abs)
    return (roi < t_dark).astype(np.uint8)


def _largest_component(mask: np.ndarray) -> tuple[int, np.ndarray]:
    """返回最大连通域面积及其掩码."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return 0, np.zeros_like(mask)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (labels == idx).astype(np.uint8)
    return int(stats[idx, cv2.CC_STAT_AREA]), comp


def gradient_symmetry(roi: np.ndarray, comp: np.ndarray) -> float:
    """梯度方向对向能量比: 无人机(闭合团状)~1, 边缘 << 1.

    以暗斑质心为原点, 统计梯度方向直方图 0~180 成对方向的能量均衡度。
    """
    ys, xs = np.nonzero(comp)
    if len(ys) < 3:
        return 0.0
    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)[comp > 0]
    ang = np.degrees(np.arctan2(gy, gx))[comp > 0] % 360.0
    if mag.sum() < 1e-6:
        return 0.0
    # 以质心为参考仅统计指向斑心方向的梯度对向均衡: 简化为方向直方图对向能量
    bins, edges = np.histogram(ang, bins=18, range=(0, 360), weights=mag)
    opp = np.roll(bins, 9)
    num = np.minimum(bins, opp).sum()
    den = np.maximum(bins, opp).sum() + 1e-6
    return float(num / den)


def handcrafted_features(roi: np.ndarray, mu_bg: float | None = None,
                         sigma_bg: float | None = None) -> np.ndarray:
    """8 维手工结构特征 (设计方案 4.2).

    暗像素统计只在中心目标区 (20x20) 内进行, 背景均值/标准差来自外圈环带,
    保证占比/面积类特征与阈值常量在同一尺度上。
    """
    roi = roi.astype(np.float32)
    if mu_bg is None or sigma_bg is None:
        mu_bg, sigma_bg = bg_stats(roi)
    core = core_region(roi)
    mask = dark_mask(core, mu_bg, sigma_bg)
    total = float(core.size)

    dark_ratio = mask.sum() / total
    area, comp = _largest_component(mask)

    # 紧凑度 4*pi*A/P^2 (微小连通域的像素周长退化, 加下限保护)
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours and area >= 4:
        per = max(cv2.arcLength(contours[0], True), 2.0 * np.sqrt(area))
        compact = float(4.0 * np.pi * area / (per * per))
    else:
        compact = 0.0

    # 闭合对比度
    blob_mean = float(core[comp > 0].mean()) if area > 0 else mu_bg
    contrast = max((mu_bg - blob_mean) / max(mu_bg, 1e-6), 0.0)

    sym = gradient_symmetry(core, comp)

    # 质心偏移 (归一化到 [-1, 1])
    ys, xs = np.nonzero(comp)
    if area > 0:
        off_x = (xs.mean() / core.shape[1] - 0.5) * 2.0
        off_y = (ys.mean() / core.shape[0] - 0.5) * 2.0
        centroid_off = float(np.hypot(off_x, off_y))
    else:
        centroid_off = 0.0

    # ROI 边缘暗像素占比 (边缘类目标暗像素常触及 ROI 边界)
    border = np.zeros_like(mask)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = 1
    edge_touch = float((mask * border).sum() / max(mask.sum(), 1))

    # 暗斑长短轴比 (PCA 主轴比), 团状 ~1, 条带 >> 1
    if area >= 3:
        pts = np.stack([xs, ys], 1).astype(np.float64)
        cov = np.cov((pts - pts.mean(0)).T)
        eig = np.linalg.eigvalsh(cov)
        axis_ratio = float(np.sqrt(max(eig.max(), 1e-9) / max(eig.min(), 1e-9)))
    else:
        axis_ratio = 1.0

    return np.array([dark_ratio, area / total, compact, contrast,
                     sym, centroid_off, edge_touch, min(axis_ratio, 10.0) / 10.0],
                    dtype=np.float32)


def rule_baseline(roi: np.ndarray, mu_bg: float | None = None,
                  sigma_bg: float | None = None) -> dict:
    """规则基线 (设计方案 3.3): 判定是否存在黑色元素, 并给出几何提示类别.

    返回:
        black_exist: 是否存在黑色元素
        cls: 0=无人机(团状) 1=天空边缘(条带) 2=云层边缘(弥散) 3=背景; 不存在时为 3
        features: 8 维手工特征
    """
    roi_f = roi.astype(np.float32)
    if mu_bg is None or sigma_bg is None:
        mu_bg, sigma_bg = bg_stats(roi_f)
    feat = handcrafted_features(roi_f, mu_bg, sigma_bg)
    dark_ratio, area_ratio, compact, contrast, sym = feat[:5]
    area = area_ratio * core_region(roi_f).size   # 面积换算回像素数 (目标区尺度)

    exist = (dark_ratio >= MIN_DARK_RATIO and area >= MIN_BLOB_AREA
             and compact >= MIN_COMPACTNESS and contrast >= MIN_CONTRAST)

    cls = 3
    if exist:
        # 粗判: 闭合对称团状→无人机; 条带/低紧凑→天空边缘; 低对比弥散→云层边缘
        # sym 实测: 无人机中位 ~0.30, 天空边缘 ~0.00 -> 阈值取 0.15 更有效分离
        if sym >= 0.15 and compact >= 0.35:
            cls = 0
        elif compact < 0.5 and feat[7] * 10.0 > 2.0:
            cls = 1
        else:
            cls = 2
    return {"black_exist": bool(exist), "cls": cls, "features": feat}
