"""评估指标 (设计方案 6.1): 混淆矩阵 / PR / FAR / 固定 FAR 下召回率."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_metrics(y_true: np.ndarray, score: np.ndarray,
                   thr: float = 0.5) -> dict:
    """二分类指标 (1=无人机, 0=其他): P/R/F1 + 每 ROI 虚警率."""
    pred = score >= thr
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    n_neg = int((y_true == 0).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-9),
            "far_per_roi": fp / max(n_neg, 1)}


def drone_pr(y_true: np.ndarray, drone_score: np.ndarray,
             far_targets=(1e-2, 1e-3, 1e-4)) -> dict:
    """无人机 vs 其他: AUC / AP / 固定 FAR 下召回率 (核心指标, 无人机为正类=1).

    重要: FAR 的**可分辨下限**是 1/n_neg。负样本只有几十个时,
    FAR=1e-3 / 1e-4 远低于该下限, 实际退化成"误报数必须为 0",
    估计值极不稳定 (换个划分集能从 0.19 跳到 1.0)。故同时返回
    min_resolvable_far 与 far_warning, 提醒不要误读。
    """
    y_bin = (y_true == 1).astype(int)
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return {"auc": float("nan"), "ap": float("nan")}
    order = np.argsort(-drone_score)
    y_sorted = y_bin[order]
    cum_fp = np.cumsum(1 - y_sorted)
    cum_tp = np.cumsum(y_sorted)
    n_neg = max(int((y_bin == 0).sum()), 1)
    n_pos = max(int(y_bin.sum()), 1)
    far = cum_fp / n_neg
    rec = cum_tp / n_pos
    out = {"auc": float(roc_auc_score(y_bin, drone_score)),
           "ap": float(average_precision_score(y_bin, drone_score)),
           "n_pos": n_pos, "n_neg": n_neg,
           "min_resolvable_far": 1.0 / n_neg,
           "far_warning": ""}
    for t in far_targets:
        ok = np.nonzero(far <= t)[0]
        out[f"recall@far{t:g}"] = float(rec[ok].max()) if len(ok) else 0.0
    too_small = [t for t in far_targets if t < out["min_resolvable_far"]]
    if too_small:
        need = ", ".join(f"{t:g}→需 ≥{int(10 / t)} 负样本" for t in too_small)
        out["far_warning"] = (
            f"负样本仅 {n_neg} 个, 最小可分辨 FAR={out['min_resolvable_far']:.4f}; "
            f"以下 FAR 目标低于该下限, 数值仅代表'零误报时的召回', 估计极不稳定: {need}")
    return out


def bootstrap_ci(y_true: np.ndarray, score: np.ndarray, fn,
                 n_boot: int = 300, seed: int = 0, alpha: float = 0.05
                 ) -> tuple[float, float]:
    """Bootstrap 置信区间: 对任意指标函数 fn(y, score) 重采样 n_boot 次."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:      # 重采样后只剩一类, 跳过
            continue
        try:
            v = fn(y_true[idx], score[idx])
        except Exception:                        # noqa: BLE001
            continue
        if np.isfinite(v):
            vals.append(float(v))
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, alpha / 2 * 100)),
            float(np.percentile(vals, (1 - alpha / 2) * 100)))


def report(y_true: np.ndarray, score: np.ndarray, thr: float = 0.5) -> str:
    """汇总输出二分类评估报告字符串 (y_true: 1=无人机, 0=其他)."""
    lines = ["=== 二分类: 是否无人机 ==="]
    for k, v in binary_metrics(y_true, score, thr).items():
        lines.append(f"  {k}: {v:.4f}")
    lines.append("=== 无人机检测 (AUC / AP / 固定 FAR 下召回率) ===")
    pr = drone_pr(y_true, score)
    for k, v in pr.items():
        if k in ("far_warning",):
            continue
        lines.append(f"  {k}: {v:.4f}")
    if pr.get("far_warning"):
        lines.append(f"  ⚠️ {pr['far_warning']}")
    return "\n".join(lines)
