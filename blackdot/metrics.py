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


def drone_pr(y_true: np.ndarray, drone_score: np.ndarray) -> dict:
    """无人机 vs 其他: AUC / AP / 固定 FAR 下召回率 (核心指标, 无人机为正类=1)."""
    y_bin = (y_true == 1).astype(int)
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return {"auc": float("nan"), "ap": float("nan")}
    order = np.argsort(-drone_score)
    y_sorted = y_bin[order]
    cum_fp = np.cumsum(1 - y_sorted)
    cum_tp = np.cumsum(y_sorted)
    n_neg = max(int((y_bin == 0).sum()), 1)
    far = cum_fp / n_neg
    rec = cum_tp / max(int(y_bin.sum()), 1)
    out = {"auc": float(roc_auc_score(y_bin, drone_score)),
           "ap": float(average_precision_score(y_bin, drone_score))}
    for target_far in (1e-2, 1e-3, 1e-4):
        ok = np.nonzero(far <= target_far)[0]
        out[f"recall@far{target_far:g}"] = float(rec[ok].max()) if len(ok) else 0.0
    return out


def report(y_true: np.ndarray, score: np.ndarray, thr: float = 0.5) -> str:
    """汇总输出二分类评估报告字符串 (y_true: 1=无人机, 0=其他)."""
    lines = ["=== 二分类: 是否无人机 ==="]
    for k, v in binary_metrics(y_true, score, thr).items():
        lines.append(f"  {k}: {v:.4f}")
    lines.append("=== 无人机检测 (AUC / AP / 固定 FAR 下召回率) ===")
    for k, v in drone_pr(y_true, score).items():
        lines.append(f"  {k}: {v:.4f}")
    return "\n".join(lines)
