"""评估指标 (设计方案 6.1): 混淆矩阵 / PR / FAR / 固定 FAR 下召回率."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

from .synthesis import CLASS_NAMES


def binary_metrics(y_true: np.ndarray, score: np.ndarray,
                   thr: float = 0.5) -> dict:
    """第一级 black_exist 二分类指标."""
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


def cls_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """第二级多分类指标: 混淆矩阵 + 各类 P/R/F1 + Macro-F1."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))), zero_division=0)
    per_cls = {CLASS_NAMES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                "f1": float(f1[i]), "support": int(sup[i])}
               for i in range(len(CLASS_NAMES))}
    return {"confusion_matrix": cm, "per_class": per_cls,
            "macro_f1": float(np.mean(f1))}


def drone_pr(y_true: np.ndarray, drone_score: np.ndarray) -> dict:
    """无人机 vs 其他: AUC / AP / 固定 FAR 下召回率 (核心指标)."""
    y_bin = (y_true == 0).astype(int)
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


def report(y_true: np.ndarray, black_score: np.ndarray,
           cls_pred: np.ndarray, drone_score: np.ndarray) -> str:
    """汇总输出评估报告字符串."""
    lines = ["=== 第一级: 黑色元素判定 ==="]
    for k, v in binary_metrics((y_true != 3).astype(int), black_score).items():
        lines.append(f"  {k}: {v:.4f}")
    cm_r = cls_metrics(y_true, cls_pred)
    lines.append("=== 第二级: 目标分类 ===")
    lines.append(f"  macro_f1: {cm_r['macro_f1']:.4f}")
    lines.append("  混淆矩阵 (行=真值, 列=预测; drone/sky_edge/cloud_edge/bg):")
    for row in cm_r["confusion_matrix"]:
        lines.append("    " + " ".join(f"{v:6d}" for v in row))
    for name, m in cm_r["per_class"].items():
        lines.append(f"  {name:12s} P={m['precision']:.3f} R={m['recall']:.3f} "
                     f"F1={m['f1']:.3f} (n={m['support']})")
    lines.append("=== 无人机 vs 其他 ===")
    for k, v in drone_pr(y_true, drone_score).items():
        lines.append(f"  {k}: {v:.4f}")
    return "\n".join(lines)
