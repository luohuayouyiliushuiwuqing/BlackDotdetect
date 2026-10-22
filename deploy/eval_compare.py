"""对比评估: 直接检测 drone vs 检测 non_drone 反推 drone (二分类 ROI 级).

对带标注目录 (如 BlackDotDataset/data/{dot,non_dot}) 逐张打分 P(drone), 然后:

  方案 A (直接):   以 drone 为正类, pred_drone = P(drone) >= thr
  方案 B (反推):   以 non_drone 为正类检测, P(non_drone) = 1 - P(drone),
                   pred_non = P(non_drone) >= thr, 反推 drone = NOT pred_non

输出: 两种方案各自目标类的 P/R/F1/AP/mAP、drone 视角的漏检/误报, 以及
      阈值扫描下的等价性/差异分析 (类别不平衡的影响)。

用法:
    python eval_compare.py --dataset /path/to/BlackDotDataset/data \
        --pos-dir dot --neg-dir non_dot --model ../runs/dronenet.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect import DroneClassifier as DotClassifier, resolve_path  # noqa: E402

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


# ------------------------------------------------------------ 指标 (纯 numpy)
def prf(tp: int, fp: int, fn: int) -> dict:
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-9)}


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """AP = Σ (R_n - R_{n-1}) * P_n (标准插值定义, 纯 numpy)."""
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y).astype(np.float64)
    fp = np.cumsum(1 - y).astype(np.float64)
    recall = tp / tp[-1]
    prec = tp / np.maximum(tp + fp, 1e-9)
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * prec))


def confusion(y_true: np.ndarray, pred: np.ndarray) -> dict:
    """以 drone=1 为正类统计."""
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def collect(root: Path, pos_dir: str, neg_dir: str) -> tuple[list, list]:
    """返回 (files, labels); label=1 为 drone 正类."""
    files, labels = [], []
    for cls_dir, label in ((pos_dir, 1), (neg_dir, 0)):
        d = root / cls_dir
        if not d.exists():
            raise FileNotFoundError(f"缺少类别目录: {d}")
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                files.append(f)
                labels.append(label)
    if not files:
        raise FileNotFoundError(f"{root} 下未找到图像")
    return files, np.array(labels, dtype=np.int64)


# ------------------------------------------------------------ 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="直接检测 drone vs 检测 non_drone 反推 对比评估")
    ap.add_argument("--dataset", required=True, help="数据根目录")
    ap.add_argument("--pos-dir", default="dot", help="drone 正类子目录名")
    ap.add_argument("--neg-dir", default="non_dot", help="non_drone 负类子目录名")
    ap.add_argument("--model", default=None, help="ONNX 模型 (默认 runs/dronenet.onnx)")
    ap.add_argument("--input-size", type=int, default=36)
    ap.add_argument("--thr", type=float, default=0.5, help="主工作点阈值")
    ap.add_argument("--sweep", action="store_true", help="打印阈值扫描表")
    ap.add_argument("--output", default=None, help="结果 JSON 输出路径")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    root = Path(args.dataset)
    model = args.model or str(Path(__file__).resolve().parent.parent / "runs" / "dronenet.onnx")
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if args.device == "cuda" else ["CPUExecutionProvider"])
    clf = DotClassifier(model, input_size=args.input_size, providers=providers)

    files, y = collect(root, args.pos_dir, args.neg_dir)
    imgs = []
    for f in files:
        im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise ValueError(f"无法读取 {f}")
        imgs.append(im)
    p_drone = clf.predict_proba(imgs, batch_size=512)
    p_non = 1.0 - p_drone
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    print("=" * 70)
    print(f"数据: {root}   drone 正类 {n_pos} 张, non_drone 负类 {n_neg} 张 "
          f"(不平衡 1:{n_neg / max(n_pos, 1):.2f})")
    print(f"模型: {model}")
    print("=" * 70)

    y_dr = y                      # drone=1
    y_non = 1 - y                 # non_drone=1

    # ---- 方案 A: 直接检测 drone ----
    pred_a = (p_drone >= args.thr).astype(np.int64)
    cm_a = confusion(y_dr, pred_a)
    m_a = prf(cm_a["tp"], cm_a["fp"], cm_a["fn"])
    ap_drone = average_precision(y_dr, p_drone)

    # ---- 方案 B: 检测 non_drone, 反推 drone ----
    pred_b_non = (p_non >= args.thr).astype(np.int64)   # non_drone 检出
    cm_b_non = confusion(y_non, pred_b_non)             # 以 non_drone 为正类
    m_b_non = prf(cm_b_non["tp"], cm_b_non["fp"], cm_b_non["fn"])
    ap_non = average_precision(y_non, p_non)
    pred_b_drone = 1 - pred_b_non                       # 反推: 未检出 non_drone => drone
    cm_b = confusion(y_dr, pred_b_drone)
    m_b = prf(cm_b["tp"], cm_b["fp"], cm_b["fn"])

    print(f"\n[方案 A] 直接检测 drone          (thr={args.thr})")
    print(f"  P={m_a['precision']:.4f} R={m_a['recall']:.4f} F1={m_a['f1']:.4f}  "
          f"AP(drone)={ap_drone:.4f}")
    print(f"  混淆: TP={cm_a['tp']} FP={cm_a['fp']} FN={cm_a['fn']} TN={cm_a['tn']}"
          f"   漏检率={cm_a['fn'] / max(n_pos, 1):.4f}  "
          f"误报率={cm_a['fp'] / max(n_neg, 1):.4f}")

    print(f"\n[方案 B] 检测 non_drone 后反推    (thr={args.thr}, P(non)=1-P(drone))")
    print(f"  non_drone 视角: P={m_b_non['precision']:.4f} R={m_b_non['recall']:.4f} "
          f"F1={m_b_non['f1']:.4f}  AP(non_drone)={ap_non:.4f}")
    print(f"  反推 drone 视角: P={m_b['precision']:.4f} R={m_b['recall']:.4f} "
          f"F1={m_b['f1']:.4f}")
    print(f"  混淆(drone 视角): TP={cm_b['tp']} FP={cm_b['fp']} FN={cm_b['fn']} TN={cm_b['tn']}")

    identical = (pred_a == pred_b_drone).all()
    print(f"\n[等价性] 同阈值下反推决策与直接检测完全一致: {'是' if identical else '否'}")
    print(f"  mAP = mean(AP(drone), AP(non_drone)) = "
          f"{(ap_drone + ap_non) / 2:.4f}   (AP(drone)={ap_drone:.4f}, "
          f"AP(non_drone)={ap_non:.4f})")
    print("  注: AP/精确率/召回率**不是**互补量 —— AP(non_drone) 高不等于 drone 检测差,")
    print("      但也**不能**用 1-AP(non_drone) 反推 AP(drone)。")

    # ---- 类别不平衡影响: 阈值扫描 ----
    if args.sweep:
        print(f"\n阈值扫描 (drone 视角; 不平衡 1:{n_neg / max(n_pos, 1):.2f}):")
        print(f"{'thr':>5} | {'A: P/R/F1':>18} | {'B(反推): P/R/F1':>18} | "
              f"{'A漏检':>6} {'A误报':>6} | 决策一致")
        for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            pa = (p_drone >= t).astype(np.int64)
            pb = 1 - (p_non >= t).astype(np.int64)
            ca, cb = confusion(y_dr, pa), confusion(y_dr, pb)
            ma, mb = prf(**{k: ca[k] for k in ("tp", "fp", "fn")}), \
                prf(**{k: cb[k] for k in ("tp", "fp", "fn")})
            same = (pa == pb).all()
            print(f"{t:>5.2f} | {ma['precision']:.3f}/{ma['recall']:.3f}/{ma['f1']:.3f} | "
                  f"{mb['precision']:.3f}/{mb['recall']:.3f}/{mb['f1']:.3f} | "
                  f"{ca['fn']:>6} {ca['fp']:>6} | {'一致' if same else '不同'}")
        # B 方案的"最优工作点"应在 non_drone 分数上取不同阈值: 展示反推的阈值语义反转
        best_b, best_t = -1.0, 0.5
        for t in np.arange(0.05, 1.0, 0.05):
            pb = 1 - (p_non >= t).astype(np.int64)
            cb = confusion(y_dr, pb)
            f1 = prf(**{k: cb[k] for k in ("tp", "fp", "fn")})["f1"]
            if f1 > best_b:
                best_b, best_t = f1, float(t)
        print(f"  反推方案的 F1 最优工作点: P(non_drone) 阈值 = {best_t:.2f} "
              f"(等价 P(drone) 阈值 {1 - best_t:.2f}), F1={best_b:.4f}")
        print("  即: 反推时'阈值语义反转' —— 想抓更多 drone, 要把 non_drone 的阈值调低;")

    # ---- 结论性分析 ----
    print("\n" + "=" * 70)
    print("分析结论:")
    print("  1) 单模型二分类下, 反推 = 互补事件 (P(non)=1-P(drone)), 不产生任何新信息;")
    print("     同阈值时两者决策逐张一致, 漏检/误报完全相同 —— 差异只在指标的'叫法'。")
    print("  2) 但 P/R/AP 不是互补量: non_drone 的 precision/recall 是 drone 的")
    print("     '特异度视角' (R(non)=特异度, P(non)=负类预测精确率),")
    print("     不能用 1-P(non) 或 1-R(non) 反推 drone 的查准/查全。")
    print(f"  3) 类别不平衡 (1:{n_neg / max(n_pos, 1):.2f}) 的影响: non_drone 是多数类,")
    print("     AP(non_drone) 数值天然偏高, 以它为主任务评估会'虚高'; 而 drone 的漏检")
    print("     藏在 non_drone 的 FP 里, 主任务为 non_drone 时最容易被忽视。")
    print("  4) 结论: 以检测 drone 为主任务 (方案 A) 是正确选择; 方案 B 仅适合作为")
    print("     特异度审计 (检查虚警来源), 不应作为主评估口径。")
    print("=" * 70)

    if args.output:
        payload = {"n_pos": n_pos, "n_neg": n_neg,
                   "scheme_a": {**m_a, **cm_a, "ap_drone": ap_drone},
                   "scheme_b_non": {**m_b_non, **cm_b_non, "ap_non_drone": ap_non},
                   "scheme_b_inferred_drone": {**m_b, **cm_b},
                   "identical_at_thr": bool(identical),
                   "map": (ap_drone + ap_non) / 2}
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"结果已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
