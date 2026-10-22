"""用训练好的模型评估指定数据划分 (默认 test 集), 并给出失败样本清单.

训练脚本只吃 train/val, test 集必须单独评估, 否则"留出测试集"形同虚设。

用法:
    python scripts/evaluate.py --ckpt runs/finetune.pt --data-root data --split test
    python scripts/evaluate.py --ckpt runs/finetune.pt --data-root data --split val --thr 0.3
    python scripts/evaluate.py --ckpt runs/finetune.pt --data-root data --split test --sweep
                                # --sweep: 打印阈值扫描表, 用于选部署工作点
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blackdot.dataset import FolderROIDataset
from blackdot.metrics import binary_metrics, bootstrap_ci, drone_pr, report
from blackdot.model import DroneNet


def load_model(ckpt: str, device: torch.device) -> DroneNet:
    ckpt_data = torch.load(ckpt, map_location=device, weights_only=False)
    net = DroneNet().to(device)
    state = ckpt_data["model"] if isinstance(ckpt_data, dict) and "model" in ckpt_data \
        else ckpt_data
    net.load_state_dict(state)          # strict: 架构不一致应直接报错
    net.eval()
    return net


@torch.no_grad()
def run(model, loader, device):
    ys, probs = [], []
    for x, label in loader:
        x = x.to(device)
        p = torch.sigmoid(model(x)).cpu().numpy()
        probs.append(p)
        ys.append(label.numpy().astype(int))
    return np.concatenate(ys), np.concatenate(probs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--thr", type=float, default=0.5, help="判定阈值")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--sweep", action="store_true", help="打印阈值扫描表")
    ap.add_argument("--show-errors", type=int, default=10, help="打印前 N 个误判文件名")
    ap.add_argument("--boot", type=int, default=300,
                    help="Bootstrap 重采样次数 (0=关闭置信区间)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    root = Path(args.data_root) / args.split
    ds = FolderROIDataset(root, train=False)
    # 保留文件名用于失败分析
    paths = [p for p, _ in ds.samples]
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=2)

    model = load_model(args.ckpt, device)
    y, prob = run(model, loader, device)

    print(f"=== 评估: {root} (n={len(y)}, 正样本 {int((y == 1).sum())}, "
          f"负样本 {int((y == 0).sum())}) ===")
    print(report(y, prob, thr=args.thr))

    if args.boot > 0:
        auc_lo, auc_hi = bootstrap_ci(y, prob, lambda a, b: drone_pr(a, b)["auc"],
                                      n_boot=args.boot)
        f1_lo, f1_hi = bootstrap_ci(
            y, prob, lambda a, b: binary_metrics(a, b, thr=args.thr)["f1"],
            n_boot=args.boot)
        print(f"\n95% 置信区间 (bootstrap {args.boot} 次): "
              f"AUC [{auc_lo:.3f}, {auc_hi:.3f}], "
              f"F1(thr={args.thr}) [{f1_lo:.3f}, {f1_hi:.3f}]")
        if auc_hi - auc_lo > 0.15:
            print("  ⚠️ 置信区间很宽: 样本量太小, 单次划分的指标不足以下结论, "
                  "建议做 K 折 / 留一段交叉验证")

    pred = prob >= args.thr
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    print(f"\n混淆矩阵 (thr={args.thr}): TP={tp} FP={fp} FN={fn} TN={tn}")

    if args.show_errors:
        fps = [paths[i] for i in np.where((pred == 1) & (y == 0))[0][:args.show_errors]]
        fns = [paths[i] for i in np.where((pred == 0) & (y == 1))[0][:args.show_errors]]
        if fps:
            print(f"\n误报 FP (前 {len(fps)}):")
            for p in fps:
                print(f"  {p}")
        if fns:
            print(f"\n漏检 FN (前 {len(fns)}):")
            for p in fns:
                print(f"  {p}")

    if args.sweep:
        print("\n阈值扫描 (用于选部署工作点):")
        print(f"{'thr':>6}{'P':>8}{'R':>8}{'F1':>8}{'FAR/ROI':>10}")
        for thr in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            m = binary_metrics(y, prob, thr=thr)
            print(f"{thr:>6.2f}{m['precision']:>8.3f}{m['recall']:>8.3f}"
                  f"{m['f1']:>8.3f}{m['far_per_roi']:>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
