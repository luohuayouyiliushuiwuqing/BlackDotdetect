"""模型生成后的理论运行验证 (正确性冒烟检查).

不依赖真实部署, 用合成数据验证训练好的 DroneNet 是否"行为正确":
  1) 权重能否正确加载 (架构/配置一致, strict 加载);
  2) 前向输出是否为合法概率 (形状 (N,), 取值 [0,1], 无 NaN/Inf);
  3) 判别是否符合预期 (正样本均分 > 负样本均分, 合成集 AUC > 0.5);
  4) 打印合成集指标作为参考。

用法: python scripts/validate_model.py --ckpt runs/finetune.pt [--n 200] [--device cpu]
退出码: 全部通过 0, 否则 1。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blackdot.dataset import encode_roi
from blackdot.metrics import binary_metrics, drone_pr
from blackdot.model import DroneNet
from blackdot.synthesis import synth_sample


def load_model(ckpt_path: str, device: torch.device) -> DroneNet:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DroneNet().to(device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)          # strict: 架构不一致应直接报错
    model.eval()
    return model


def _report(checks, extra=None) -> None:
    print("=== 理论运行验证 ===")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if extra:
        print("  --- 合成集指标 ---")
        for k, v in extra.items():
            print(f"    {k}: {v:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="训练保存的 .pt 路径")
    ap.add_argument("--n", type=int, default=200, help="每类合成样本数")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    device = torch.device(args.device)
    checks: list[tuple[str, bool, str]] = []

    # 1) 加载
    try:
        model = load_model(args.ckpt, device)
        checks.append(("权重加载(架构一致)", True, "DroneNet 纯 CNN"))
    except Exception as e:  # noqa: BLE001
        checks.append(("权重加载(架构一致)", False, str(e)))
        _report(checks)
        return 1

    # 2) 输入数据准备 (合成, 覆盖 4 类语义 -> 二分类标签)
    rng = np.random.default_rng(args.seed)
    xs, ys = [], []
    for cls in range(4):
        for _ in range(args.n):
            img = synth_sample(rng, cls)
            xs.append(encode_roi(img))
            ys.append(1 if cls == 0 else 0)
    x = torch.from_numpy(np.stack(xs)).to(device)
    y = np.array(ys)

    # 3) 前向推理
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).cpu().numpy()

    checks.append(("输出形状 (N,)", prob.shape == (len(y),), f"got {prob.shape}"))
    checks.append(("无 NaN/Inf", bool(np.isfinite(prob).all()), ""))
    checks.append(("概率取值 ∈ [0,1]",
                   bool((prob >= 0).all() and (prob <= 1).all()),
                   f"min={prob.min():.4f} max={prob.max():.4f}"))

    # 4) 判别正确性
    pos_m, neg_m = float(prob[y == 1].mean()), float(prob[y == 0].mean())
    checks.append(("正样本均分 > 负样本均分", pos_m > neg_m,
                   f"pos={pos_m:.4f} neg={neg_m:.4f}"))
    auc = drone_pr(y, prob).get("auc", float("nan"))
    checks.append(("合成集 AUC > 0.5", bool(np.isfinite(auc) and auc > 0.5),
                   f"AUC={auc:.4f}"))
    bm = binary_metrics(y, prob)
    checks.append(("合成集 F1 > 0.3 (冒烟下限)", bm["f1"] > 0.3, f"F1={bm['f1']:.4f}"))

    _report(checks, extra={**bm, "auc": auc})
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
