"""ONNX 导出正确性验证: 对比 ONNX 与 PyTorch 输出是否一致.

关键步骤:
  1) 输入数据准备: 合成真实分布样本(覆盖 4 类语义) + 随机输入(边界覆盖), 组装成批;
  2) 推理结果对比: 同一批输入分别过 PyTorch 与 ONNX Runtime, 逐样本比较概率;
  3) 误差容忍: 统计 max_abs_diff / mean_abs_diff, 默认容忍 1e-4 (float32 精度损失);
     另比较 0.5 阈值下的分类决策是否完全一致。

用法: python scripts/verify_onnx.py --ckpt runs/finetune.pt --onnx runs/dronenet.onnx [--tol 1e-4] [--n 64]
依赖: pip install onnxruntime
退出码: 一致 0, 不一致 1。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blackdot.dataset import encode_roi
from blackdot.features import NUM_HANDCRAFTED
from blackdot.model import DroneNet
from blackdot.synthesis import ROI, synth_sample


def _load_torch(ckpt_path: str, device: torch.device) -> DroneNet:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    use_hc = not meta.get("no_handcrafted", False)
    net = DroneNet(use_handcrafted=use_hc).to(device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    net.load_state_dict(state)
    net.eval()
    return net


def _prepare_inputs(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """输入数据准备: 合成样本(真实分布) + 随机输入(极端值域, 检验数值鲁棒性)."""
    rng = np.random.default_rng(seed)
    xs, feats = [], []
    for cls in range(4):                       # 合成: 覆盖 4 类语义
        for _ in range(n):
            x, feat = encode_roi(synth_sample(rng, cls))
            xs.append(x)
            feats.append(feat)
    for _ in range(n):                          # 随机: 边界覆盖
        xs.append(np.stack([rng.uniform(0, 1, (ROI, ROI)).astype(np.float32),
                            np.ones((ROI, ROI), np.float32)]))
        feats.append(rng.uniform(0, 1, NUM_HANDCRAFTED).astype(np.float32))
    return np.stack(xs).astype(np.float32), np.stack(feats).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--tol", type=float, default=1e-4, help="max_abs_diff 容忍阈值")
    ap.add_argument("--n", type=int, default=64, help="每类/每组样本数")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        print("缺少 onnxruntime 依赖, 请先: pip install onnxruntime")
        return 1

    device = torch.device("cpu")  # 对比统一在 CPU, 消除设备差异
    net = _load_torch(args.ckpt, device)
    x, feat = _prepare_inputs(args.n, args.seed)

    # ONNX Runtime 会话 (先建, 便于读取输入签名: 纯 CNN 版只需 roi)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    use_feat = "feat" in {i.name for i in sess.get_inputs()}

    # PyTorch 推理
    with torch.no_grad():
        f_t = torch.from_numpy(feat).to(device) if use_feat else None
        prob_t = torch.sigmoid(net(torch.from_numpy(x).to(device), f_t)).cpu().numpy()

    # ONNX Runtime 推理
    feed = {"roi": x}
    if use_feat:
        feed["feat"] = feat
    prob_o = sess.run(["prob"], feed)[0]

    # 推理结果对比
    diff = np.abs(prob_t - prob_o)
    max_abs, mean_abs = float(diff.max()), float(diff.mean())
    same_cls = bool(((prob_t >= 0.5) == (prob_o >= 0.5)).all())

    print("=== ONNX vs PyTorch 一致性验证 ===")
    print(f"  样本数: {len(x)}")
    print(f"  max_abs_diff : {max_abs:.3e}  (容忍阈值 tol={args.tol:.1e})")
    print(f"  mean_abs_diff: {mean_abs:.3e}")
    print(f"  0.5 阈值分类决策一致: {same_cls}")

    ok = (max_abs <= args.tol) and same_cls
    print(f"  结果: {'PASS' if ok else 'FAIL'}")
    if not ok:
        worst = int(np.argmax(diff))
        print(f"  最差样本 idx={worst}: torch={prob_t[worst]:.6f} onnx={prob_o[worst]:.6f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
