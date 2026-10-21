"""ONNX 推理基准: 在目标硬件上实测延迟/吞吐, 为"是否需要 NPU 量化"提供依据.

分别测量三段耗时:
  A) 预处理 (encode_roi: resize/归一化 + 8 维手工特征, 纯 CPU/OpenCV);
  B) 模型推理 (ONNX Runtime CPU, 多个批大小);
  C) 端到端 (batch=30: 预处理 + 推理), 直接对照 2ms 预算。

用法: python scripts/benchmark_onnx.py --onnx runs/smoke_ft.onnx [--sizes 1,30,256] [--repeats 50]
在 RK3588 等目标板上运行本脚本, 即可判断 CPU 方案是否满足时延预算。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blackdot.dataset import encode_roi
from blackdot.synthesis import ROI, synth_sample


def _bench(fn, repeats: int) -> float:
    fn()  # 预热
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats * 1000.0  # ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--sizes", default="1,30,256", help="逗号分隔的批大小列表")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--budget-ms", type=float, default=2.0, help="端到端时延预算")
    args = ap.parse_args()

    import onnxruntime as ort
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    use_feat = "feat" in {i.name for i in sess.get_inputs()}

    def _encode(img):
        """按模型签名编码: 需 feat 时算 8 维手工特征, 纯 CNN 版只做 resize+归一化。"""
        if use_feat:
            return encode_roi(img)
        h, w = img.shape
        canvas = np.zeros((ROI, ROI), np.float32)
        mask = np.zeros((ROI, ROI), np.float32)
        hh, ww = min(h, ROI), min(w, ROI)
        canvas[:hh, :ww] = img[:hh, :ww] / 255.0
        mask[:hh, :ww] = 1.0
        return np.stack([canvas, mask]), None

    def _feed(xb, fb):
        d = {"roi": xb}
        if use_feat:
            d["feat"] = fb
        return d

    rng = np.random.default_rng(0)
    imgs = [synth_sample(rng, i % 4) for i in range(256)]

    # A) 预处理基准 (单 ROI)
    tag = "encode_roi(含8维手工特征)" if use_feat else "resize+归一化(纯CNN,无手工特征)"
    t_prep = _bench(lambda: [_encode(im) for im in imgs[:30]],
                    max(args.repeats // 2, 5)) / 30
    print(f"A) 预处理 {tag}: {t_prep:.3f} ms/ROI")

    # B) 模型推理基准
    print("B) 模型推理 (ONNX Runtime CPU):")
    for bs in [int(s) for s in args.sizes.split(",")]:
        x = np.random.rand(bs, 2, ROI, ROI).astype(np.float32)
        f = np.random.rand(bs, 8).astype(np.float32)
        t = _bench(lambda: sess.run(["prob"], _feed(x, f)), args.repeats)
        print(f"   batch={bs:3d}: {t:8.3f} ms/批  ({t / bs * 1000:7.1f} us/ROI, "
              f"{bs / t * 1000:8.0f} ROI/s)")

    # C) 端到端 (batch=30: 预处理 + 推理), 对照时延预算
    def _e2e(batch_imgs):
        xs, fs = [], []
        for im in batch_imgs:
            x, feat = _encode(im)
            xs.append(x)
            if feat is not None:
                fs.append(feat)
        fb = np.stack(fs) if use_feat else None
        return sess.run(["prob"], _feed(np.stack(xs), fb))

    n_e2e = 30
    t_e2e = _bench(lambda: _e2e(imgs[:n_e2e]), max(args.repeats // 2, 5))
    # 预算是"每 ROI"口径, 必须与批总耗时对齐后再比较
    per_roi = t_e2e / n_e2e
    verdict = "达标" if per_roi <= args.budget_ms else \
        "不达标 -> 建议 NPU INT8 / GPU FP16 / 放宽预算"
    print(f"C) 端到端 batch={n_e2e} (预处理+推理): 批总 {t_e2e:.3f} ms, "
          f"折合 {per_roi:.3f} ms/ROI (预算 {args.budget_ms} ms/ROI -> {verdict})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
