"""ONNX 推理基准: 在目标硬件上实测延迟/吞吐, 为"是否需要 NPU 量化"提供依据.

分别测量三段耗时:
  A) 预处理 (encode_roi: 缩放/居中/归一化, 纯 CPU/OpenCV);
  B) 模型推理 (ONNX Runtime CPU, 多个批大小);
  C) 端到端 (batch=30: 预处理 + 推理), 按每 ROI 口径对照预算。

用法: python scripts/benchmark_onnx.py --onnx runs/dronenet.onnx [--sizes 1,30,256] [--repeats 50]
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
    ap.add_argument("--budget-ms", type=float, default=2.0, help="端到端每 ROI 时延预算")
    args = ap.parse_args()

    import onnxruntime as ort
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    if "feat" in {i.name for i in sess.get_inputs()}:
        print("[错误] ONNX 含手工特征输入 'feat', 请使用纯 CNN 模型重新导出")
        return 1

    rng = np.random.default_rng(0)
    imgs = [synth_sample(rng, i % 4) for i in range(256)]

    # A) 预处理基准 (单 ROI)
    t_prep = _bench(lambda: [encode_roi(im) for im in imgs[:30]],
                    max(args.repeats // 2, 5)) / 30
    print(f"A) 预处理 encode_roi(缩放/居中/归一化): {t_prep:.3f} ms/ROI")

    # B) 模型推理基准
    print("B) 模型推理 (ONNX Runtime CPU):")
    for bs in [int(s) for s in args.sizes.split(",")]:
        x = np.random.rand(bs, 2, ROI, ROI).astype(np.float32)
        t = _bench(lambda: sess.run(["prob"], {"roi": x}), args.repeats)
        print(f"   batch={bs:3d}: {t:8.3f} ms/批  ({t / bs * 1000:7.1f} us/ROI, "
              f"{bs / t * 1000:8.0f} ROI/s)")

    # C) 端到端 (batch=30: 预处理 + 推理), 按每 ROI 对照预算
    def _e2e(batch_imgs):
        xs = [encode_roi(im) for im in batch_imgs]
        return sess.run(["prob"], {"roi": np.stack(xs)})

    n_e2e = 30
    t_e2e = _bench(lambda: _e2e(imgs[:n_e2e]), max(args.repeats // 2, 5))
    per_roi = t_e2e / n_e2e
    verdict = "达标" if per_roi <= args.budget_ms else \
        "不达标 -> 建议 NPU INT8 / GPU FP16 / 放宽预算"
    print(f"C) 端到端 batch={n_e2e} (预处理+推理): 批总 {t_e2e:.3f} ms, "
          f"折合 {per_roi:.3f} ms/ROI (预算 {args.budget_ms} ms/ROI -> {verdict})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
