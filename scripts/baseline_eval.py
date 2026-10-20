"""规则基线评估 (设计方案 3.3 / 6.2 基线 1).

在合成测试集上运行规则基线, 输出与 CNN 相同的评估报告, 用于对比。
用法: python scripts/baseline_eval.py --n-per-class 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blackdot.features import rule_baseline
from blackdot.metrics import report
from blackdot.synthesis import synth_sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=500)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    ys, bs, preds, dscores = [], [], [], []
    for cls in range(4):
        for _ in range(args.n_per_class):
            img = synth_sample(rng, cls)
            r = rule_baseline(img)
            ys.append(cls)
            bs.append(1.0 if r["black_exist"] else 0.0)
            preds.append(r["cls"])
            # 规则法无概率输出, 用对比度近似无人机得分
            dscores.append(float(r["features"][3]) if r["cls"] == 0 else 0.0)

    print(report(np.array(ys), np.array(bs), np.array(preds), np.array(dscores)))


if __name__ == "__main__":
    main()
