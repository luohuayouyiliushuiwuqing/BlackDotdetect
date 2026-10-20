"""将训练好的 DroneNet 导出为 ONNX.

用法: python scripts/export_onnx.py --ckpt runs/finetune.pt --out runs/dronenet.onnx [--opset 17]
依赖: pip install onnx
导出后用 scripts/verify_onnx.py 校验与 PyTorch 输出一致性。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blackdot.features import NUM_HANDCRAFTED
from blackdot.model import DroneNet
from blackdot.synthesis import ROI


class _ProbWrapper(torch.nn.Module):
    """包装 DroneNet: 输出 sigmoid 概率而非 logit, 便于部署与对比。"""

    def __init__(self, net: DroneNet):
        super().__init__()
        self.net = net

    def forward(self, roi, feat):
        return torch.sigmoid(self.net(roi, feat))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="训练保存的 .pt 路径")
    ap.add_argument("--out", default="runs/dronenet.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    try:
        import onnx
    except ImportError:
        print("缺少 onnx 依赖, 请先: pip install onnx")
        return 1

    device = torch.device("cpu")  # 固定 CPU 导出, 保证可复现
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    meta = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    use_hc = not meta.get("no_handcrafted", False)
    net = DroneNet(use_handcrafted=use_hc).to(device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    net.load_state_dict(state)
    net.eval()
    model = _ProbWrapper(net)

    dummy_roi = torch.zeros(1, 2, ROI, ROI)
    dummy_feat = torch.zeros(1, NUM_HANDCRAFTED)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, (dummy_roi, dummy_feat), str(out),
        input_names=["roi", "feat"], output_names=["prob"],
        dynamic_axes={"roi": {0: "batch"}, "feat": {0: "batch"}, "prob": {0: "batch"}},
        opset_version=args.opset, do_constant_folding=True,
    )

    # torch>=2.9 的新导出器会把大权重写成外部数据文件 (*.onnx.data),
    # 这里合并回单个自包含 .onnx, 避免部署时漏带伴生文件。
    m = onnx.load(str(out))
    from onnx.external_data_helper import convert_model_from_external_data
    convert_model_from_external_data(m)
    onnx.save_model(m, str(out))
    data_file = Path(str(out) + ".data")
    if data_file.exists():
        data_file.unlink()
    onnx.checker.check_model(m)
    print(f"ONNX 已导出并通过结构校验: {out}")
    print(f"  输入: roi (batch,2,{ROI},{ROI}), feat (batch,{NUM_HANDCRAFTED})")
    print("  输出: prob (batch,)  # sigmoid 概率")
    print(f"  opset={args.opset}, use_handcrafted={use_hc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
