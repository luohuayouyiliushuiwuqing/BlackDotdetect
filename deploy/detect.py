"""黑色小目标 (drone) ONNX 推理部署 —— 自包含, 仅依赖 numpy / opencv / onnxruntime.

模型契约 (固定):
    输入 : 36×36 像素方形 Y 分量(灰度)图像
    输出 : ① 类别 是否 drone (class_name / is_drone)
           ② 置信度 confidence (该类别的置信度分数)

整个目录可整体拷贝到任意机器/目录直接运行, 路径不硬编码
(相对路径以配置文件或脚本所在目录为基准), 不依赖 PyTorch / 训练代码。

用法:
    # 单张 36x36 ROI
    python detect.py --model model.onnx --input roi.png
    # 整个目录批量推理 (所有 36x36 ROI 逐张给出类别与置信度)
    python detect.py --model model.onnx --input ./rois/ --output result.json
    # 通过配置文件指定模型路径/输入尺寸/阈值
    python detect.py --config config.json --input roi.png --conf 0.8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import cv2
import onnxruntime as ort

CLASS_NAMES = {0: "non_drone", 1: "drone"}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def resolve_path(p: str | Path, base: Path) -> Path:
    """相对路径按 base 解析 (配置内相对路径以配置文件所在目录为基准)."""
    p = Path(p)
    return p if p.is_absolute() else (base / p)


class DroneClassifier:
    """固定 36×36 Y 分量输入 -> (是否 drone, 置信度)."""

    def __init__(self, model_path: str | Path, input_size: int = 36,
                 providers: list[str] | None = None):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        self.input_size = int(input_size)
        self.sess = ort.InferenceSession(str(model_path),
                                         providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        names = {i.name for i in self.sess.get_inputs()}
        if "feat" in names:
            raise RuntimeError(
                "该 ONNX 需要 8 维手工特征输入 'feat'(依赖训练期实现)。\n"
                "部署请用纯 CNN 模型: python scripts/train.py --no-handcrafted ...\n"
                "再用 scripts/export_onnx.py 导出 (输入签名应只有 'roi')。")
        if self.input_name != "roi":
            raise RuntimeError(f"意外的输入签名: {names}, 期望 'roi'")

    # -------- 预处理 (与训练期 encode_roi 严格一致: 只缩不放) --------
    def preprocess(self, crops: list[np.ndarray]) -> np.ndarray:
        """灰度 ROI 列表 -> (N, 2, H, W) float32 (通道0=Y/255, 通道1=有效像素掩码)."""
        H = W = self.input_size
        out = np.zeros((len(crops), 2, H, W), dtype=np.float32)
        for i, crop in enumerate(crops):
            h, w = crop.shape[:2]
            if max(h, w) > self.input_size:  # 超界: 等比缩小(保留全部内容)
                s = self.input_size / float(max(h, w))
                nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
                patch = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
            else:  # 不足: 原尺寸居中放入
                patch = crop
            ph, pw = patch.shape[:2]
            ch, cw = min(ph, H), min(pw, W)
            sy, sx = (ph - ch) // 2, (pw - cw) // 2
            oy, ox = (H - ch) // 2, (W - cw) // 2
            out[i, 0, oy:oy + ch, ox:ox + cw] = patch[sy:sy + ch, sx:sx + cw] / 255.0
            out[i, 1, oy:oy + ch, ox:ox + cw] = 1.0
        return out

    def predict_proba(self, crops: list[np.ndarray], batch_size: int = 512) -> np.ndarray:
        """批量返回 P(drone)."""
        if not crops:
            return np.zeros(0, dtype=np.float32)
        probs = []
        for i in range(0, len(crops), batch_size):
            x = self.preprocess(crops[i:i + batch_size])
            p = self.sess.run([self.output_name], {self.input_name: x})[0]
            probs.append(np.asarray(p).reshape(-1))
        return np.concatenate(probs)

    # -------- 核心接口: 36x36 ROI -> 类别 + 置信度 --------
    def predict(self, roi: np.ndarray, conf_thr: float = 0.5) -> dict:
        """输入 36×36 Y 分量 ROI, 返回类别与置信度.

        返回字段:
            score      = P(drone) 原始概率
            is_drone     = score >= conf_thr
            class_id   = 1(drone) / 0(non_drone)
            class_name = "drone" / "non_drone"
            confidence = 预测类别的置信度 (drone 为 score, non_drone 为 1-score)
        """
        if roi.ndim == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        score = float(self.predict_proba([roi], batch_size=1)[0])
        is_drone = score >= conf_thr
        cls_id = 1 if is_drone else 0
        return {"is_drone": bool(is_drone), "class_id": cls_id,
                "class_name": CLASS_NAMES[cls_id],
                "confidence": round(score if is_drone else 1.0 - score, 4),
                "score": round(score, 4)}

    def predict_folder(self, folder: str | Path, conf_thr: float = 0.5,
                       batch_size: int = 512, recursive: bool = False) -> list[dict]:
        """批量推理一个目录下的所有图像, 每张输出类别与置信度."""
        folder = Path(folder)
        pattern = "**/*" if recursive else "*"
        files = [f for f in sorted(folder.glob(pattern))
                 if f.is_file() and f.suffix.lower() in IMG_EXTS]
        if not files:
            raise FileNotFoundError(f"{folder} 下未找到图像 ({IMG_EXTS})")
        imgs = []
        for f in files:
            im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if im is None:
                raise ValueError(f"无法读取 {f}")
            imgs.append(im)
        probs = self.predict_proba(imgs, batch_size=batch_size)
        out = []
        for f, im, p in zip(files, imgs, probs):
            score = float(p)
            is_drone = score >= conf_thr
            cls_id = 1 if is_drone else 0
            out.append({"file": f.name, "size": f"{im.shape[1]}x{im.shape[0]}",
                        "is_drone": bool(is_drone), "class_id": cls_id,
                        "class_name": CLASS_NAMES[cls_id],
                        "confidence": round(score if is_drone else 1.0 - score, 4),
                        "score": round(score, 4)})
        return out


# ---------------------------------------------------------------- 配置 / CLI
DEFAULT_CONFIG = {
    "model": "model.onnx",
    "input_size": 36,
    "conf_threshold": 0.5,
    "batch_size": 512,
}


def load_config(path, base: Path) -> tuple[dict, Path]:
    cfg = dict(DEFAULT_CONFIG)
    if path:
        p = resolve_path(path, Path.cwd())
        if not p.exists():
            raise FileNotFoundError(f"配置文件不存在: {p}")
        cfg.update(json.loads(p.read_text(encoding="utf-8")))
        base = p.parent
    return cfg, base


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="drone推理: 36x36 Y 分量图像 -> 类别 + 置信度")
    ap.add_argument("--config", default=None, help="JSON 配置 (默认取本目录 config.json)")
    ap.add_argument("--model", default=None, help="ONNX 模型路径 (覆盖配置)")
    ap.add_argument("--input", required=True,
                    help="36x36 ROI 图像, 或包含多张 ROI 的目录 (批量)")
    ap.add_argument("--input-size", type=int, default=None, help="模型输入边长 (覆盖配置)")
    ap.add_argument("--conf", type=float, default=None, help="判定为 drone 的置信度阈值")
    ap.add_argument("--batch-size", type=int, default=None, help="批量推理大小")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--output", default=None, help="结果 JSON 输出路径")
    return ap.parse_args()


NAME_W, SIZE_W, CLS_W = 40, 7, 10  # 按你的数据分布固定好，改这里即可


def fmt_row(name, w, h, cls, conf, plabel, pval):
    if len(name) > NAME_W:  # 超长截断，防止撑破对齐
        name = name[: NAME_W - 3] + "..."
    size = f"{w}x{h}"
    return (f"  {name:<{NAME_W}}  {size:>{SIZE_W}}  "
            f"类别={cls:<{CLS_W}} 置信度={conf:.4f}  P({plabel})={pval:.4f}")


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    default_cfg = script_dir / "config.json"
    cfg, base = load_config(args.config or (default_cfg if default_cfg.exists() else None),
                            script_dir)
    for key, val in (("model", args.model), ("input_size", args.input_size),
                     ("conf_threshold", args.conf), ("batch_size", args.batch_size)):
        if val is not None:
            cfg[key] = val

    model_path = resolve_path(cfg["model"], base)
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if args.device == "cuda" else ["CPUExecutionProvider"])
    clf = DroneClassifier(model_path, input_size=int(cfg["input_size"]), providers=providers)
    conf_thr = float(cfg["conf_threshold"])

    input_path = resolve_path(args.input, Path.cwd())
    if input_path.is_dir():
        results = clf.predict_folder(input_path, conf_thr=conf_thr,
                                     batch_size=int(cfg["batch_size"]))
        n_drone = sum(r["is_drone"] for r in results)
        print(f"模型: {model_path}  输入尺寸: {cfg['input_size']}x{cfg['input_size']}")
        print(f"目录: {input_path}  共 {len(results)} 张, 判定 drone {n_drone} 张 / "
              f"non_drone {len(results) - n_drone} 张 (阈值 {conf_thr})")
        for r in results:
            print(fmt_row(r["file"], *map(int, r["size"].split("x")),
                          r["class_name"], r["confidence"], "drone", r["score"]))
        result = {"mode": "batch", "dir": str(input_path), "threshold": conf_thr,
                  "n_drone": n_drone, "n_non_drone": len(results) - n_drone, "items": results}
    else:
        if not input_path.exists():
            raise FileNotFoundError(f"输入图像不存在: {input_path}")
        image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"无法读取图像: {input_path}")
        r = clf.predict(image, conf_thr=conf_thr)
        print(f"模型: {model_path}  输入尺寸: {cfg['input_size']}x{cfg['input_size']}")
        print(f"图像: {input_path} ({image.shape[1]}x{image.shape[0]})")
        print(f"  类别: {r['class_name']} (is_drone={r['is_drone']})")
        print(f"  置信度: {r['confidence']:.4f}   P(drone)={r['score']:.4f}")
        result = {"mode": "single", "image": str(input_path), **r}

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"结果已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
