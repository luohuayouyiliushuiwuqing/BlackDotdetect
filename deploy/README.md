# drone 推理部署包（自包含）

黑色小目标（drone）ONNX 推理。**模型契约：输入 36×36 Y 分量（灰度）图像 → 输出类别（drone / non_drone）+ 置信度。**

仅依赖 `numpy / opencv / onnxruntime`，**不含 PyTorch、不 import 任何训练代码**。

## 目录结构

```
deploy/
├── detect.py          # 推理代码: 模型加载 + 预处理 + 推理 + 阈值判定
└── requirements.txt   # 依赖清单
```

模型默认取 `<项目>/runs/dronenet.onnx`（也可用 `--model` 或 `--config` 指定）。

## 安装

```bash
pip install -r requirements.txt
# 即 numpy + opencv-python-headless + onnxruntime (无其他依赖)
```

## 标准用法（在项目根目录执行）

```bash
python deploy/detect.py \
    --input  /home/igs/yhj_demo/PythonProject/BlackDotDataset/data/non_dot/ \
    --model  /home/igs/yhj_demo/PythonProject/BlackDotdetect/runs/dronenet.onnx
```

- `--input` 传**目录**则批量推理目录下所有 ROI（逐张打印类别与置信度，最后给汇总）；
  传**单张图像**则只推理这一张。
- 未传 `--model` 时依次回退：`deploy/model.onnx` → `runs/dronenet.onnx`。
- `--recursive` 递归扫描子目录；`--output result.json` 保存结构化结果；
  `--conf 0.8` 提高判定阈值（test 实测 0.8 时零误报）；`--device cuda` 用 GPU。

## 编程接口

```python
from deploy.detect import DroneClassifier

clf = DroneClassifier("runs/dronenet.onnx", input_size=36)
r = clf.predict(roi_gray)     # {"is_drone": True, "class_name": "drone",
                              #  "confidence": 0.96, "score": 0.96}
items = clf.predict_folder("dir/", conf_thr=0.5)   # 目录批量
```

## 编程接口

```python
from detect import DotClassifier

clf = DotClassifier("model.onnx", input_size=36)   # cuda: providers=["CUDAExecutionProvider"]
r = clf.predict(roi_gray)                          # roi_gray: 36x36 灰度图 (numpy)
# r = {"is_dot": True, "class_id": 1, "class_name": "dot",
#      "confidence": 0.96, "score": 0.96}          # score = P(dot) 原始概率

probs = clf.predict_proba(list_of_rois)            # 批量 P(dot)
```

## 预处理规则（与训练一致）

- 超过 36×36：等比缩小到最长边 36（`INTER_AREA`），不裁内容；
- 不足 36×36：**原尺寸**居中放入画布，四周填 0，掩码通道置 0 —— **绝不放大**；
- 像素归一化 `/255`；彩色输入自动转灰度 Y。

## 阈值选择（实拍 test 集实测）

| conf_threshold | P | R | 特点 |
|---|---|---|---|
| 0.5 | 0.986 | 1.000 | F1 最优 |
| 0.8 | 1.000 | 0.917 | 零误报 |

> 注意：本模型是 ROI 级分类器，输入应已是目标居中的 36×36 ROI（由上游检测/滑窗给出）。
> 上游 ROI 尺寸不必严格 36（13~36 均可，代码自动居中缩放/填充）。

## 换模型 / 换输入尺寸

```bash
# 训练侧导出 (项目根目录):
python scripts/train.py --data-root data --no-handcrafted ... --out runs/m.pt
python scripts/export_onnx.py --ckpt runs/m.pt --out deploy/model.onnx
```

- 部署要求 ONNX 为**纯 CNN 单输入签名** `roi(batch,2,H,W)`（训练时加 `--no-handcrafted`）；
  若误导出带 `feat` 输入的版本，本脚本会报错提示。
- 换输入尺寸 H：更新 `config.json` 的 `input_size` 或 `--input-size`（须与导出模型一致）。
