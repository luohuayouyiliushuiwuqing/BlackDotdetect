# dot 推理部署包（自包含）

黑色小目标（dot）ONNX 推理。**模型契约：输入 36×36 Y 分量（灰度）图像 → 输出类别（dot / non_dot）+ 置信度。**

仅依赖 `numpy / opencv / onnxruntime`，**不含 PyTorch、不 import 任何训练代码**。
整个目录（本 README + detect.py + config.json + model.onnx + samples/ + demo.py）
拷贝到任意机器/目录即可直接运行，路径不硬编码。

## 目录结构

```
deploy/
├── detect.py          # 推理代码: 模型加载 + 预处理 + 推理 + 后处理(阈值判定)
├── config.json        # 配置: 模型路径 / 输入尺寸 / 置信度阈值 / 批大小
├── model.onnx         # ONNX 模型 (单输入 roi(2,36,36) -> 单输出 prob)
├── samples/           # 内置验证样本 (真实 ROI: 6 dot + 3 non_dot, 尺寸不一)
├── demo.py            # 一键部署验证
└── requirements.txt   # 依赖清单
```

## 安装

```bash
pip install -r requirements.txt
# 即 numpy + opencv-python-headless + onnxruntime (无其他依赖)
```

## 一键验证

```bash
python demo.py          # 退出码 0 = 部署可用
```

会用 `samples/` 中 9 张真实 ROI 验证: ① 单张分类与真值一致; ② 批量路径与单张一致。

## 使用

```bash
# 单张 ROI
python detect.py --model model.onnx --input roi.png

# 批量: 目录下所有图像逐张推理
python detect.py --input ./rois/ --output result.json

# 通过参数覆盖配置 (模型路径 / 输入尺寸 / 阈值)
python detect.py --model /data/model.onnx --input roi.png --input-size 36 --conf 0.8 --device cuda
```

输出示例:

```
  类别: dot (is_dot=True)
  置信度: 0.9585   P(dot)=0.9585
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
