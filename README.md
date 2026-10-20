# BlackDotDetect — 远距离无人机黑色小目标 ROI 二分类检测

对 36×36 像素 Y 分量 ROI（目标居中、外圈为背景）判定其中是否存在**无人机/飞机**这类黑色小目标，输出二分类概率。设计细节见 `docs/黑色小目标检测模型设计方案.md`（注：项目现已改造为二分类任务，原"四类区分"改为"无人机 vs 其余"）。**训练与验证的完整流程见 `docs/训练与验证指南.md`。**

## 环境

```bash
conda activate yolov8   # /home/igs/miniconda3/envs/yolov8, Python 3.10 + torch 2.7
```

## 目录结构

```
blackdot/
  features.py    # 暗像素判定规则基线 + 8 维手工结构特征
  synthesis.py   # 四类样本合成管线 (无人机为正/其余为负; 输出 36x36; 统一退化链)
  dataset.py     # 在线合成 / 实拍 ROI 文件夹 Dataset (36x36 输入) + 增强
  model.py       # 轻量二分类 CNN (~7.6 万参数) + 二分类 Focal Loss
  metrics.py     # 二分类 PR / AUC / 固定 FAR 下召回率
scripts/
  baseline_eval.py  # 规则基线评估 (对照组)
  train.py          # 训练 + 验证评估
  validate_model.py # 训练产物正确性冒烟检查
  export_onnx.py    # 导出 ONNX
  verify_onnx.py    # ONNX 与 PyTorch 一致性校验
```

## 快速开始

规则基线（对照组，验证"传统算法无法区分"）：

```bash
python scripts/baseline_eval.py --n-per-class 500
```

训练 CNN（无实拍数据时使用在线合成数据）。**推荐加 `--no-handcrafted` 用纯 CNN**（消融表明更准且部署更快）：

```bash
python scripts/train.py --epochs 30 --n-per-class 5000 --device cuda --no-handcrafted --out runs/synth_pre.pt
```

实拍微调（加载合成预训练权重，小学习率；开关须与预训练一致）：

```bash
python scripts/train.py --data-root data --init-from runs/synth_pre.pt --no-handcrafted --epochs 15 --lr 1e-4 --device cuda --out runs/finetune.pt
```

接入实拍数据（按视频场次划分 train/val 防泄漏）。二分类任务下**负样本无需细分**，支持两种目录布局：

```
# 极简二分类（推荐）：负样本全放 non_drone/ 即可
data/
  train/drone/*.png        # 正样本（飞机/无人机）
  train/non_drone/*.png    # 负样本（天空/云边缘、杂波等，混入即可）
  val/drone/*.png
  val/non_drone/*.png

# 兼容旧四类布局（drone 为正，其余三类自动并入负类）
data/
  train/{drone,sky_edge,cloud_edge,background}/*.png
  val/{drone,sky_edge,cloud_edge,background}/*.png
```

```bash
python scripts/train.py --data-root data --epochs 30
```

消融实验（去手工特征分支）：`python scripts/train.py --no-handcrafted`

## 模型验证与 ONNX 导出

训练产物正确性冒烟检查：

```bash
python scripts/validate_model.py --ckpt runs/finetune.pt
```

导出 ONNX 并校验与 PyTorch 输出一致（需 `pip install onnx onnxruntime`）：

```bash
python scripts/export_onnx.py --ckpt runs/finetune.pt --out runs/dronenet.onnx
python scripts/verify_onnx.py --ckpt runs/finetune.pt --onnx runs/dronenet.onnx --tol 1e-4
```

## 实测参考（合成数据, CPU, 10 epochs × 4800 样本）

| 方法 | F1 | AUC | AP | recall@FAR=1e-3 |
|---|---|---|---|---|
| 规则基线（传统算法） | — | 0.586 | — | — |
| CNN + 手工特征 | 0.772 | 0.928 | 0.892 | 0.597 |
| **纯 CNN（`--no-handcrafted`，推荐）** | **0.821** | **0.945** | **0.919** | **0.647** |

> **消融结论**：训练充分后，纯 CNN 全面优于"CNN + 手工特征"——手工特征是归纳先验，小数据有助收敛，但训练充分后学到的表示更强。纯 CNN 还让部署预处理快 28×、端到端快 1.9×，且整网无 OpenCV 依赖，可直接上 NPU。详见 `docs/训练与验证指南.md`。
> 合成数据仅用于管线验证与预训练，最终指标须以 100% 实拍的独立测试集为准。
