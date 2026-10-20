# BlackDotDetect — 远距离无人机黑色小目标 ROI 判别

对 <20×20 像素 Y 分量 ROI 判定是否存在黑色元素，并区分 **无人机 / 天空边缘 / 云层边缘 / 背景** 四类。设计细节见 `docs/黑色小目标检测模型设计方案.md`。

## 环境

```bash
conda activate yolov8   # /home/igs/miniconda3/envs/yolov8, Python 3.10 + torch 2.7
```

## 目录结构

```
blackdot/
  features.py    # 暗像素判定规则基线 + 8 维手工结构特征
  synthesis.py   # 四类样本合成管线 (统一退化链: 模糊/噪声/亮度漂移)
  dataset.py     # 在线合成 Dataset / 实拍 ROI 文件夹 Dataset + 增强
  model.py       # 双头轻量 CNN (~7.7 万参数) + Focal Loss
  metrics.py     # 混淆矩阵 / PR / 固定 FAR 下召回率
scripts/
  baseline_eval.py  # 规则基线评估 (对照组)
  train.py          # 训练 + 验证评估
```

## 快速开始

规则基线（对照组，验证"传统算法无法区分"）：

```bash
python scripts/baseline_eval.py --n-per-class 500
```

训练 CNN（无实拍数据时使用在线合成数据）：

```bash
python scripts/train.py --epochs 30 --n-per-class 5000 --device cuda
```

接入实拍数据（按视频场次划分 train/val 防泄漏）：

```
data/
  train/{drone,sky_edge,cloud_edge,background}/*.png
  val/{drone,sky_edge,cloud_edge,background}/*.png
```

```bash
python scripts/train.py --data-root data --epochs 30
```

消融实验（去手工特征分支）：`python scripts/train.py --no-handcrafted`

## 冒烟验证结果（合成数据, 800 样本/类, 5 epochs）

| 方法 | 无人机 F1 | 无人机 AUC | Macro-F1 |
|---|---|---|---|
| 规则基线 | 0.258 | 0.570 | 0.205 |
| 双头 CNN | 0.822 | 0.960 | 0.659 |

> 合成数据仅用于管线验证与预训练，最终指标须以 100% 实拍的独立测试集为准（设计方案 5.5 / 6.2）。
