# BlackDotDetect — 远距离无人机黑色小目标 ROI 判别

判定 36×36 Y 分量 ROI 中心 20×20 区域内是否存在黑色元素，并判断其是否为无人机
（二分类：无人机 = 正类；天空边缘 / 云层边缘 / 纯背景 = 负类）。
设计细节见 `docs/黑色小目标检测模型设计方案.md`（注：文档原始版本为 4 分类 20×20 方案，
代码已演进为二分类 36×36，待同步）。

## 环境

```bash
conda activate yolov8   # /home/igs/miniconda3/envs/yolov8, Python 3.10 + torch 2.7
pip install onnx onnxruntime   # 仅导出/验证/基准脚本需要
```

## 目录结构

```
blackdot/
  features.py    # 暗像素判定规则基线 + 8 维手工结构特征(在中心 20x20 目标区统计)
  synthesis.py   # 四类样本合成管线 (统一退化链 + size-aware 模糊)
  dataset.py     # 在线合成 Dataset / 实拍 ROI 文件夹 Dataset + 增强
  model.py       # DroneNet 二分类 CNN (~7.6 万参数) + 二分类 Focal Loss
  metrics.py     # P/R/F1 / AUC / AP / 固定 FAR 下召回率
scripts/
  train.py           # 训练 + 验证评估 (支持 --init-from 微调)
  validate_model.py  # 权重加载/输出合法性/判别正确性冒烟校验
  export_onnx.py     # 导出 ONNX (自包含单文件, 支持纯 CNN 单输入签名)
  verify_onnx.py     # ONNX vs PyTorch 输出一致性对比
  benchmark_onnx.py  # 预处理/推理/端到端时延基准 (按 ROI 折算对比预算)
  baseline_eval.py   # 规则基线评估 (对照组)
```

## 标准流程

```bash
# 1) 合成数据预训练
python scripts/train.py --epochs 30 --n-per-class 5000 --out runs/synth_pre.pt

# 2) 实拍数据微调 (目录名含 drone/plane/target/uav/positive 视为正类)
#    默认开启早停 (--patience 8, 监控 val_loss), 可改 --monitor f1 按业务指标早停
python scripts/train.py --data-root data --init-from runs/synth_pre.pt --epochs 15 --lr 1e-4

# 3) 模型校验
python scripts/validate_model.py --ckpt runs/finetune.pt

# 4) 导出 + 一致性验证 + 时延基准
python scripts/export_onnx.py --ckpt runs/finetune.pt --out runs/dronenet.onnx
python scripts/verify_onnx.py --ckpt runs/finetune.pt --onnx runs/dronenet.onnx
python scripts/benchmark_onnx.py --onnx runs/dronenet.onnx
```

## 实拍数据格式

```
data/train/{drone,sky_edge,cloud_edge,background}/*.png    # 也支持 drone/ + non_drone/ 二目录
data/val/{...}/*.png
```

36×36 灰度 ROI（目标居中、外圈 8px 为背景）；不足 36 的部分自动填 0 并用掩码通道标记。
**同一视频序列/采集场次只能进一个集合**，不得跨 train/val 拆分相邻帧。

## 冒烟验证结果（合成数据，1500 样本/类，15 epochs）

| 方法 | F1 | AUC | FAR/ROI | 端到端时延 |
|---|---|---|---|---|
| 规则基线 | 0.019 | 0.708 | 0.020 | — |
| DroneNet | 0.876 | 0.958 | 0.0089 | 0.252 ms/ROI |

ONNX 一致性：max_abs_diff ≈ 2.2e-06（容忍 1e-4），0.5 阈值决策 100% 一致。

> 合成数据仅用于管线验证与预训练，最终指标须以 100% 实拍的独立测试集为准。
