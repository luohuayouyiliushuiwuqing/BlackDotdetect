"""训练二分类 CNN (DroneNet) 并输出评估报告 (设计方案 5/6).

支持早停: 监控 val_loss / f1 / auc, 连续 --patience 轮无改善即停止, 并自动恢复最优权重。

示例:
    # 合成数据预训练 (无限在线合成, 先把特征学出来)
    python scripts/train.py --epochs 30 --n-per-class 5000 --device cuda --out runs/synth_pre.pt
    # 实拍数据微调 (加载合成预训练权重, 用小学习率)
    python scripts/train.py --data-root data --init-from runs/synth_pre.pt \
        --epochs 15 --lr 1e-4 --device cuda --out runs/finetune.pt
    # 按业务指标早停 (低虚警下召回优先时推荐监控 f1 或 auc)
    python scripts/train.py --data-root data --epochs 60 --monitor f1 --patience 10
    # 关闭早停, 跑满轮数
    python scripts/train.py --epochs 30 --patience 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blackdot.dataset import FolderROIDataset, SyntheticROIDataset
from blackdot.metrics import binary_metrics, drone_pr, report
from blackdot.model import DroneNet, BinaryFocalLoss, count_params


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="",
                   help="实拍 ROI 数据集根目录; 为空则使用在线合成数据")
    p.add_argument("--n-per-class", type=int, default=3000, help="合成数据每类样本数")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--init-from", type=str, default="",
                   help="预训练权重 .pt 路径, 用于合成预训练→实拍微调; 为空则随机初始化")
    p.add_argument("--no-handcrafted", action="store_true", help="消融: 去掉手工特征分支")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="runs/dronenet.pt")
    p.add_argument("--patience", type=int, default=8,
                   help="早停: 连续 N 轮监控指标无改善即停止; 0=关闭早停")
    p.add_argument("--min-delta", type=float, default=1e-4,
                   help="早停: 视为'有改善'的最小变化量")
    p.add_argument("--monitor", choices=["val_loss", "f1", "auc"], default="val_loss",
                   help="早停/择优监控指标: val_loss(越小越好) 或 f1/auc(越大越好)")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, str, dict]:
    """返回 (val_loss, 报告字符串, 指标字典) —— 指标字典供早停监控使用。"""
    model.eval()
    ys, scores, losses = [], [], []
    focal = BinaryFocalLoss(gamma=2.0).to(device)
    for x, feat, label in loader:
        x, feat, label = (x.to(device), feat.to(device), label.to(device))
        logit = model(x, feat)
        loss = focal(logit, label)
        losses.append(loss.item())
        ys.append(label.cpu().numpy().astype(int))
        scores.append(torch.sigmoid(logit).cpu().numpy())
    y = np.concatenate(ys)
    s = np.concatenate(scores)
    metrics = {**binary_metrics(y, s), **drone_pr(y, s)}
    return float(np.mean(losses)), report(y, s), metrics


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    if args.data_root:
        root = Path(args.data_root)
        train_ds = FolderROIDataset(root / "train", train=True, seed=args.seed)
        val_ds = FolderROIDataset(root / "val", train=False, seed=args.seed + 1)
    else:
        train_ds = SyntheticROIDataset(args.n_per_class, seed=args.seed, train=True)
        val_ds = SyntheticROIDataset(max(args.n_per_class // 4, 100),
                                     seed=args.seed + 999, train=False)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, drop_last=True, persistent_workers=True)
    val_ld = DataLoader(val_ds, batch_size=512, num_workers=2,
                        persistent_workers=True)

    model = DroneNet(use_handcrafted=not args.no_handcrafted).to(device)
    print(f"参数量: {count_params(model):,}  设备: {device}")

    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[微调] 加载预训练权重 {args.init_from} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
        if missing or unexpected:
            print("[警告] 权重不完全匹配, 请确认 --no-handcrafted 设置与预训练时一致")

    focal = BinaryFocalLoss(gamma=2.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # 早停: 统一转成"越大越好"的分数 (val_loss 取负)
    best = float("-inf")
    best_state: dict | None = None      # 最优权重快照, 早停后恢复
    bad_epochs = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stopped_epoch = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for x, feat, label in train_ld:
            x, feat, label = (x.to(device), feat.to(device), label.to(device))
            logit = model(x, feat)
            loss = focal(logit, label)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
        sch.step()
        val_loss, _, m = evaluate(model, val_ld, device)
        cur = -val_loss if args.monitor == "val_loss" else float(m[args.monitor])
        shown = val_loss if args.monitor == "val_loss" else cur
        print(f"[epoch {ep:02d}] train_loss={tot / len(train_ld):.4f} "
              f"val_loss={val_loss:.4f} {args.monitor}={shown:.4f} "
              f"({time.time() - t0:.1f}s)")

        if cur > best + args.min_delta:
            best = cur
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "monitor": args.monitor, "best_score": float(cur)},
                       out_path)
        else:
            bad_epochs += 1
            if args.patience > 0 and bad_epochs >= args.patience:
                stopped_epoch = ep
                print(f"[早停] {args.monitor} 连续 {bad_epochs} 轮无改善 "
                      f"(阈值 min_delta={args.min_delta}), 在第 {ep} 轮停止训练")
                break

    if best_state is not None:
        model.load_state_dict(best_state)   # 恢复最优权重, 避免停在过拟合的最后一轮
    _, rep, _ = evaluate(model, val_ld, device)
    tail = f" (早停于第 {stopped_epoch} 轮)" if stopped_epoch else ""
    print(f"\n最优模型已保存到 {out_path} [监控 {args.monitor}]{tail}\n\n{rep}")


if __name__ == "__main__":
    main()
