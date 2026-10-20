"""训练二分类 CNN (DroneNet) 并输出评估报告 (设计方案 5/6).

示例:
    # 合成数据预训练 (无限在线合成, 先把特征学出来)
    python scripts/train.py --epochs 30 --n-per-class 5000 --device cuda --out runs/synth_pre.pt
    # 实拍数据微调 (加载合成预训练权重, 用小学习率)
    python scripts/train.py --data-root data --init-from runs/synth_pre.pt \
        --epochs 15 --lr 1e-4 --device cuda --out runs/finetune.pt
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
from blackdot.metrics import report
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
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, str]:
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
    return float(np.mean(losses)), report(y, np.concatenate(scores))


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

    best = float("-inf")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
        val_loss, _ = evaluate(model, val_ld, device)
        print(f"[epoch {ep:02d}] train_loss={tot / len(train_ld):.4f} "
              f"val_loss={val_loss:.4f} ({time.time() - t0:.1f}s)")
        score = -val_loss
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "args": vars(args)}, out_path)

    model.load_state_dict(torch.load(out_path, weights_only=False)["model"])
    val_loss, rep = evaluate(model, val_ld, device)
    print(f"\n最优模型已保存到 {out_path}\n\n{rep}")


if __name__ == "__main__":
    main()
