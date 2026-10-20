"""训练双头 CNN 并输出评估报告 (设计方案 5/6).

示例:
    python scripts/train.py --epochs 10 --n-per-class 3000 --device cuda
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
from blackdot.model import DualHeadNet, FocalLoss, count_params


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="",
                   help="实拍 ROI 数据集根目录; 为空则使用在线合成数据")
    p.add_argument("--n-per-class", type=int, default=3000, help="合成数据每类样本数")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lambda-cls", type=float, default=2.0, help="分类损失权重 λ2")
    p.add_argument("--no-handcrafted", action="store_true", help="消融: 去掉手工特征分支")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="runs/dualhead.pt")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, str]:
    model.eval()
    ys, bs, preds, dscores, losses = [], [], [], [], []
    focal = FocalLoss(gamma=2.0).to(device)
    for x, feat, cls, black in loader:
        x, feat, cls, black = (x.to(device), feat.to(device),
                               cls.to(device), black.to(device))
        lb, lc = model(x, feat)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lb, black) \
            + 2.0 * focal(lc, cls)
        losses.append(loss.item())
        ys.append(cls.cpu().numpy())
        bs.append(torch.sigmoid(lb).cpu().numpy())
        preds.append(lc.argmax(1).cpu().numpy())
        dscores.append(torch.softmax(lc, 1)[:, 0].cpu().numpy())
    y = np.concatenate(ys)
    return float(np.mean(losses)), report(y, np.concatenate(bs),
                                          np.concatenate(preds),
                                          np.concatenate(dscores))


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

    model = DualHeadNet(use_handcrafted=not args.no_handcrafted).to(device)
    print(f"参数量: {count_params(model):,}  设备: {device}")

    focal = FocalLoss(gamma=2.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = float("-inf")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for x, feat, cls, black in train_ld:
            x, feat, cls, black = (x.to(device), feat.to(device),
                                   cls.to(device), black.to(device))
            lb, lc = model(x, feat)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(lb, black) \
                + args.lambda_cls * focal(lc, cls)
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
