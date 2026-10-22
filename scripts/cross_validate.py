"""留一段视频交叉验证 (leave-one-clip-out): 跨场景泛化检验.

同一段视频内的 train/val 划分无法回答"模型能否泛化到未见过的场景"——
同场景的背景、光照、云形会被模型记住, 指标虚高 (实测同场景 AUC 1.000,
换到未见视频段后 F1 掉到 0.534)。本脚本把所有已标注样本按 clip 分组,
每次留一整段做验证集, 其余做训练集, 轮换后给出各段指标与均值。

用法:
    python scripts/cross_validate.py                      # 默认 30 epochs, batch 32
    python scripts/cross_validate.py --epochs 50 --batch-size 16 --lr 1e-4
    python scripts/cross_validate.py --data-root data --min-clip-samples 50

输出: runs/cv/val_<clip>/model.pt 及各段评估报告。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLASSES = ["drone", "non_drone"]
EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def clip_of(name: str) -> str:
    """文件名前 3 段为视频段 ID: ff90_1013_021_000525_001049.png -> ff90_1013_021"""
    return "_".join(name.split("_")[:3])


def collect(data_root: Path) -> dict:
    """从 <data_root>/{train,val,test}/<class> 汇集所有样本 -> {clip: {class: [files]}}"""
    pool: dict[str, dict[str, list[Path]]] = {}
    for split in ("train", "val", "test"):
        for cls in CLASSES:
            d = data_root / split / cls
            if not d.exists():
                continue
            for f in d.iterdir():
                if f.is_file() and f.suffix.lower() in EXTS:
                    pool.setdefault(clip_of(f.name), {}).setdefault(cls, []).append(f)
    if not pool:
        raise FileNotFoundError(f"{data_root} 下未找到样本, 请先运行 scripts/split_data.py")
    return pool


def build(exp_dir: Path, pool: dict, val_clip: str) -> tuple[int, int]:
    """用软链接组装一次实验的 train/val 目录 (不复制数据)."""
    if exp_dir.exists():
        shutil.rmtree(exp_dir)
    for split in ("train", "val"):
        for cls in CLASSES:
            (exp_dir / split / cls).mkdir(parents=True, exist_ok=True)
    for clip, classes in pool.items():
        split = "val" if clip == val_clip else "train"
        for cls, files in classes.items():
            for f in files:
                link = exp_dir / split / cls / f.name
                if not link.exists():
                    link.symlink_to(f.resolve())
    n_tr = sum(len(list((exp_dir / "train" / c).iterdir())) for c in CLASSES)
    n_va = sum(len(list((exp_dir / "val" / c).iterdir())) for c in CLASSES)
    return n_tr, n_va


def run(exp_dir: Path, epochs: int, batch: int, lr: float, monitor: str,
        patience: int) -> str:
    cmd = [sys.executable, str(ROOT / "scripts" / "train.py"),
           "--data-root", str(exp_dir), "--epochs", str(epochs),
           "--batch-size", str(batch), "--lr", str(lr),
           "--monitor", monitor, "--patience", str(patience),
           "--out", str(exp_dir / "model.pt")]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if res.returncode != 0:
        return f"[失败] {res.stderr[-1500:]}"
    return "\n".join(res.stdout.strip().splitlines()[-11:])


def main() -> int:
    ap = argparse.ArgumentParser(description="留一段视频交叉验证")
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--monitor", default="f1", choices=["f1", "auc", "val_loss"])
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min-clip-samples", type=int, default=20,
                    help="样本数少于该值的视频段不单独做验证集")
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "cv"))
    args = ap.parse_args()

    pool = collect(Path(args.data_root))
    counts = {c: sum(len(v) for v in d.values()) for c, d in pool.items()}
    print("各视频段样本数:", dict(sorted(counts.items(), key=lambda x: -x[1])))
    clips = [c for c, n in sorted(counts.items(), key=lambda x: -x[1])
             if n >= args.min_clip_samples]
    if len(clips) < 2:
        print(f"[警告] 只有 {len(clips)} 段视频达到 --min-clip-samples {args.min_clip_samples}, "
              f"无法做留一段验证; 请补充不同场地/时段/天气的视频")
        return 1

    out_root = Path(args.out_dir)
    f1s, aucs = [], []
    for val_clip in clips:
        exp = out_root / f"val_{val_clip}"
        n_tr, n_va = build(exp, pool, val_clip)
        print(f"\n{'=' * 62}\n留一段验证: val={val_clip} "
              f"(train {n_tr} 张 / val {n_va} 张)\n{'=' * 62}")
        out = run(exp, args.epochs, args.batch_size, args.lr, args.monitor,
                  args.patience)
        print(out)
        for line in out.splitlines():
            if line.strip().startswith("f1:"):
                f1s.append(float(line.split(":")[1]))
            if line.strip().startswith("auc:"):
                aucs.append(float(line.split(":")[1]))

    print(f"\n{'=' * 62}\n留一段交叉验证汇总 ({len(clips)} 段)")
    for c, f in zip(clips, f1s):
        print(f"  val={c}: F1={f:.4f}")
    if f1s:
        print(f"  平均 F1 = {sum(f1s) / len(f1s):.4f}   "
              f"平均 AUC = {sum(aucs) / len(aucs):.4f}")
        print("  注: 各段差异越大, 说明越依赖场景记忆, 泛化越差; "
              "应补充更多场景数据直至各段指标接近。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
