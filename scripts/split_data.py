"""按"帧"分组划分数据集, 防止同一帧内容跨 train/val 泄漏.

背景: 实拍文件名形如 ff90_1013_021_000525_001049.png
      └─ 序列ID ──┘  └帧号┘  └裁剪序号┘
同一帧号下的多个裁剪块内容高度相似; 逐文件随机划分会让"同帧的不同块"分别进入
train 与 val, 造成指标虚高。故默认以【帧】为最小分组单位。

用法:
    # 划分到 data/train, data/val, data/test (7:1.5:1.5)
    python scripts/split_data.py --src data/base_data --dst data

    # 只要 train/val (8:2), 用软链接节省空间
    python scripts/split_data.py --src data/base_data --dst data \
        --train 0.8 --val 0.2 --test 0 --mode symlink

    # 相近帧也视为同组 (每 5 帧一组), 进一步降低时序泄漏
    python scripts/split_data.py --src data/base_data --dst data --block 5

    # 只看计划不落盘
    python scripts/split_data.py --src data/base_data --dst data --dry-run
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

# 文件名: <clip>_<frame>_<idx>.png, 取前缀若干段作为分组键
NAME_RE = re.compile(r"^(?P<clip>[^_]+_[^_]+_[^_]+)_(?P<frame>\d+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="按帧分组划分 ROI 数据集")
    p.add_argument("--src", required=True, help="源数据根目录: <src>/<class>/*.png")
    p.add_argument("--dst", required=True, help="输出根目录, 生成 <dst>/{train,val,test}/<class>/")
    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--group-by", choices=["frame", "clip", "file"], default="frame",
                   help="frame=同一帧同组(默认, 推荐); clip=整段视频同组; file=逐文件(不推荐)")
    p.add_argument("--block", type=int, default=1,
                   help="每 N 个连续帧号合并为一组 (>=2 可降低相邻帧的时序泄漏)")
    p.add_argument("--mode", choices=["copy", "symlink", "move"], default="copy")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="只打印划分计划, 不写文件")
    p.add_argument("--clean", action="store_true",
                   help="写入前清空 <dst>/{train,val,test}/<class> 下已有文件")
    return p.parse_args()


def group_key(path: Path, mode: str, block: int) -> str:
    """按模式生成分组键: 同一键的文件必须落在同一个 split."""
    stem = path.stem
    if mode == "file":
        return f"file::{stem}"
    m = NAME_RE.match(stem)
    if not m:
        # 命名不规范: 退化为按文件名去掉末尾数字序号, 尽量把同帧归到一起
        base = re.sub(r"_\d+$", "", stem)
        return f"unk::{base}"
    clip, frame = m.group("clip"), int(m.group("frame"))
    if mode == "clip":
        return f"clip::{clip}"
    return f"frame::{clip}::{frame // max(block, 1)}"


def scan(src: Path) -> dict[str, list[Path]]:
    """返回 {类名: [文件路径]}."""
    classes: dict[str, list[Path]] = {}
    for cls_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        files = sorted(f for f in cls_dir.rglob("*")
                       if f.is_file() and f.suffix.lower() in EXTS)
        if files:
            classes[cls_dir.name] = files
    if not classes:
        raise FileNotFoundError(f"{src} 下未找到任何 {EXTS} 图像")
    return classes


def _imbalance(counts, targets, classes, splits) -> float:
    """各类各 split 的归一化偏差之和 (越小越均衡)."""
    return sum(abs(counts[c][s] - targets[c][s]) / max(len(classes[c]), 1)
               for c in classes for s in splits)


def assign_groups(classes: dict[str, list[Path]], ratios: dict[str, float],
                  mode: str, block: int, seed: int) -> tuple[dict[str, str], dict]:
    """把分组键分配到各 split.

    规则:
      1) 一个分组键只能属于一个 split (保证同帧不跨集, 同一帧的正负样本同集);
      2) 贪心: 按"组整体"评分选 split —— 兼顾组内所有类别的缺口,
         避免只优化一个类导致其他类比例失衡;
      3) 精调: 反复尝试把单个组换到别的 split, 只要总偏差下降就接受。
    """
    # groups: key -> {class: [files]}
    groups: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for cls, files in classes.items():
        for f in files:
            groups[group_key(f, mode, block)][cls].append(f)

    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    # 大组先放: 配额宽松时优先安置, 降低末尾无法填满造成的失衡
    keys.sort(key=lambda k: -sum(len(v) for v in groups[k].values()))

    splits = [s for s, r in ratios.items() if r > 0]
    totals = {c: len(f) for c, f in classes.items()}
    targets = {c: {s: ratios[s] * totals[c] for s in splits} for c in classes}
    counts = {c: {s: 0 for s in splits} for c in classes}
    assigned: dict[str, str] = {}

    for k in keys:
        best, best_score = splits[0], -1e18
        for s in splits:
            score = 0.0
            for c, fs in groups[k].items():
                deficit = (targets[c][s] - counts[c][s]) / max(totals[c], 1)
                score += (len(fs) / max(totals[c], 1)) * deficit
            if score > best_score:
                best_score, best = score, s
        assigned[k] = best
        for c, fs in groups[k].items():
            counts[c][best] += len(fs)

    def move(k: str, frm: str, to: str) -> None:
        for c, fs in groups[k].items():
            counts[c][frm] -= len(fs)
            counts[c][to] += len(fs)

    # ---- 精调 1: 单组换 split, 偏差下降即接受 ----
    for _ in range(50):
        improved = False
        cur = _imbalance(counts, targets, classes, splits)
        for k in keys:
            src_split = assigned[k]
            for s in splits:
                if s == src_split:
                    continue
                move(k, src_split, s)
                new = _imbalance(counts, targets, classes, splits)
                if new < cur - 1e-12:
                    assigned[k], cur, improved = s, new, True
                    src_split = s
                else:
                    move(k, s, src_split)
        if not improved:
            break

    # ---- 精调 2: 两组互换 (单组移动已饱和时, 靠大小组互换继续逼近目标) ----
    for _ in range(30):
        improved = False
        cur = _imbalance(counts, targets, classes, splits)
        for i, k1 in enumerate(keys):
            for k2 in keys[i + 1:]:
                s1, s2 = assigned[k1], assigned[k2]
                if s1 == s2:
                    continue
                move(k1, s1, s2)
                move(k2, s2, s1)
                new = _imbalance(counts, targets, classes, splits)
                if new < cur - 1e-12:
                    assigned[k1], assigned[k2], cur, improved = s2, s1, new, True
                else:
                    move(k1, s2, s1)
                    move(k2, s1, s2)
        if not improved:
            break

    return assigned, {"groups": groups, "counts": counts, "targets": targets,
                      "splits": splits}


def main() -> int:
    args = parse_args()
    src, dst = Path(args.src), Path(args.dst)
    ratios = {"train": args.train, "val": args.val, "test": args.test}
    total_r = sum(ratios.values())
    if abs(total_r - 1.0) > 1e-6:
        raise ValueError(f"train+val+test 必须等于 1.0, 当前 {total_r:.3f}")

    classes = scan(src)
    assigned, info = assign_groups(classes, ratios, args.group_by, args.block, args.seed)
    splits = info["splits"]
    groups, counts = info["groups"], info["counts"]

    # ---- 计划报告 ----
    print(f"源数据: {src}")
    for cls, files in classes.items():
        print(f"  {cls}: {len(files)} 张")
    print(f"分组方式: {args.group_by} (block={args.block}) -> 共 {len(groups)} 组")
    print(f"划分比例: " + ", ".join(f"{s}={ratios[s]:.2f}" for s in splits))
    print()
    print(f"{'类':<12}{'split':<8}{'数量':>6}{'占比':>9}")
    for cls in classes:
        n = len(classes[cls])
        for s in splits:
            c = counts[cls][s]
            print(f"{cls:<12}{s:<8}{c:>6}{c / n * 100:>8.1f}%")

    # ---- 泄漏自检: 同一分组键不得出现在多个 split ----
    seen: dict[str, set] = defaultdict(set)
    for k, s in assigned.items():
        seen[k].add(s)
    bad = [k for k, v in seen.items() if len(v) > 1]
    if bad:
        print(f"\n[错误] 有 {len(bad)} 个分组被分到多个 split, 划分逻辑异常")
        return 1
    print("\n[自检] 无分组跨 split 泄漏: PASS")

    if args.dry_run:
        print("\n[dry-run] 未写入任何文件")
        return 0

    # ---- 落盘 ----
    if args.clean:
        for s in splits:
            for cls in classes:
                d = dst / s / cls
                if d.exists():
                    for f in d.iterdir():
                        if f.is_file() or f.is_symlink():
                            f.unlink()
    written = 0
    for cls, files in classes.items():
        for f in files:
            s = assigned[group_key(f, args.group_by, args.block)]
            out_dir = dst / s / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f.name
            if out.exists() or out.is_symlink():
                out = out_dir / f"{f.stem}__dup{f.suffix}"
            if args.mode == "copy":
                shutil.copy2(f, out)
            elif args.mode == "symlink":
                out.symlink_to(f.resolve())
            else:
                shutil.move(str(f), out)
            written += 1
    print(f"\n已写入 {written} 个文件到 {dst} (模式: {args.mode})")
    for s in splits:
        tot = sum(counts[cls][s] for cls in classes)
        print(f"  {s}: {tot} 张")
    return 0


if __name__ == "__main__":
    sys.exit(main())
