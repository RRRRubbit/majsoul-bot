"""
快速批量标注工具
使用方式：
  python tools/label_tiles.py                    # 从 templates/tiles/raw/
  python tools/label_tiles.py --src <目录>

在一个窗口中同时显示全部已提取的牌，
然后在终端输入逗号分隔的牌名，一次性完成命名。

牌名格式：
  万子 1m~9m   筒子 1p~9p   索子 1s~9s
  字牌 1z东 2z南 3z西 4z北 5z白 6z发 7z中
  赤宝 0m/0p/0s    跳过某张 用 - 占位
"""

import argparse
import os
import sys
import shutil
from pathlib import Path

import cv2
import numpy as np

_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

VALID_NAMES = set(
    [f"{i}m" for i in range(0, 10)]
    + [f"{i}p" for i in range(0, 10)]
    + [f"{i}s" for i in range(0, 10)]
    + [f"{i}z" for i in range(1, 8)]
)

TILE_HINT = {
    "1z": "东", "2z": "南", "3z": "西", "4z": "北",
    "5z": "白", "6z": "发", "7z": "中",
}


def build_grid(tiles_with_path, cell_w=100, cell_h=140):
    """将牌图拼成一行预览图，加序号标签"""
    cells = []
    for idx, (img, _) in enumerate(tiles_with_path):
        cell = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_LANCZOS4)
        # 序号标签
        tag = "摸" if idx == 13 else str(idx)
        cv2.putText(cell, tag, (3, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 220, 255) if idx == 13 else (80, 255, 80), 1, cv2.LINE_AA)
        cells.append(cell)

    row = np.hstack(cells)
    return row


def load_raw_tiles(src_dir: Path):
    """按 tile_0 ~ tile_N 顺序加载图像"""
    tiles = []
    i = 0
    while True:
        p = src_dir / f"tile_{i}.png"
        if not p.exists():
            break
        img = cv2.imread(str(p))
        if img is None:
            break
        tiles.append((img, p))
        i += 1
    return tiles


def run(src_dir: str, out_dir: str):
    src = Path(src_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tiles = load_raw_tiles(src)
    if not tiles:
        print(f"[ERROR] 未找到 tile_0.png 等文件在: {src}")
        sys.exit(1)

    n = len(tiles)
    print(f"\n已加载 {n} 张牌图像（来自 {src.resolve()}）")

    # ── 构建预览图 ───────────────────────────────────────
    grid = build_grid(tiles)
    cv2.namedWindow("Tile Labeler", cv2.WINDOW_NORMAL)
    win_w = min(1400, grid.shape[1])
    win_h = int(win_w / grid.shape[1] * grid.shape[0]) + 10
    cv2.resizeWindow("Tile Labeler", win_w, win_h)
    cv2.imshow("Tile Labeler", grid)
    cv2.waitKey(1)

    # ── 打印说明 ─────────────────────────────────────────
    print("=" * 60)
    print("窗口中从左到右依次是牌 0, 1, 2, ... , 摸牌(最右)")
    print()
    print("请在下方输入逗号分隔的牌名，顺序与窗口一致")
    print("示例: 1m,1m,2m,3m,9m,2p,5p,2z,3z,4z,6z,1s,5s,5m")
    print("跳过某张(不保存)用 - 占位，如: 1m,-,2m,...")
    print()
    print("牌名参考:")
    print("  万字: 1m 2m 3m 4m 5m 6m 7m 8m 9m  (赤五万=0m)")
    print("  筒子: 1p 2p 3p 4p 5p 6p 7p 8p 9p  (赤五筒=0p)")
    print("  索子: 1s 2s 3s 4s 5s 6s 7s 8s 9s  (赤五索=0s)")
    print("  字牌: 1z东 2z南 3z西 4z北 5z白 6z发 7z中")
    print("=" * 60)

    while True:
        raw = input(f"\n输入 {n} 个牌名（逗号分隔）: ").strip()
        if not raw:
            continue

        parts = [p.strip().lower() for p in raw.split(",")]
        if len(parts) != n:
            print(f"  ✗ 数量不对：输入 {len(parts)} 个，应为 {n} 个")
            continue

        # 验证
        errors = []
        for i, name in enumerate(parts):
            if name != "-" and name not in VALID_NAMES:
                tag = "摸牌" if i == 13 else f"第{i}张"
                errors.append(f"  [{tag}] 无效牌名「{name}」")
        if errors:
            for e in errors:
                print(e)
            print("  ✗ 请重新输入")
            continue

        # 确认
        preview_parts = []
        for i, name in enumerate(parts):
            if name == "-":
                preview_parts.append("(跳过)")
            else:
                hint = TILE_HINT.get(name, "")
                preview_parts.append(f"{name}{hint}")
        print("\n  标注结果预览:")
        for i, p in enumerate(preview_parts):
            tag = "摸牌" if i == 13 else f" {i:2d} "
            print(f"    [{tag}] tile_{i}.png → {p}")

        confirm = input("\n  确认保存？(y=确认 / n=重新输入 / q=退出): ").strip().lower()
        if confirm == "q":
            break
        if confirm != "y":
            continue

        # 保存
        saved = 0
        skipped = 0
        overwritten = 0
        for i, (name, (img, src_path)) in enumerate(zip(parts, tiles)):
            if name == "-":
                skipped += 1
                continue
            dst = out / f"{name}.png"
            existed = dst.exists()
            cv2.imwrite(str(dst), img)
            mark = "覆盖" if existed else "新建"
            tag = "摸牌" if i == 13 else f"第{i}张"
            print(f"  [{tag}] → {dst.name}  ({mark})")
            saved += 1
            if existed:
                overwritten += 1

        print(f"\n✅ 完成！保存 {saved} 张 | 跳过 {skipped} 张 | 覆盖 {overwritten} 张")
        print(f"   模板目录: {out.resolve()}")
        break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量标注已提取的手牌图像")
    parser.add_argument("--src", default="templates/tiles",
                        help="原始牌图目录（默认: templates/tiles/raw）")
    parser.add_argument("--out", default="templates/tiles",
                        help="输出模板目录（默认: templates/tiles）")
    args = parser.parse_args()
    run(args.src, args.out)
