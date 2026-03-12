"""
从游戏截图中提取手牌模板
使用方式：
  python tools/extract_tiles_from_screenshot.py  <截图路径>

功能：
  1. 读取一张游戏截图
  2. 按手牌区域坐标裁取每张牌的图像
  3. 在窗口中逐一展示，让用户输入牌名
  4. 保存到 templates/tiles/<牌名>.png

牌名格式（对应 ALL_TILE_NAMES）：
  万子: 1m~9m    筒子: 1p~9p    索子: 1s~9s
  字牌: 1z东 2z南 3z西 4z北 5z白 6z发 7z中
  赤宝牌: 0m / 0p / 0s
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# 确保项目根目录在 sys.path
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

from majsoul_bot.vision.regions import DEFAULT_REGIONS, ScreenRegions


# ── 合法牌名集合 ────────────────────────────────────────────────
VALID_NAMES = set(
    [f"{i}m" for i in range(0, 10)]   # 0m 是赤宝五万
    + [f"{i}p" for i in range(0, 10)]
    + [f"{i}s" for i in range(0, 10)]
    + [f"{i}z" for i in range(1, 8)]
)


def extract_and_label(screenshot_path: str, out_dir: str = "templates/tiles"):
    """从截图中提取手牌并交互式标注"""
    img = cv2.imread(screenshot_path)
    if img is None:
        print(f"[ERROR] 无法读取图片: {screenshot_path}")
        sys.exit(1)

    h, w = img.shape[:2]
    print(f"[INFO] 截图尺寸: {w}×{h}")

    regions = ScreenRegions.load_from_json()
    os.makedirs(out_dir, exist_ok=True)

    # 检测 14 张牌（13 手牌 + 1 摸牌）
    total = 14
    tiles: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []

    for i in range(total):
        is_drawn = i >= 13
        x_rel, y_rel, tw_rel, th_rel = regions.get_tile_rect(i, 13, is_drawn)

        x1 = int(x_rel * w)
        y1 = int(y_rel * h)
        x2 = min(int((x_rel + tw_rel) * w), w)
        y2 = min(int((y_rel + th_rel) * h), h)

        if x1 >= w or y1 >= h or x2 <= x1 or y2 <= y1:
            print(f"  牌{i}: 超出图像范围，跳过")
            continue

        tile_img = img[y1:y2, x1:x2].copy()
        tiles.append((tile_img, (x1, y1, x2, y2)))

    if not tiles:
        print("[ERROR] 未能提取到任何牌，请检查坐标设置")
        sys.exit(1)

    print(f"\n[INFO] 共提取 {len(tiles)} 张牌图像")
    print("=" * 50)
    print("操作说明：")
    print("  - 在图像窗口展示当前牌")
    print("  - 在终端输入牌名（如 1m、5p、7z），回车确认")
    print("  - 输入 's' 跳过这张牌")
    print("  - 输入 'q' 退出")
    print("=" * 50)

    saved = 0
    skipped = 0
    cv2.namedWindow("Tile Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tile Preview", 200, 280)

    # 准备全景预览图
    overview = _make_overview(img, tiles)

    for i, (tile_img, (x1, y1, x2, y2)) in enumerate(tiles):
        tag = "摸牌" if i >= 13 else f"第{i+1}张"

        # 放大显示
        display = cv2.resize(tile_img, (180, 250), interpolation=cv2.INTER_NEAREST)
        # 在预览图上标注当前位置
        ov = overview.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(ov, tag, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 拼合展示（放大牌 + 全景）
        ov_resized = cv2.resize(ov, (600, int(600 * h / w)))
        h2 = ov_resized.shape[0]
        pad = np.zeros((h2, 200, 3), dtype=np.uint8)
        tile_placed = cv2.resize(display, (200, min(h2, 280)))
        pad[:tile_placed.shape[0], :] = tile_placed
        combined = np.hstack([ov_resized, pad])

        cv2.imshow("Tile Preview", combined)
        cv2.waitKey(1)

        # 用户输入
        while True:
            name = input(f"  [{tag}] 牌名（或 s=跳过，q=退出）: ").strip().lower()
            if name == "q":
                cv2.destroyAllWindows()
                print(f"\n已保存 {saved} 张，跳过 {skipped} 张，退出。")
                return
            if name == "s":
                skipped += 1
                print(f"    ↳ 跳过")
                break
            if name in VALID_NAMES:
                save_path = Path(out_dir) / f"{name}.png"
                if save_path.exists():
                    ow = input(f"    ⚠ {save_path} 已存在，覆盖？(y/n): ").strip().lower()
                    if ow != "y":
                        print("    ↳ 已取消，跳过")
                        skipped += 1
                        break
                cv2.imwrite(str(save_path), tile_img)
                print(f"    ↳ 已保存: {save_path}")
                saved += 1
                break
            else:
                print(f"    ✗ 无效牌名「{name}」，合法格式: 1m~9m / 1p~9p / 1s~9s / 1z~7z / 0m/0p/0s")

    cv2.destroyAllWindows()
    print(f"\n完成！已保存 {saved} 张，跳过 {skipped} 张")
    print(f"模板目录: {Path(out_dir).resolve()}")


def _make_overview(img: np.ndarray, tiles: list) -> np.ndarray:
    """在截图上绘制所有牌位边框，作为全景参考"""
    ov = img.copy()
    for i, (_, (x1, y1, x2, y2)) in enumerate(tiles):
        color = (0, 200, 255) if i >= 13 else (0, 255, 80)
        cv2.rectangle(ov, (x1, y1), (x2, y2), color, 1)
        cv2.putText(ov, str(i), (x1 + 1, y1 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return ov


def batch_extract_no_label(screenshot_path: str, out_dir: str = "templates/tiles/raw"):
    """
    非交互模式：直接将所有牌提取为 tile_0.png ~ tile_13.png
    用户之后可以手动重命名
    """
    img = cv2.imread(screenshot_path)
    if img is None:
        print(f"[ERROR] 无法读取图片: {screenshot_path}")
        sys.exit(1)

    h, w = img.shape[:2]
    regions = ScreenRegions.load_from_json()
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    for i in range(14):
        is_drawn = i >= 13
        x_rel, y_rel, tw_rel, th_rel = regions.get_tile_rect(i, 13, is_drawn)
        x1 = int(x_rel * w)
        y1 = int(y_rel * h)
        x2 = min(int((x_rel + tw_rel) * w), w)
        y2 = min(int((y_rel + th_rel) * h), h)

        if x1 >= w or y1 >= h or x2 <= x1 or y2 <= y1:
            continue

        tile_img = img[y1:y2, x1:x2]
        out_path = Path(out_dir) / f"tile_{i}.png"
        cv2.imwrite(str(out_path), tile_img)
        print(f"  tile_{i}.png  → ({x1},{y1})~({x2},{y2})")
        saved += 1

    print(f"\n已提取 {saved} 张牌图像到 {Path(out_dir).resolve()}")
    print("请将文件重命名为对应牌名（如 1m.png），然后移动到 templates/tiles/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从雀魂截图提取手牌模板")
    parser.add_argument("screenshot", nargs="?", default="logs/debug_latest.png",
                        help="截图路径（默认: logs/debug_latest.png）")
    parser.add_argument("--out", default="templates/tiles",
                        help="输出目录（默认: templates/tiles）")
    parser.add_argument("--batch", action="store_true",
                        help="批量模式：不交互，直接保存为 tile_N.png")
    args = parser.parse_args()

    print(f"截图: {args.screenshot}")
    if args.batch:
        batch_extract_no_label(args.screenshot, out_dir=args.out + "/raw")
    else:
        extract_and_label(args.screenshot, out_dir=args.out)
