"""
模板捕获工具
从游戏截图中截取麻将牌和按钮模板，保存为图片文件供识别使用

使用说明：
  1. 启动雀魂游戏并进入对局（确保手牌可见）
  2. 运行此脚本：python tools/capture_templates.py
  3. 按照提示操作

可选参数：
  --mode tiles    仅捕获牌型模板（默认）
  --mode buttons  仅捕获按钮模板
  --mode all      同时捕获牌型和按钮模板
  --output DIR    指定模板输出根目录（默认：templates）
"""

import argparse
import os
import sys
from pathlib import Path

import cv2

# 添加项目根目录到 sys.path
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

from majsoul_bot.vision.screen_capture import ScreenCapture
from majsoul_bot.vision.regions import DEFAULT_REGIONS, ScreenRegions
from majsoul_bot.vision.tile_recognizer import ALL_TILE_NAMES


# ──────────────────────────────────────────────
# 牌型模板捕获
# ──────────────────────────────────────────────

def capture_tile_templates(output_dir: str = "templates/tiles"):
    """
    交互式捕获手牌模板

    流程：
    1. 截取当前游戏画面
    2. 切割出每个手牌位置的图像
    3. 提示用户输入每张牌的名称并保存
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sc = ScreenCapture()
    sc.find_game_window()
    regions = DEFAULT_REGIONS

    print("\n" + "=" * 52)
    print("  雀魂牌型模板捕获工具")
    print("=" * 52)
    print("说明：")
    print("  每次按 Enter 截图，然后输入每张牌的名称")
    print("  牌名格式：1m~9m（万）/ 1p~9p（筒）/ 1s~9s（索）/ 1z~7z（字）")
    print("  输入 skip 跳过某张牌，输入 quit 退出")
    print("  已保存的模板会被跳过（除非重新输入）")
    print()

    already_saved = {p.stem for p in out.glob("*.png")}
    if already_saved:
        print(f"  已有 {len(already_saved)} 个模板，重复的将被覆盖")
    print()

    while True:
        cmd = input("按 Enter 截图（或输入 quit 退出）> ").strip().lower()
        if cmd == "quit":
            break

        screenshot = sc.capture()
        img_h, img_w = screenshot.shape[:2]
        reg = regions.hand

        # 保存预览图（标注手牌区域）
        preview = screenshot.copy()
        for i in range(reg.max_tiles + 1):
            is_drawn = i >= reg.max_tiles
            x_rel, y_rel, tw_rel, th_rel = regions.get_tile_rect(i, reg.max_tiles, is_drawn)
            x1 = int(x_rel * img_w)
            y1 = int(y_rel * img_h)
            x2 = x1 + int(tw_rel * img_w)
            y2 = y1 + int(th_rel * img_h)
            color = (0, 200, 255) if is_drawn else (0, 255, 100)
            cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
            cv2.putText(preview, str(i), (x1 + 2, y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        os.makedirs("logs", exist_ok=True)
        preview_path = "logs/capture_preview.png"
        cv2.imwrite(preview_path, preview)
        print(f"\n  预览图已保存到: {preview_path}  (截图尺寸: {img_w}×{img_h})\n")

        # 逐个处理每张牌
        for i in range(reg.max_tiles + 1):
            is_drawn = i >= reg.max_tiles
            x_rel, y_rel, tw_rel, th_rel = regions.get_tile_rect(i, reg.max_tiles, is_drawn)

            x_start = int(x_rel * img_w)
            y_start = int(y_rel * img_h)
            tile_w = max(1, int(tw_rel * img_w))
            tile_h = max(1, int(th_rel * img_h))

            tile_img = screenshot[y_start:y_start + tile_h, x_start:x_start + tile_w]
            if tile_img.size == 0:
                continue

            # 保存单张预览
            tile_preview = f"logs/tile_pos_{i}.png"
            cv2.imwrite(tile_preview, tile_img)

            label = "摸牌" if is_drawn else f"手牌[{i}]"
            name = input(f"  {label} 的牌名（1m/9z/skip/quit）> ").strip().lower()

            if name == "quit":
                print("\n退出捕获")
                return
            if name in ("skip", ""):
                continue
            if name not in ALL_TILE_NAMES:
                print(f"  ⚠ 无效牌名「{name}」，有效值如：1m, 5p, 7z")
                continue

            save_path = out / f"{name}.png"
            cv2.imwrite(str(save_path), tile_img)
            print(f"  ✅ 已保存: {save_path}")

        saved_count = len(list(out.glob("*.png")))
        print(f"\n  当前共 {saved_count}/{len(ALL_TILE_NAMES)} 个牌型模板\n")

    print("\n模板捕获完成！")
    print(f"模板目录: {out.resolve()}")


# ──────────────────────────────────────────────
# 按钮模板捕获
# ──────────────────────────────────────────────

def capture_button_templates(output_dir: str = "templates/buttons"):
    """
    交互式捕获操作按钮模板

    说明：
    - 在游戏出现碰/吃/杠等按钮时运行
    - 拖动鼠标框选按钮区域后输入按钮名称
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    BUTTON_NAMES = ["pon", "chi", "kan", "riichi", "tsumo", "ron", "skip"]
    BUTTON_LABELS = {
        "pon": "碰",
        "chi": "吃",
        "kan": "杠",
        "riichi": "立直",
        "tsumo": "自摸",
        "ron": "荣和",
        "skip": "跳过/过",
    }

    sc = ScreenCapture()
    sc.find_game_window()

    print("\n" + "=" * 52)
    print("  雀魂按钮模板捕获工具")
    print("=" * 52)
    print("说明：当屏幕上出现操作按钮时，")
    print("      在弹出的预览窗口中用鼠标拖选按钮区域，")
    print("      然后输入按钮名称保存。")
    print()
    print("按钮名称映射：")
    for k, v in BUTTON_LABELS.items():
        print(f"  {k:8s} → {v}")
    print()

    # 全局变量，用于鼠标回调
    _sel_start = [None]
    _sel_end = [None]
    _selecting = [False]
    _selected_roi = [None]
    _base_img = [None]

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            _sel_start[0] = (x, y)
            _sel_end[0] = (x, y)
            _selecting[0] = True
        elif event == cv2.EVENT_MOUSEMOVE and _selecting[0]:
            _sel_end[0] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            _sel_end[0] = (x, y)
            _selecting[0] = False
            x1 = min(_sel_start[0][0], _sel_end[0][0])
            y1 = min(_sel_start[0][1], _sel_end[0][1])
            x2 = max(_sel_start[0][0], _sel_end[0][0])
            y2 = max(_sel_start[0][1], _sel_end[0][1])
            if x2 - x1 > 5 and y2 - y1 > 5:
                _selected_roi[0] = (x1, y1, x2, y2)

    WIN = "按钮捕获（拖动选区 | q退出 | s保存）"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    display_w = min(1280, sc.window_size[0])
    display_h = int(display_w * sc.window_size[1] / sc.window_size[0])
    cv2.resizeWindow(WIN, display_w, display_h)
    cv2.setMouseCallback(WIN, mouse_cb)

    print("窗口已打开，等待截图（按空格键截图，q退出）...")

    while True:
        key = cv2.waitKey(1) & 0xFF

        # 空格：刷新截图
        if key == ord(" "):
            _base_img[0] = sc.capture()
            _selected_roi[0] = None
            print("  已截图")

        # q：退出
        if key == ord("q"):
            break

        # s：保存选区
        if key == ord("s") and _selected_roi[0] and _base_img[0] is not None:
            x1, y1, x2, y2 = _selected_roi[0]
            img_h, img_w = _base_img[0].shape[:2]
            scale_x = img_w / display_w
            scale_y = img_h / display_h
            rx1, ry1 = int(x1 * scale_x), int(y1 * scale_y)
            rx2, ry2 = int(x2 * scale_x), int(y2 * scale_y)
            roi = _base_img[0][ry1:ry2, rx1:rx2]

            print(f"  选区尺寸: {roi.shape[1]}×{roi.shape[0]}")
            btn_name = input(f"  按钮名称（{'/'.join(BUTTON_NAMES)}）> ").strip().lower()
            if btn_name in BUTTON_NAMES:
                save_path = out / f"{btn_name}.png"
                cv2.imwrite(str(save_path), roi)
                print(f"  ✅ 已保存: {save_path}")
                _selected_roi[0] = None
            else:
                print(f"  ⚠ 无效按钮名称")

        # 显示预览
        if _base_img[0] is not None:
            disp = cv2.resize(_base_img[0], (display_w, display_h))
            if _sel_start[0] and _sel_end[0]:
                cv2.rectangle(disp, _sel_start[0], _sel_end[0], (0, 0, 255), 2)
            if _selected_roi[0]:
                x1, y1, x2, y2 = _selected_roi[0]
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow(WIN, disp)
        else:
            blank = __import__("numpy").zeros((display_h, display_w, 3), dtype=__import__("numpy").uint8)
            cv2.putText(blank, "Press SPACE to take screenshot", (50, display_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imshow(WIN, blank)

    cv2.destroyAllWindows()
    saved = list(out.glob("*.png"))
    print(f"\n已捕获 {len(saved)} 个按钮模板：{', '.join(p.stem for p in saved)}")


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="雀魂模板捕获工具")
    parser.add_argument(
        "--mode", choices=["tiles", "buttons", "all"], default="tiles",
        help="捕获模式：tiles=牌型  buttons=按钮  all=两者（默认：tiles）"
    )
    parser.add_argument(
        "--output", default="templates",
        help="模板根目录（默认：templates）"
    )
    args = parser.parse_args()

    if args.mode in ("tiles", "all"):
        capture_tile_templates(f"{args.output}/tiles")

    if args.mode in ("buttons", "all"):
        capture_button_templates(f"{args.output}/buttons")


if __name__ == "__main__":
    main()
