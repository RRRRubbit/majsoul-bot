"""
手牌区域校准工具
通过可视化界面帮助用户调整手牌坐标参数，适配不同分辨率

使用说明：
  python tools/calibrate_regions.py

操作说明（在弹出的窗口中）：
  空格    - 刷新截图
  鼠标拖选 - 框选手牌区域（从第1张到第13张）
  s      - 保存当前校准结果
  r      - 重置选区
  d      - 切换调试模式（显示当前区域网格）
  q      - 退出
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

from majsoul_bot.vision.screen_capture import ScreenCapture
from majsoul_bot.vision.regions import ScreenRegions, DEFAULT_REGIONS

# ──────────────────────────────────────────────
# 鼠标回调状态
# ──────────────────────────────────────────────
_state = {
    "start": None,
    "end": None,
    "selecting": False,
    "selected": None,   # (x1, y1, x2, y2) 在显示图上的坐标
    "show_grid": True,
}

def _mouse_callback(event, x, y, flags, param):
    s = _state
    if event == cv2.EVENT_LBUTTONDOWN:
        s["start"] = (x, y)
        s["end"] = (x, y)
        s["selecting"] = True
        s["selected"] = None
    elif event == cv2.EVENT_MOUSEMOVE and s["selecting"]:
        s["end"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        s["end"] = (x, y)
        s["selecting"] = False
        x1 = min(s["start"][0], s["end"][0])
        y1 = min(s["start"][1], s["end"][1])
        x2 = max(s["start"][0], s["end"][0])
        y2 = max(s["start"][1], s["end"][1])
        if x2 - x1 > 10 and y2 - y1 > 5:
            s["selected"] = (x1, y1, x2, y2)


def _draw_grid(display: np.ndarray, regions: ScreenRegions,
               orig_w: int, orig_h: int,
               disp_w: int, disp_h: int) -> np.ndarray:
    """在显示图上绘制当前区域网格"""
    out = display.copy()
    reg = regions.hand
    sx = disp_w / orig_w
    sy = disp_h / orig_h

    # 13 张手牌
    for i in range(reg.max_tiles):
        x_rel, y_rel, tw_rel, th_rel = regions.get_tile_rect(i, reg.max_tiles, False)
        x1 = int(x_rel * orig_w * sx)
        y1 = int(y_rel * orig_h * sy)
        x2 = x1 + int(tw_rel * orig_w * sx)
        y2 = y1 + int(th_rel * orig_h * sy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 100), 1)
        cv2.putText(out, str(i), (x1 + 2, y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 100), 1)

    # 摸牌
    x_rel, y_rel, tw_rel, th_rel = regions.get_tile_rect(13, reg.max_tiles, True)
    x1 = int(x_rel * orig_w * sx)
    y1 = int(y_rel * orig_h * sy)
    x2 = x1 + int(tw_rel * orig_w * sx)
    y2 = y1 + int(th_rel * orig_h * sy)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)
    cv2.putText(out, "摸", (x1 + 2, y1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

    return out


def calibrate():
    sc = ScreenCapture()
    sc.find_game_window()
    regions = ScreenRegions.load_from_json()  # 加载已有校准（若存在）

    orig_w, orig_h = sc.window_size
    disp_w = min(1440, orig_w)
    disp_h = int(disp_w * orig_h / orig_w)

    WIN = "区域校准（空格截图 | 拖选手牌范围 | s保存 | r重置 | d切换网格 | q退出）"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, disp_w, disp_h)
    cv2.setMouseCallback(WIN, _mouse_callback)

    screenshot = None
    print("=" * 52)
    print("  手牌区域校准工具")
    print("=" * 52)
    print("  按空格键截图，然后用鼠标拖选完整的13张手牌区域")
    print("  (从第1张手牌左侧 到 第13张手牌右侧)")
    print()

    while True:
        key = cv2.waitKey(30) & 0xFF

        # 空格 → 刷新截图
        if key == ord(" "):
            screenshot = sc.capture()
            orig_h, orig_w = screenshot.shape[:2]
            disp_h = int(disp_w * orig_h / orig_w)
            cv2.resizeWindow(WIN, disp_w, disp_h)
            _state["selected"] = None
            print(f"  截图刷新，尺寸 {orig_w}×{orig_h}")

        # r → 重置选区
        if key == ord("r"):
            _state["selected"] = None
            _state["start"] = None
            _state["end"] = None

        # d → 切换网格显示
        if key == ord("d"):
            _state["show_grid"] = not _state["show_grid"]

        # s → 保存校准
        if key == ord("s") and _state["selected"] and screenshot is not None:
            x1d, y1d, x2d, y2d = _state["selected"]
            # 将显示坐标转回原始坐标
            sx = orig_w / disp_w
            sy = orig_h / disp_h
            x1 = int(x1d * sx)
            y1 = int(y1d * sy)
            x2 = int(x2d * sx)
            y2 = int(y2d * sy)

            # 计算归一化坐标
            new_x_start = x1 / orig_w
            new_y_start = y1 / orig_h
            new_tile_w = (x2 - x1) / orig_w / regions.hand.max_tiles
            new_tile_h = (y2 - y1) / orig_h

            regions.hand.x_start = round(new_x_start, 4)
            regions.hand.y_start = round(new_y_start, 4)
            regions.hand.tile_width = round(new_tile_w, 4)
            regions.hand.tile_height = round(new_tile_h, 4)

            regions.save_to_json()
            print(f"\n  ✅ 校准已保存！")
            print(f"     x_start={regions.hand.x_start}")
            print(f"     y_start={regions.hand.y_start}")
            print(f"     tile_width={regions.hand.tile_width}")
            print(f"     tile_height={regions.hand.tile_height}")

        # q → 退出
        if key == ord("q"):
            break

        # 构建显示图
        if screenshot is not None:
            disp = cv2.resize(screenshot, (disp_w, disp_h))
        else:
            disp = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
            cv2.putText(disp, "Press SPACE to take screenshot",
                        (30, disp_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

        # 绘制当前区域网格
        if _state["show_grid"] and screenshot is not None:
            disp = _draw_grid(disp, regions, orig_w, orig_h, disp_w, disp_h)

        # 绘制拖选框
        if _state["selecting"] and _state["start"] and _state["end"]:
            cv2.rectangle(disp, _state["start"], _state["end"], (0, 0, 255), 2)
        if _state["selected"]:
            x1, y1, x2, y2 = _state["selected"]
            cv2.rectangle(disp, (x1, y1), (x2, y2), (50, 50, 255), 2)
            cv2.putText(disp, "按 s 保存", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 255), 2)

        # 顶部提示栏
        cv2.rectangle(disp, (0, 0), (disp_w, 30), (30, 30, 30), -1)
        grid_hint = "网格:ON" if _state["show_grid"] else "网格:OFF"
        cv2.putText(disp,
                    f"空格=截图  拖选=框选手牌  s=保存  r=重置  d={grid_hint}  q=退出",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 100), 1)

        cv2.imshow(WIN, disp)

    cv2.destroyAllWindows()
    print("\n校准工具已关闭。")


if __name__ == "__main__":
    calibrate()
