"""
区域校准工具（扩展版）
通过可视化界面帮助用户调整手牌坐标参数，并标定新增区域。

使用说明：
  python majsoul_bot/tools/calibrate_regions.py

操作说明（在弹出的窗口中）：
  空格       - 刷新截图
  鼠标拖选   - 框选当前选中的目标区域
  Tab        - 切换目标区域（hand/dora/wall/meld_self/meld_right/meld_opposite/meld_left）
  s          - 保存当前校准结果
  r          - 重置当前区域选框
  d          - 切换调试网格显示
  q          - 退出
"""

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CALIBRATION_PATH = PROJECT_ROOT / "config" / "vision_calibration.json"

from majsoul_bot.vision.screen_capture import ScreenCapture
from majsoul_bot.vision.regions import ScreenRegions, DEFAULT_REGIONS

# ──────────────────────────────────────────────
# 可标定的区域列表（按 Tab 循环）
# ──────────────────────────────────────────────
REGION_TARGETS = [
    "hand",
    "dora",
    "wall",
    "meld_self",
    "meld_right",
    "meld_opposite",
    "meld_left",
]
REGION_COLORS = {
    "hand":          (0, 255, 100),
    "dora":          (0, 200, 255),
    "wall":          (255, 180, 0),
    "meld_self":     (255, 80,  80),
    "meld_right":    (255, 140, 0),
    "meld_opposite": (200, 0,  200),
    "meld_left":     (0,  160, 255),
}
REGION_LABELS = {
    "hand":          "手牌区",
    "dora":          "宝牌区",
    "wall":          "牌堆区",
    "meld_self":     "自家副露",
    "meld_right":    "下家副露",
    "meld_opposite": "对家副露",
    "meld_left":     "上家副露",
}

# 每张牌与摸牌之间的间距比例（仅手牌区使用）
_DRAWN_GAP_RATIO = 0.291645

# ──────────────────────────────────────────────
# 鼠标回调状态
# ──────────────────────────────────────────────
_state = {
    "start": None,
    "end": None,
    "selecting": False,
    "selected": None,   # (x1, y1, x2, y2) 显示坐标
    "show_grid": True,
    "target_idx": 0,    # 当前选中的目标区域索引
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


def _draw_all_regions(display: np.ndarray, regions: ScreenRegions,
                      orig_w: int, orig_h: int,
                      disp_w: int, disp_h: int) -> np.ndarray:
    """在显示图上绘制所有区域框与手牌网格。"""
    out = display.copy()
    sx = disp_w / orig_w
    sy = disp_h / orig_h

    def _rect(rx, ry, rw, rh, color, label):
        x1 = int(rx * orig_w * sx)
        y1 = int(ry * orig_h * sy)
        x2 = int((rx + rw) * orig_w * sx)
        y2 = int((ry + rh) * orig_h * sy)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        cv2.putText(out, label, (x1 + 2, y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    # 宝牌 / 牌堆 / 四家副露
    for name in ("dora", "wall", "meld_self", "meld_right", "meld_opposite", "meld_left"):
        rx, ry, rw, rh = regions.get_named_rect(name)
        _rect(rx, ry, rw, rh, REGION_COLORS[name], REGION_LABELS[name])

    # 手牌 13 张网格
    reg = regions.hand
    for i in range(reg.max_tiles):
        x_rel, y_rel, tw_rel, th_rel = regions.get_tile_rect(i, reg.max_tiles, False)
        x1 = int(x_rel * orig_w * sx)
        y1 = int(y_rel * orig_h * sy)
        x2 = x1 + int(tw_rel * orig_w * sx)
        y2 = y1 + int(th_rel * orig_h * sy)
        cv2.rectangle(out, (x1, y1), (x2, y2), REGION_COLORS["hand"], 1)
        cv2.putText(out, str(i), (x1 + 2, y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, REGION_COLORS["hand"], 1)

    # 摸牌位
    x_rel, y_rel, tw_rel, th_rel = regions.get_tile_rect(13, reg.max_tiles, True)
    x1 = int(x_rel * orig_w * sx)
    y1 = int(y_rel * orig_h * sy)
    x2 = x1 + int(tw_rel * orig_w * sx)
    y2 = y1 + int(th_rel * orig_h * sy)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)
    cv2.putText(out, "摸", (x1 + 2, y1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

    return out


def _save_region(regions: ScreenRegions, target: str,
                 x1d: int, y1d: int, x2d: int, y2d: int,
                 orig_w: int, orig_h: int, disp_w: int, disp_h: int):
    """将显示坐标转为归一化坐标并写入 regions。"""
    sx = orig_w / disp_w
    sy = orig_h / disp_h
    x1 = int(x1d * sx)
    y1 = int(y1d * sy)
    x2 = int(x2d * sx)
    y2 = int(y2d * sy)

    if target == "hand":
        reg = regions.hand
        new_x_start = x1 / orig_w
        new_y_start = y1 / orig_h
        new_tile_w = (x2 - x1) / orig_w / reg.max_tiles
        new_tile_h = (y2 - y1) / orig_h
        reg.x_start = round(new_x_start, 4)
        reg.y_start = round(new_y_start, 4)
        reg.tile_width = round(new_tile_w, 4)
        reg.tile_height = round(new_tile_h, 4)
        reg.drawn_gap = round(new_tile_w * _DRAWN_GAP_RATIO, 4)
        print(f"\n  ✅ 手牌区已更新：x_start={reg.x_start} y_start={reg.y_start} "
              f"tile_w={reg.tile_width} tile_h={reg.tile_height} gap={reg.drawn_gap}")
    else:
        rx = round(x1 / orig_w, 4)
        ry = round(y1 / orig_h, 4)
        rw = round((x2 - x1) / orig_w, 4)
        rh = round((y2 - y1) / orig_h, 4)
        regions.set_named_rect(target, rx, ry, rw, rh)
        label = REGION_LABELS.get(target, target)
        print(f"\n  ✅ {label} 已更新：x={rx} y={ry} w={rw} h={rh}")


def calibrate():
    sc = ScreenCapture()
    sc.find_game_window()
    regions = ScreenRegions.load_from_json(str(CALIBRATION_PATH))

    orig_w, orig_h = sc.window_size
    disp_w = min(1440, orig_w)
    disp_h = int(disp_w * orig_h / orig_w)

    WIN = "区域校准（空格截图 | 拖选 | Tab切换区域 | s保存 | r重置 | d网格 | q退出）"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, disp_w, disp_h)
    cv2.setMouseCallback(WIN, _mouse_callback)

    screenshot = None
    print("=" * 60)
    print("  区域校准工具（扩展版）")
    print("=" * 60)
    print("  按 空格 截图，Tab 切换目标区域，拖选框出该区域后按 s 保存")
    print("  可标定区域：" + " / ".join(REGION_LABELS.values()))
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

        # Tab → 切换目标区域
        if key == 9:  # Tab
            _state["target_idx"] = (_state["target_idx"] + 1) % len(REGION_TARGETS)
            _state["selected"] = None
            target = REGION_TARGETS[_state["target_idx"]]
            print(f"  当前目标区域：{REGION_LABELS[target]}")

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
            target = REGION_TARGETS[_state["target_idx"]]
            _save_region(regions, target, x1d, y1d, x2d, y2d,
                         orig_w, orig_h, disp_w, disp_h)
            regions.save_to_json(str(CALIBRATION_PATH))
            print(f"     已保存到: {CALIBRATION_PATH}")

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

        # 绘制所有区域网格
        if _state["show_grid"] and screenshot is not None:
            disp = _draw_all_regions(disp, regions, orig_w, orig_h, disp_w, disp_h)

        # 绘制拖选框
        if _state["selecting"] and _state["start"] and _state["end"]:
            cv2.rectangle(disp, _state["start"], _state["end"], (0, 0, 255), 2)
        if _state["selected"]:
            x1, y1, x2, y2 = _state["selected"]
            target = REGION_TARGETS[_state["target_idx"]]
            color = REGION_COLORS.get(target, (50, 50, 255))
            cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
            cv2.putText(disp, "按 s 保存", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 顶部提示栏
        target = REGION_TARGETS[_state["target_idx"]]
        target_label = REGION_LABELS[target]
        grid_hint = "ON" if _state["show_grid"] else "OFF"
        cv2.rectangle(disp, (0, 0), (disp_w, 32), (30, 30, 30), -1)
        cv2.putText(
            disp,
            f"[Tab切换] 目标:{target_label}  空格=截图  s=保存  r=重置  d=网格({grid_hint})  q=退出",
            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 100), 1,
        )

        cv2.imshow(WIN, disp)

    cv2.destroyAllWindows()
    print("\n校准工具已关闭。")


if __name__ == "__main__":
    calibrate()
