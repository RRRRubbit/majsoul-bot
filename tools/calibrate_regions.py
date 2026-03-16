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
# 可标定的区域列表
CALIBRATION_REGIONS = [
    ("hand", "手牌区域", (0, 255, 100)),
    ("dora", "宝牌区域", (255, 200, 0)),
    ("wall", "牌堆区域", (255, 100, 200)),
    ("button_scan", "按钮扫描区域", (100, 200, 255)),
    ("meld_self", "自家副露区域", (200, 100, 255)),
    ("meld_right", "右家副露区域", (255, 150, 100)),
    ("meld_opposite", "对家副露区域", (150, 255, 100)),
    ("meld_left", "左家副露区域", (100, 255, 200)),
]

_state = {
    "start": None,
    "end": None,
    "selecting": False,
    "selected": None,   # (x1, y1, x2, y2) 在显示图上的坐标
    "show_grid": True,
    "current_region_idx": 0,  # 当前标定区域索引
}

def _mouse_callback(event, x, y, flags, param):
    """鼠标事件回调函数"""
    s = _state
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"  [DEBUG] 鼠标按下: ({x}, {y})")
        s["start"] = (x, y)
        s["end"] = (x, y)
        s["selecting"] = True
        s["selected"] = None
    elif event == cv2.EVENT_MOUSEMOVE and s["selecting"]:
        s["end"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        print(f"  [DEBUG] 鼠标释放: ({x}, {y})")
        s["end"] = (x, y)
        s["selecting"] = False
        x1 = min(s["start"][0], s["end"][0])
        y1 = min(s["start"][1], s["end"][1])
        x2 = max(s["start"][0], s["end"][0])
        y2 = max(s["start"][1], s["end"][1])
        if x2 - x1 > 10 and y2 - y1 > 5:
            s["selected"] = (x1, y1, x2, y2)
            print(f"  [DEBUG] 选区确定: ({x1}, {y1}) -> ({x2}, {y2})")
        else:
            print(f"  [DEBUG] 选区太小，已忽略 (宽度={x2-x1}, 高度={y2-y1})")


def _draw_grid(display: np.ndarray, regions: ScreenRegions,
               orig_w: int, orig_h: int,
               disp_w: int, disp_h: int,
               current_region: str) -> np.ndarray:
    """在显示图上绘制当前区域网格"""
    out = display.copy()
    sx = disp_w / orig_w
    sy = disp_h / orig_h

    # 绘制当前选中的区域
    if current_region == "hand":
        reg = regions.hand
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
        cv2.putText(out, "drawn", (x1 + 2, y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
    else:
        # 绘制其他区域
        try:
            # 获取区域坐标
            if current_region == "button_scan":
                x_rel = regions.button_scan_x
                y_rel = regions.button_scan_y
                w_rel = regions.button_scan_w
                h_rel = regions.button_scan_h
            else:
                x_rel, y_rel, w_rel, h_rel = regions.get_named_rect(current_region)
            
            x1 = int(x_rel * orig_w * sx)
            y1 = int(y_rel * orig_h * sy)
            x2 = x1 + int(w_rel * orig_w * sx)
            y2 = y1 + int(h_rel * orig_h * sy)
            
            # 获取当前区域的颜色
            color = (0, 255, 100)
            for name, _, c in CALIBRATION_REGIONS:
                if name == current_region:
                    color = c
                    break
            
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, current_region, (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception as e:
            pass

    return out


def calibrate():
    sc = ScreenCapture()
    sc.find_game_window()
    regions = ScreenRegions.load_from_json()  # 加载已有校准（若存在）

    orig_w, orig_h = sc.window_size
    disp_w = min(1440, orig_w)
    disp_h = int(disp_w * orig_h / orig_w)

    WIN = "Hand Region Calibration Tool"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, disp_w, disp_h)
    cv2.setMouseCallback(WIN, _mouse_callback)
    print(f"  [DEBUG] 窗口已创建: '{WIN}'")
    print(f"  [DEBUG] 鼠标回调已设置")

    screenshot = None
    print("=" * 60)
    print("  区域校准工具 - 支持多区域标定")
    print("=" * 60)
    print("  快捷键说明：")
    print("    Space    - 刷新截图")
    print("    Tab/N    - 切换到下一个区域")
    print("    P        - 切换到上一个区域")
    print("    1-8      - 直接跳转到指定区域")
    print("    拖选鼠标  - 框选当前区域")
    print("    S        - 保存当前区域")
    print("    R        - 重置选区")
    print("    D        - 切换网格显示")
    print("    Q        - 退出")
    print()
    print(f"  当前区域: [{CALIBRATION_REGIONS[0][1]}]")
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

        # Tab/n → 下一个区域
        if key == 9 or key == ord("n"):  # Tab key or 'n'
            _state["current_region_idx"] = (_state["current_region_idx"] + 1) % len(CALIBRATION_REGIONS)
            _state["selected"] = None
            region_name, region_label, _ = CALIBRATION_REGIONS[_state["current_region_idx"]]
            print(f"\n  📍 切换到: [{region_label}] ({region_name})")
        
        # p → 上一个区域
        if key == ord("p"):
            _state["current_region_idx"] = (_state["current_region_idx"] - 1) % len(CALIBRATION_REGIONS)
            _state["selected"] = None
            region_name, region_label, _ = CALIBRATION_REGIONS[_state["current_region_idx"]]
            print(f"\n  📍 切换到: [{region_label}] ({region_name})")
        
        # 数字键 1-8 → 直接跳转到区域
        if ord("1") <= key <= ord("8"):
            idx = key - ord("1")
            if idx < len(CALIBRATION_REGIONS):
                _state["current_region_idx"] = idx
                _state["selected"] = None
                region_name, region_label, _ = CALIBRATION_REGIONS[idx]
                print(f"\n  📍 切换到: [{region_label}] ({region_name})")

        # r → 重置选区
        if key == ord("r"):
            _state["selected"] = None
            _state["start"] = None
            _state["end"] = None
            print("  重置选区")

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

            # 获取当前区域
            region_name, region_label, _ = CALIBRATION_REGIONS[_state["current_region_idx"]]
            
            # 计算归一化坐标
            new_x = x1 / orig_w
            new_y = y1 / orig_h
            new_w = (x2 - x1) / orig_w
            new_h = (y2 - y1) / orig_h
            
            if region_name == "hand":
                # 手牌区域特殊处理
                drawn_gap_ratio = 0.291645
                new_tile_w = new_w / regions.hand.max_tiles
                new_drawn_gap = new_tile_w * drawn_gap_ratio
                regions.hand.x_start = round(new_x, 4)
                regions.hand.y_start = round(new_y, 4)
                regions.hand.tile_width = round(new_tile_w, 4)
                regions.hand.tile_height = round(new_h, 4)
                regions.hand.drawn_gap = round(new_drawn_gap, 4)
                print(f"\n  ✅ [{region_label}] 校准已保存！")
                print(f"     x_start={regions.hand.x_start}")
                print(f"     y_start={regions.hand.y_start}")
                print(f"     tile_width={regions.hand.tile_width}")
                print(f"     tile_height={regions.hand.tile_height}")
                print(f"     drawn_gap={regions.hand.drawn_gap}")
            elif region_name == "button_scan":
                # 按钮扫描区域
                regions.button_scan_x = round(new_x, 4)
                regions.button_scan_y = round(new_y, 4)
                regions.button_scan_w = round(new_w, 4)
                regions.button_scan_h = round(new_h, 4)
                print(f"\n  ✅ [{region_label}] 校准已保存！")
                print(f"     x={regions.button_scan_x}, y={regions.button_scan_y}")
                print(f"     w={regions.button_scan_w}, h={regions.button_scan_h}")
            else:
                # 其他矩形区域
                try:
                    regions.set_named_rect(region_name, new_x, new_y, new_w, new_h)
                    print(f"\n  ✅ [{region_label}] 校准已保存！")
                    print(f"     x={round(new_x, 4)}, y={round(new_y, 4)}")
                    print(f"     w={round(new_w, 4)}, h={round(new_h, 4)}")
                except Exception as e:
                    print(f"  ❌ 保存失败: {e}")
                    
            regions.save_to_json()
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
            region_name = CALIBRATION_REGIONS[_state["current_region_idx"]][0]
            disp = _draw_grid(disp, regions, orig_w, orig_h, disp_w, disp_h, region_name)

        # 绘制拖选框
        if _state["selecting"] and _state["start"] and _state["end"]:
            cv2.rectangle(disp, _state["start"], _state["end"], (0, 0, 255), 2)
        if _state["selected"]:
            x1, y1, x2, y2 = _state["selected"]
            cv2.rectangle(disp, (x1, y1), (x2, y2), (50, 50, 255), 2)
            cv2.putText(disp, "按 s 保存", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 255), 2)

        # 顶部提示栏（两行）
        cv2.rectangle(disp, (0, 0), (disp_w, 52), (30, 30, 30), -1)
        
        # 第一行：当前区域
        region_name, region_label, region_color = CALIBRATION_REGIONS[_state["current_region_idx"]]
        region_text = f"Current Region: [{region_label}] ({_state['current_region_idx']+1}/{len(CALIBRATION_REGIONS)})"
        cv2.putText(disp, region_text, (8, 18), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, region_color, 1)
        
        # 第二行：快捷键
        grid_hint = "Grid:ON" if _state["show_grid"] else "Grid:OFF"
        status_text = f"Space=Shot | Tab/P=Switch | 1-8=Jump | S=Save | R=Reset | D={grid_hint} | Q=Quit"
        cv2.putText(disp, status_text, (8, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 100), 1)

        cv2.imshow(WIN, disp)

    cv2.destroyAllWindows()
    print("\n校准工具已关闭。")


if __name__ == "__main__":
    calibrate()
