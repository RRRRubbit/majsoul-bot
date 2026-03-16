"""
自动校准工具 - 基于手牌区域几何计算其他区域
只需标定手牌区域，自动计算其他所有区域的位置
"""

import json
import sys
from pathlib import Path
from typing import Dict

_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

from majsoul_bot.vision.regions import ScreenRegions


def auto_calibrate_from_hand(regions: ScreenRegions) -> Dict[str, float]:
    """
    基于手牌区域的标定结果，通过几何关系自动计算其他区域
    
    麻将桌面几何关系：
    - 手牌区域在底部中央
    - 屏幕中心约在 (0.5, 0.5)
    - 其他三家呈90度旋转对称分布
    """
    
    hand = regions.hand
    
    # 计算手牌区域的关键尺寸
    hand_full_width = hand.tile_width * hand.max_tiles
    hand_center_x = hand.x_start + hand_full_width / 2
    hand_center_y = hand.y_start + hand.tile_height / 2
    
    # 屏幕中心（理论上应该在正中）
    screen_center_x = 0.5
    screen_center_y = 0.5
    
    print("=" * 60)
    print("  自动校准 - 基于手牌区域几何计算")
    print("=" * 60)
    print(f"\n手牌区域分析:")
    print(f"  位置: ({hand.x_start:.4f}, {hand.y_start:.4f})")
    print(f"  单张牌: {hand.tile_width:.4f} × {hand.tile_height:.4f}")
    print(f"  13张总宽: {hand_full_width:.4f}")
    print(f"  手牌中心: ({hand_center_x:.4f}, {hand_center_y:.4f})")
    
    # ===== 1. 宝牌区域（右上角固定位置）=====
    # 宝牌通常在屏幕右上方，距离右边缘一定距离
    dora_width = hand.tile_width * 5  # 通常显示5个宝牌指示
    dora_height = hand.tile_height * 0.6  # 宝牌比手牌小一些
    dora_x = 0.98 - dora_width  # 右侧对齐，留2%边距
    dora_y = 0.04  # 距离顶部4%
    
    # ===== 2. 牌堆区域（中央偏下）=====
    # 牌堆在屏幕中央稍下方
    wall_width = hand_full_width * 0.35  # 约为手牌宽度的35%
    wall_height = hand.tile_height * 1.3  # 略高于单张牌
    wall_x = screen_center_x - wall_width / 2
    wall_y = screen_center_y - wall_height / 2 + 0.08  # 中央偏下一点
    
    # ===== 3. 按钮扫描区域（手牌上方中央）=====
    # 操作按钮（碰/吃/杠等）出现在手牌上方
    button_width = hand_full_width * 0.6
    button_height = hand.tile_height * 1.5
    button_x = hand_center_x - button_width / 2
    button_y = hand.y_start - button_height - 0.15  # 手牌上方，留15%间距
    
    # ===== 4. 自家副露区域（手牌右侧）=====
    # 自己碰/吃/杠的牌显示在手牌右侧
    meld_self_width = hand_full_width * 0.5
    meld_self_height = hand.tile_height * 1.2
    meld_self_x = hand.x_start + hand_full_width + 0.02  # 手牌右侧，留2%间距
    meld_self_y = hand.y_start
    
    # ===== 5. 右家副露区域（右侧垂直）=====
    # 右家的副露牌在右侧垂直排列
    meld_right_width = hand.tile_height * 1.5  # 宽度相当于横向牌的高度
    meld_right_height = hand_full_width * 0.5  # 高度相当于手牌宽度的一半
    meld_right_x = 0.98 - meld_right_width  # 右对齐
    meld_right_y = screen_center_y - meld_right_height / 2
    
    # ===== 6. 对家副露区域（顶部水平）=====
    # 对家的副露牌在顶部水平排列
    meld_opposite_width = hand_full_width * 0.45
    meld_opposite_height = hand.tile_height * 0.9  # 比手牌略小
    meld_opposite_x = screen_center_x - meld_opposite_width / 2
    meld_opposite_y = 0.04
    
    # ===== 7. 左家副露区域（左侧垂直）=====
    # 左家的副露牌在左侧垂直排列
    meld_left_width = hand.tile_height * 1.5
    meld_left_height = hand_full_width * 0.5
    meld_left_x = 0.02  # 左对齐，留2%边距
    meld_left_y = screen_center_y - meld_left_height / 2
    
    # 构建结果字典
    result = {
        # 宝牌区域
        "dora_x": round(dora_x, 4),
        "dora_y": round(dora_y, 4),
        "dora_w": round(dora_width, 4),
        "dora_h": round(dora_height, 4),
        
        # 牌堆区域
        "wall_x": round(wall_x, 4),
        "wall_y": round(wall_y, 4),
        "wall_w": round(wall_width, 4),
        "wall_h": round(wall_height, 4),
        
        # 按钮扫描区域
        "button_scan_x": round(button_x, 4),
        "button_scan_y": round(button_y, 4),
        "button_scan_w": round(button_width, 4),
        "button_scan_h": round(button_height, 4),
        
        # 自家副露
        "meld_self_x": round(meld_self_x, 4),
        "meld_self_y": round(meld_self_y, 4),
        "meld_self_w": round(meld_self_width, 4),
        "meld_self_h": round(meld_self_height, 4),
        
        # 右家副露
        "meld_right_x": round(meld_right_x, 4),
        "meld_right_y": round(meld_right_y, 4),
        "meld_right_w": round(meld_right_width, 4),
        "meld_right_h": round(meld_right_height, 4),
        
        # 对家副露
        "meld_opposite_x": round(meld_opposite_x, 4),
        "meld_opposite_y": round(meld_opposite_y, 4),
        "meld_opposite_w": round(meld_opposite_width, 4),
        "meld_opposite_h": round(meld_opposite_height, 4),
        
        # 左家副露
        "meld_left_x": round(meld_left_x, 4),
        "meld_left_y": round(meld_left_y, 4),
        "meld_left_w": round(meld_left_width, 4),
        "meld_left_h": round(meld_left_height, 4),
    }
    
    print(f"\n自动计算的区域:")
    print(f"\n  宝牌区域: ({result['dora_x']}, {result['dora_y']}) "
          f"{result['dora_w']} × {result['dora_h']}")
    print(f"  牌堆区域: ({result['wall_x']}, {result['wall_y']}) "
          f"{result['wall_w']} × {result['wall_h']}")
    print(f"  按钮区域: ({result['button_scan_x']}, {result['button_scan_y']}) "
          f"{result['button_scan_w']} × {result['button_scan_h']}")
    print(f"  自家副露: ({result['meld_self_x']}, {result['meld_self_y']}) "
          f"{result['meld_self_w']} × {result['meld_self_h']}")
    print(f"  右家副露: ({result['meld_right_x']}, {result['meld_right_y']}) "
          f"{result['meld_right_w']} × {result['meld_right_h']}")
    print(f"  对家副露: ({result['meld_opposite_x']}, {result['meld_opposite_y']}) "
          f"{result['meld_opposite_w']} × {result['meld_opposite_h']}")
    print(f"  左家副露: ({result['meld_left_x']}, {result['meld_left_y']}) "
          f"{result['meld_left_w']} × {result['meld_left_h']}")
    
    return result


def apply_and_save(regions: ScreenRegions, auto_values: Dict[str, float]):
    """应用自动计算的值并保存"""
    # 应用到regions对象
    regions.dora_x = auto_values["dora_x"]
    regions.dora_y = auto_values["dora_y"]
    regions.dora_w = auto_values["dora_w"]
    regions.dora_h = auto_values["dora_h"]
    
    regions.wall_x = auto_values["wall_x"]
    regions.wall_y = auto_values["wall_y"]
    regions.wall_w = auto_values["wall_w"]
    regions.wall_h = auto_values["wall_h"]
    
    regions.button_scan_x = auto_values["button_scan_x"]
    regions.button_scan_y = auto_values["button_scan_y"]
    regions.button_scan_w = auto_values["button_scan_w"]
    regions.button_scan_h = auto_values["button_scan_h"]
    
    regions.meld_self_x = auto_values["meld_self_x"]
    regions.meld_self_y = auto_values["meld_self_y"]
    regions.meld_self_w = auto_values["meld_self_w"]
    regions.meld_self_h = auto_values["meld_self_h"]
    
    regions.meld_right_x = auto_values["meld_right_x"]
    regions.meld_right_y = auto_values["meld_right_y"]
    regions.meld_right_w = auto_values["meld_right_w"]
    regions.meld_right_h = auto_values["meld_right_h"]
    
    regions.meld_opposite_x = auto_values["meld_opposite_x"]
    regions.meld_opposite_y = auto_values["meld_opposite_y"]
    regions.meld_opposite_w = auto_values["meld_opposite_w"]
    regions.meld_opposite_h = auto_values["meld_opposite_h"]
    
    regions.meld_left_x = auto_values["meld_left_x"]
    regions.meld_left_y = auto_values["meld_left_y"]
    regions.meld_left_w = auto_values["meld_left_w"]
    regions.meld_left_h = auto_values["meld_left_h"]
    
    # 保存到配置文件
    regions.save_to_json()
    print(f"\n✅ 自动校准结果已保存到配置文件！")
    print(f"   {regions.vision_calibration_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='基于手牌区域自动计算其他区域')
    parser.add_argument('--yes', '-y', action='store_true', help='自动应用，不询问确认')
    args = parser.parse_args()
    
    print("加载当前配置...")
    regions = ScreenRegions.load_from_json()
    
    # 自动计算
    auto_values = auto_calibrate_from_hand(regions)
    
    # 询问是否应用
    print(f"\n" + "=" * 60)
    
    if args.yes:
        response = 'y'
        print("自动应用模式（--yes）")
    else:
        response = input("是否应用自动计算的值并保存？(y/n): ").strip().lower()
    
    if response == 'y':
        apply_and_save(regions, auto_values)
        print("\n✅ 完成！所有区域已基于手牌区域自动校准。")
        print("\n💡 提示：可以使用 tools/calibrate_regions.py 查看和微调各个区域")
    else:
        print("\n❌ 取消应用。配置文件未修改。")
