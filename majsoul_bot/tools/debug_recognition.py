"""
手牌识别调试工具
用于诊断和解决牌型识别问题
"""
import cv2
import numpy as np
import json
from pathlib import Path
import sys
import os

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from majsoul_bot.vision.screen_capture import ScreenCapture
from majsoul_bot.vision.tile_recognizer import TileRecognizer


class RecognitionDebugger:
    """识别调试器"""

    def __init__(self):
        self.screen = ScreenCapture()
        self.recognizer = TileRecognizer()
        self.output_dir = Path("logs/debug_recognition")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def check_templates(self):
        """检查模板文件"""
        print("\n" + "="*60)
        print("📂 模板文件检查")
        print("="*60)

        templates_dir = Path("templates/tiles")
        if not templates_dir.exists():
            print(f"❌ 模板目录不存在: {templates_dir}")
            return False

        templates = list(templates_dir.glob("*.png"))
        if not templates:
            print(f"❌ 模板目录为空: {templates_dir}")
            print("\n💡 解决方案:")
            print("   1. 运行: python tools/capture_templates.py")
            print("   2. 或者: 手动截取牌图片保存到 templates/tiles/")
            return False

        print(f"✅ 找到 {len(templates)} 个模板文件:")
        for tmpl in sorted(templates)[:10]:
            img = cv2.imread(str(tmpl))
            if img is not None:
                h, w = img.shape[:2]
                print(f"   - {tmpl.name:12} ({w}x{h})")
            else:
                print(f"   - {tmpl.name:12} (⚠️ 无法读取)")

        if len(templates) > 10:
            print(f"   ... 还有 {len(templates) - 10} 个文件")

        return True

    def check_calibration(self):
        """检查校准数据"""
        print("\n" + "="*60)
        print("📐 校准数据检查")
        print("="*60)

        calib_file = Path("config/vision_calibration.json")
        if not calib_file.exists():
            print(f"❌ 校准文件不存在: {calib_file}")
            print("\n💡 解决方案:")
            print("   运行: python tools/calibrate_regions.py")
            return False

        try:
            with open(calib_file) as f:
                data = json.load(f)

            print("✅ 校准数据已加载:")
            print(f"   - 手牌区域: ({data['hand_region']['x']}, {data['hand_region']['y']}) "
                  f"{data['hand_region']['width']}x{data['hand_region']['height']}")
            print(f"   - 牌宽度: {data.get('tile_width', 'N/A')}")
            print(f"   - 牌间距: {data.get('tile_spacing', 'N/A')}")
            return True

        except Exception as e:
            print(f"❌ 校准数据读取失败: {e}")
            return False

    def capture_and_analyze(self):
        """截图并分析识别过程"""
        print("\n" + "="*60)
        print("📸 截图并分析手牌")
        print("="*60)

        # 截图
        print("\n🔄 正在截图...")
        screenshot = self.screen.capture_game_area()
        if screenshot is None:
            print("❌ 截图失败（游戏窗口未找到）")
            return

        print(f"✅ 截图成功: {screenshot.shape[1]}x{screenshot.shape[0]}")

        # 保存原始截图
        screenshot_path = self.output_dir / "01_full_screenshot.png"
        cv2.imwrite(str(screenshot_path), screenshot)
        print(f"   保存到: {screenshot_path}")

        # 识别手牌
        print("\n🔍 识别手牌...")
        result = self.recognizer.recognize_hand(screenshot)

        if not result:
            print("❌ 未识别到任何牌")
            return

        # 解包结果
        tiles = [name for name, pos in result]
        positions = [pos for name, pos in result]

        print(f"✅ 识别到 {len(tiles)} 张牌:")
        for i, (tile, pos) in enumerate(zip(tiles, positions)):
            print(f"   [{i:2d}] {tile:15} @ ({pos[0]}, {pos[1]})")

        # 可视化
        self._visualize_recognition(screenshot, tiles, positions)

    def _visualize_recognition(self, screenshot, tiles, positions):
        """可视化识别结果"""
        print("\n🎨 生成可视化图像...")

        # 加载校准数据
        try:
            with open("config/vision_calibration.json") as f:
                calib = json.load(f)
            tile_width = calib.get("tile_width", 40)
            tile_height = calib.get("tile_height", 60)
        except:
            tile_width, tile_height = 40, 60

        vis = screenshot.copy()

        # 绘制每张牌的位置和识别结果
        for i, (tile, (x, y)) in enumerate(zip(tiles, positions)):
            # 绘制矩形框
            if tile.startswith("unknown"):
                color = (0, 0, 255)  # 红色 - 未识别
            else:
                color = (0, 255, 0)  # 绿色 - 已识别

            cv2.rectangle(vis, (x, y), (x + tile_width, y + tile_height), color, 2)

            # 绘制序号
            cv2.putText(vis, f"{i}", (x + 5, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 绘制识别结果
            cv2.putText(vis, tile, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 保存可视化结果
        vis_path = self.output_dir / "02_recognition_result.png"
        cv2.imwrite(str(vis_path), vis)
        print(f"   保存到: {vis_path}")

        # 统计
        unknown_count = sum(1 for t in tiles if t.startswith("unknown"))
        recognized_count = len(tiles) - unknown_count

        print(f"\n📊 识别统计:")
        print(f"   - 已识别: {recognized_count}/{len(tiles)} ({recognized_count*100/len(tiles):.1f}%)")
        print(f"   - 未识别: {unknown_count}/{len(tiles)}")

    def extract_tile_samples(self):
        """提取每张牌的小图片用于分析"""
        print("\n" + "="*60)
        print("✂️  提取单张牌图片")
        print("="*60)

        # 截图
        screenshot = self.screen.capture_game_area()
        if screenshot is None:
            print("❌ 截图失败")
            return

        # 识别位置
        result = self.recognizer.recognize_hand(screenshot)
        if not result:
            print("❌ 未找到手牌")
            return

        tiles = [name for name, pos in result]
        positions = [pos for name, pos in result]

        # 加载校准数据
        try:
            with open("config/vision_calibration.json") as f:
                calib = json.load(f)
            tile_width = calib.get("tile_width", 40)
            tile_height = calib.get("tile_height", 60)
        except:
            tile_width, tile_height = 40, 60

        # 创建输出目录
        tiles_dir = self.output_dir / "tiles"
        tiles_dir.mkdir(exist_ok=True)

        print(f"\n✂️  提取 {len(tiles)} 张牌:")

        for i, (tile, (cx, cy)) in enumerate(zip(tiles, positions)):
            # 根据中心点切割牌图片
            x = cx - tile_width // 2
            y = cy - tile_height // 2
            tile_img = screenshot[y:y+tile_height, x:x+tile_width]

            if tile_img.size == 0:
                continue

            # 保存
            filename = f"tile_{i:02d}_{tile}.png"
            filepath = tiles_dir / filename
            cv2.imwrite(str(filepath), tile_img)

            print(f"   [{i:2d}] {tile:15} → {filename}")

        print(f"\n✅ 所有牌图片已保存到: {tiles_dir}")
        print("\n💡 使用这些图片作为模板:")
        print(f"   1. 打开 {tiles_dir}")
        print(f"   2. 删除 unknown 的图片")
        print(f"   3. 将其他图片重命名（如 1m.png, 5p.png）")
        print(f"   4. 复制到 templates/tiles/ 目录")

    def test_template_matching(self):
        """测试模板匹配效果"""
        print("\n" + "="*60)
        print("🧪 测试模板匹配")
        print("="*60)

        # 检查是否有模板
        templates_dir = Path("templates/tiles")
        templates = list(templates_dir.glob("*.png"))
        if not templates:
            print("❌ 没有模板文件")
            return

        # 截图
        screenshot = self.screen.capture_game_area()
        if screenshot is None:
            print("❌ 截图失败")
            return

        # 测试第一张牌的匹配
        result = self.recognizer.recognize_hand(screenshot)
        if not result:
            print("❌ 未找到手牌")
            return

        # 加载校准数据
        try:
            with open("config/vision_calibration.json") as f:
                calib = json.load(f)
            tile_width = calib.get("tile_width", 40)
            tile_height = calib.get("tile_height", 60)
        except:
            tile_width, tile_height = 40, 60

        # 选择第一张牌
        tile_name, (cx, cy) = result[0]
        x = cx - tile_width // 2
        y = cy - tile_height // 2
        tile_img = screenshot[y:y+tile_height, x:x+tile_width]

        print(f"\n🎯 测试第一张牌 @ ({cx}, {cy}):")
        print(f"   图片尺寸: {tile_img.shape[1]}x{tile_img.shape[0]}")

        # 对每个模板进行匹配
        print(f"\n🔍 与所有模板匹配:")
        scores = []

        for tmpl_path in sorted(templates):
            tmpl = cv2.imread(str(tmpl_path))
            if tmpl is None:
                continue

            # 调整模板大小
            tmpl_resized = cv2.resize(tmpl, (tile_img.shape[1], tile_img.shape[0]))

            # 模板匹配
            result_match = cv2.matchTemplate(tile_img, tmpl_resized, cv2.TM_CCOEFF_NORMED)
            score = result_match[0][0]

            scores.append((tmpl_path.stem, score))

        # 排序并显示
        scores.sort(key=lambda x: x[1], reverse=True)

        threshold = self.recognizer.threshold
        print(f"\n📊 匹配结果（阈值 = {threshold:.2f}）:")
        for name, score in scores[:10]:
            status = "✅" if score >= threshold else "❌"
            bar = "█" * int(score * 30)
            print(f"   {status} {score:.3f} {bar:30} {name}")

        if scores[0][1] < threshold:
            print(f"\n⚠️  最高得分 {scores[0][1]:.3f} 低于阈值 {threshold:.2f}")
            print(f"💡 建议:")
            print(f"   1. 降低阈值（config.yaml 中设置 template_threshold: 0.60）")
            print(f"   2. 重新捕获更清晰的模板")


def main():
    """主函数"""
    debugger = RecognitionDebugger()

    print("\n" + "="*60)
    print("🔧 手牌识别调试工具")
    print("="*60)

    # 1. 检查模板
    has_templates = debugger.check_templates()

    # 2. 检查校准
    has_calibration = debugger.check_calibration()

    if not has_calibration:
        print("\n❌ 请先运行校准工具: python tools/calibrate_regions.py")
        return

    # 3. 截图分析
    if has_templates:
        debugger.capture_and_analyze()
        debugger.test_template_matching()
    else:
        print("\n⚠️  没有模板，尝试提取牌图片...")
        debugger.extract_tile_samples()

    print("\n" + "="*60)
    print("✅ 调试完成")
    print("="*60)
    print(f"\n📁 所有调试文件保存在: {debugger.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  已中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
