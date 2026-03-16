"""
测试YOLO麻将牌识别效果
"""
import cv2
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from majsoul_bot.vision.tile_recognizer import TileRecognizer
from majsoul_bot.vision.yolo_tile_detector import get_detector
from majsoul_bot.vision.screen_capture import ScreenCapture
from loguru import logger


def test_yolo_standalone():
    """测试YOLO检测器单独使用"""
    logger.info("=" * 60)
    logger.info("测试1: YOLO检测器单独测试")
    logger.info("=" * 60)
    
    # 获取YOLO检测器
    detector = get_detector()
    
    # 捕获截图
    capturer = ScreenCapture()
    screenshot = capturer.capture()
    
    if screenshot is None:
        logger.error("无法捕获截图")
        return
    
    logger.info(f"截图尺寸: {screenshot.shape}")
    
    # 裁剪手牌区域 (根据默认区域配置)
    img_h, img_w = screenshot.shape[:2]
    crop_y1 = int(0.75 * img_h)  # 手牌大概在屏幕下方25%
    crop_y2 = int(0.88 * img_h)
    crop_x1 = int(0.28 * img_w)
    crop_x2 = int(0.60 * img_w)
    
    hand_region = screenshot[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # YOLO检测
    detections = detector.detect_tiles(hand_region)
    
    logger.info(f"检测到 {len(detections)} 张牌:")
    for tile_name, confidence, (x1, y1, x2, y2) in detections:
        logger.info(f"  {tile_name}: {confidence:.3f} at ({x1},{y1},{x2},{y2})")
    
    # 可视化
    vis_image = detector.visualize_detections(hand_region, detections)
    
    # 保存结果
    output_path = project_root / "test_yolo_standalone.png"
    cv2.imwrite(str(output_path), vis_image)
    logger.info(f"可视化结果已保存: {output_path}")


def test_yolo_in_recognizer():
    """测试集成到TileRecognizer的YOLO识别"""
    logger.info("=" * 60)
    logger.info("测试2: TileRecognizer中的YOLO识别")
    logger.info("=" * 60)
    
    # 创建识别器 (启用YOLO)
    recognizer = TileRecognizer(
        yolo_enabled=True,
        yolo_priority=True,
        nn_enabled=False,  # 禁用NN加快测试
    )
    
    logger.info(f"YOLO模型已加载: {recognizer.has_yolo_model()}")
    logger.info(f"模板已加载: {recognizer.has_templates()}")
    
    # 捕获截图
    capturer = ScreenCapture()
    screenshot = capturer.capture()
    
    if screenshot is None:
        logger.error("无法捕获截图")
        return
    
    # 使用YOLO识别手牌
    results = recognizer.recognize_hand_by_yolo(screenshot, has_drawn_tile=True)
    
    logger.info(f"识别到 {len(results)} 张牌:")
    for tile_name, (cx, cy) in results:
        logger.info(f"  {tile_name} at ({cx}, {cy})")
    
    # 绘制结果
    vis_image = recognizer.draw_scan_results(screenshot, results)
    
    # 保存结果
    output_path = project_root / "test_yolo_recognizer.png"
    cv2.imwrite(str(output_path), vis_image)
    logger.info(f"可视化结果已保存: {output_path}")
    
    # 打印识别详情
    if recognizer.last_recognition_details:
        logger.info("\n识别详情:")
        for detail in recognizer.last_recognition_details:
            logger.info(f"  位置{detail['index']}: {detail['recognized_name']} "
                       f"(置信度: {detail['best_score']:.3f}, "
                       f"摸牌: {detail['is_drawn']})")


def test_yolo_vs_template():
    """对比YOLO与模板匹配的识别效果"""
    logger.info("=" * 60)
    logger.info("测试3: YOLO vs 模板匹配对比")
    logger.info("=" * 60)
    
    # 捕获截图
    capturer = ScreenCapture()
    screenshot = capturer.capture()
    
    if screenshot is None:
        logger.error("无法捕获截图")
        return
    
    # YOLO识别
    recognizer_yolo = TileRecognizer(
        yolo_enabled=True,
        yolo_priority=True,
        nn_enabled=False,
    )
    results_yolo = recognizer_yolo.recognize_hand_by_yolo(screenshot, has_drawn_tile=True)
    
    # 模板匹配识别
    recognizer_template = TileRecognizer(
        yolo_enabled=False,
        nn_enabled=False,
    )
    results_template = recognizer_template.recognize_hand_by_scan(screenshot, has_drawn_tile=True)
    
    logger.info(f"\nYOLO识别结果 ({len(results_yolo)} 张牌):")
    for i, (tile_name, _) in enumerate(results_yolo):
        logger.info(f"  [{i}] {tile_name}")
    
    logger.info(f"\n模板匹配结果 ({len(results_template)} 张牌):")
    for i, (tile_name, _) in enumerate(results_template):
        logger.info(f"  [{i}] {tile_name}")
    
    # 对比差异
    logger.info("\n识别差异:")
    max_len = max(len(results_yolo), len(results_template))
    diff_count = 0
    for i in range(max_len):
        yolo_name = results_yolo[i][0] if i < len(results_yolo) else "N/A"
        template_name = results_template[i][0] if i < len(results_template) else "N/A"
        
        if yolo_name != template_name:
            logger.warning(f"  位置{i}: YOLO={yolo_name}, 模板={template_name}")
            diff_count += 1
    
    if diff_count == 0:
        logger.success("✅ 两种方法识别结果完全一致!")
    else:
        logger.warning(f"⚠️  发现 {diff_count} 处差异")
    
    # 绘制对比图
    vis_yolo = recognizer_yolo.draw_scan_results(screenshot, results_yolo)
    vis_template = recognizer_template.draw_scan_results(screenshot, results_template)
    
    # 并排拼接
    vis_combined = cv2.hconcat([vis_yolo, vis_template])
    
    # 添加标签
    cv2.putText(vis_combined, "YOLO", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(vis_combined, "Template", (vis_yolo.shape[1] + 50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
    
    output_path = project_root / "test_yolo_vs_template.png"
    cv2.imwrite(str(output_path), vis_combined)
    logger.info(f"对比图已保存: {output_path}")


def test_from_image_file(image_path: str):
    """从图像文件测试YOLO识别"""
    logger.info("=" * 60)
    logger.info(f"测试4: 从图像文件识别 - {image_path}")
    logger.info("=" * 60)
    
    # 读取图像
    screenshot = cv2.imread(image_path)
    if screenshot is None:
        logger.error(f"无法读取图像: {image_path}")
        return
    
    logger.info(f"图像尺寸: {screenshot.shape}")
    
    # YOLO识别
    recognizer = TileRecognizer(
        yolo_enabled=True,
        yolo_priority=True,
        nn_enabled=False,
    )
    
    results = recognizer.recognize_hand_by_yolo(screenshot, has_drawn_tile=True)
    
    logger.info(f"识别到 {len(results)} 张牌:")
    for tile_name, (cx, cy) in results:
        logger.info(f"  {tile_name} at ({cx}, {cy})")
    
    # 可视化
    vis_image = recognizer.draw_scan_results(screenshot, results)
    
    # 保存结果
    output_path = Path(image_path).parent / f"{Path(image_path).stem}_yolo_result.png"
    cv2.imwrite(str(output_path), vis_image)
    logger.info(f"结果已保存: {output_path}")


def main():
    """主测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="测试YOLO麻将牌识别效果")
    parser.add_argument("--test", type=int, default=0, 
                       help="选择测试: 0=全部, 1=YOLO单独, 2=Recognizer集成, 3=对比, 4=从文件")
    parser.add_argument("--image", type=str, help="测试图像路径 (用于test=4)")
    
    args = parser.parse_args()
    
    try:
        if args.test == 0 or args.test == 1:
            test_yolo_standalone()
            print()
        
        if args.test == 0 or args.test == 2:
            test_yolo_in_recognizer()
            print()
        
        if args.test == 0 or args.test == 3:
            test_yolo_vs_template()
            print()
        
        if args.test == 4:
            if not args.image:
                logger.error("请使用 --image 参数指定图像路径")
                return
            test_from_image_file(args.image)
        
        logger.success("✅ 所有测试完成!")
        
    except Exception as e:
        logger.exception(f"测试失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
