"""
YOLO麻将牌识别实时可视化工具
按 'q' 退出，按 's' 保存当前帧
"""
import cv2
import sys
from pathlib import Path
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from majsoul_bot.vision.screen_capture import ScreenCapture
from majsoul_bot.vision.tile_recognizer import TileRecognizer
from loguru import logger


def main():
    """实时显示YOLO识别效果"""
    
    # 创建识别器（仅启用YOLO）
    recognizer = TileRecognizer(
        yolo_enabled=True,
        yolo_priority=True,
        yolo_conf_threshold=0.3,  # 降低阈值以看到更多检测
        nn_enabled=False,
        templates_dir="templates/tiles",
    )
    
    # 创建屏幕捕获
    capturer = ScreenCapture(auto_topmost=True)
    
    # 查找游戏窗口
    found = capturer.find_game_window()
    if not found:
        logger.error("未找到游戏窗口")
        return
    
    logger.info("✅ 游戏窗口已定位")
    logger.info(f"YOLO检测器状态: {'已加载' if recognizer.has_yolo_model() else '未加载'}")
    logger.info("按 'q' 退出，按 's' 保存当前帧，按 '+' 提高阈值，按 '-' 降低阈值")
    logger.info("-" * 60)
    
    frame_count = 0
    fps_start = time.time()
    fps_count = 0
    fps = 0.0
    
    conf_threshold = 0.3
    
    while True:
        try:
            # 捕获截图
            screenshot = capturer.capture()
            if screenshot is None:
                logger.warning("截图失败")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            fps_count += 1
            
            # 计算FPS
            if time.time() - fps_start >= 1.0:
                fps = fps_count / (time.time() - fps_start)
                fps_start = time.time()
                fps_count = 0
            
            # 更新检测器阈值
            if recognizer.yolo_detector:
                recognizer.yolo_detector.conf_threshold = conf_threshold
            
            # 使用YOLO识别手牌
            results = recognizer.recognize_hand_by_yolo(screenshot, has_drawn_tile=True)
            
            # 绘制识别结果
            vis_image = recognizer.draw_scan_results(screenshot, results)
            
            # 裁剪手牌区域用于详细显示
            img_h, img_w = screenshot.shape[:2]
            h_reg = recognizer.regions.hand
            crop_y1 = max(0, int((h_reg.y_start - 0.02) * img_h))
            crop_y2 = min(img_h, int((h_reg.y_start + h_reg.tile_height + 0.02) * img_h))
            crop_x1 = max(0, int((h_reg.x_start - 0.05) * img_w))
            crop_x2 = min(img_w, int(recognizer.regions.hand_x_max * img_w))
            
            hand_region = screenshot[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # YOLO检测手牌区域
            if recognizer.yolo_detector and hand_region.size > 0:
                detections = recognizer.yolo_detector.detect_tiles(
                    hand_region, 
                    conf_threshold=conf_threshold
                )
                hand_vis = recognizer.yolo_detector.visualize_detections(
                    hand_region, detections, show_conf=True
                )
                
                # 放大手牌区域显示
                if hand_vis.shape[0] > 0 and hand_vis.shape[1] > 0:
                    scale = min(800 / hand_vis.shape[1], 400 / hand_vis.shape[0])
                    if scale > 1.0:
                        new_w = int(hand_vis.shape[1] * scale)
                        new_h = int(hand_vis.shape[0] * scale)
                        hand_vis = cv2.resize(hand_vis, (new_w, new_h), 
                                             interpolation=cv2.INTER_LINEAR)
            else:
                hand_vis = hand_region.copy() if hand_region.size > 0 else screenshot.copy()
            
            # 添加信息文本
            info_text = [
                f"Frame: {frame_count}  FPS: {fps:.1f}",
                f"Detected: {len(results)} tiles",
                f"YOLO Conf: {conf_threshold:.2f}",
                f"Hand Count: {recognizer.last_scan_hand_count}",
                f"Has Drawn: {recognizer.last_scan_has_drawn}",
            ]
            
            y_offset = 30
            for text in info_text:
                cv2.putText(vis_image, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
            
            # 显示识别牌名
            if results:
                tiles_text = f"Tiles: {' '.join([name for name, _ in results])}"
                cv2.putText(vis_image, tiles_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # 显示完整画面
            display = vis_image.copy()
            
            # 缩放以适应屏幕
            max_height = 900
            if display.shape[0] > max_height:
                scale = max_height / display.shape[0]
                new_w = int(display.shape[1] * scale)
                new_h = int(display.shape[0] * scale)
                display = cv2.resize(display, (new_w, new_h))
            
            cv2.imshow('YOLO Real-time Recognition - Full View', display)
            
            # 显示手牌区域放大图
            if hand_vis.size > 0:
                cv2.imshow('YOLO Real-time Recognition - Hand Region', hand_vis)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                logger.info("退出")
                break
            elif key == ord('s'):
                # 保存当前帧
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_path_full = project_root / f"yolo_debug_full_{timestamp}.png"
                save_path_hand = project_root / f"yolo_debug_hand_{timestamp}.png"
                cv2.imwrite(str(save_path_full), vis_image)
                cv2.imwrite(str(save_path_hand), hand_vis)
                logger.info(f"已保存截图: {save_path_full.name}, {save_path_hand.name}")
            elif key == ord('+') or key == ord('='):
                # 提高阈值
                conf_threshold = min(0.9, conf_threshold + 0.05)
                logger.info(f"置信度阈值: {conf_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                # 降低阈值
                conf_threshold = max(0.1, conf_threshold - 0.05)
                logger.info(f"置信度阈值: {conf_threshold:.2f}")
            
            # 控制帧率
            time.sleep(0.03)  # ~30 FPS
            
        except KeyboardInterrupt:
            logger.info("收到中断信号")
            break
        except Exception as e:
            logger.error(f"错误: {e}")
            import traceback
            traceback.print_exc()
            break
    
    cv2.destroyAllWindows()
    logger.info("实时显示已停止")


if __name__ == "__main__":
    main()
