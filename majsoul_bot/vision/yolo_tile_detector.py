"""
YOLOv5 麻将牌检测器
使用训练好的YOLO模型进行麻将牌识别
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from ultralytics import YOLO


class YOLOTileDetector:
    """YOLO麻将牌检测器"""
    
    # 类别ID到麻将牌名称的映射（与训练时的classes.yaml对应）
    CLASS_NAMES = {
        # 万子（Manzu）
        0: '1m', 1: '2m', 2: '3m', 3: '4m', 4: '5m',
        5: '6m', 6: '7m', 7: '8m', 8: '9m',
        # 饼子（Pinzu）
        9: '1p', 10: '2p', 11: '3p', 12: '4p', 13: '5p',
        14: '6p', 15: '7p', 16: '8p', 17: '9p',
        # 索子（Souzu）
        18: '1s', 19: '2s', 20: '3s', 21: '4s', 22: '5s',
        23: '6s', 24: '7s', 25: '8s', 26: '9s',
        # 字牌（Jihai / Honors）
        27: '1z', 28: '2z', 29: '3z', 30: '4z',
        31: '5z', 32: '6z', 33: '7z',
        # 赤宝牌（Red Dora）
        34: '0m', 35: '0p', 36: '0s',
    }
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        """
        初始化YOLO检测器
        
        Args:
            model_path: YOLO模型文件路径
            conf_threshold: 置信度阈值
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        self.conf_threshold = conf_threshold
        self.model = YOLO(str(self.model_path))
        
        print(f"✅ YOLO模型加载成功: {self.model_path}")
    
    def detect_tiles(
        self, 
        image: np.ndarray,
        conf_threshold: Optional[float] = None
    ) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        检测图像中的麻将牌
        
        Args:
            image: 输入图像 (numpy array, BGR格式)
            conf_threshold: 置信度阈值，如果不指定则使用初始化时的值
            
        Returns:
            检测结果列表，每个元素为 (tile_name, confidence, bbox)
            bbox格式: (x1, y1, x2, y2)
        """
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        
        # 运行YOLO推理
        results = self.model(image, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # 获取类别和置信度
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                
                # 获取麻将牌名称
                tile_name = self.CLASS_NAMES.get(class_id, f"unknown_{class_id}")
                
                detections.append((
                    tile_name,
                    confidence,
                    (int(x1), int(y1), int(x2), int(y2))
                ))
        
        return detections
    
    def detect_hand_tiles(
        self,
        image: np.ndarray,
        sort_by_x: bool = True
    ) -> List[str]:
        """
        检测手牌区域的麻将牌并按x坐标排序
        
        Args:
            image: 手牌区域图像
            sort_by_x: 是否按x坐标排序
            
        Returns:
            按位置排序的麻将牌名称列表
        """
        detections = self.detect_tiles(image)
        
        if sort_by_x:
            # 按x坐标排序（从左到右）
            detections.sort(key=lambda x: x[2][0])
        
        return [tile_name for tile_name, _, _ in detections]
    
    def visualize_detections(
        self,
        image: np.ndarray,
        detections: List[Tuple[str, float, Tuple[int, int, int, int]]],
        show_conf: bool = True
    ) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 原始图像
            detections: 检测结果
            show_conf: 是否显示置信度
            
        Returns:
            标注后的图像
        """
        vis_image = image.copy()
        
        for tile_name, confidence, (x1, y1, x2, y2) in detections:
            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"{tile_name}"
            if show_conf:
                label += f" {confidence:.2f}"
            
            # 计算文本大小并绘制背景
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                vis_image,
                (x1, y1 - text_h - 4),
                (x1 + text_w, y1),
                (0, 255, 0),
                -1
            )
            
            # 绘制文本
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
        
        return vis_image


# 创建全局检测器实例（延迟初始化）
_global_detector: Optional[YOLOTileDetector] = None


def get_detector(model_path: Optional[str] = None) -> YOLOTileDetector:
    """
    获取全局YOLO检测器实例（单例模式）
    
    Args:
        model_path: 模型文件路径，如果不指定则使用默认路径
        
    Returns:
        YOLOTileDetector实例
    """
    global _global_detector
    
    if _global_detector is None:
        if model_path is None:
            # 使用默认路径（训练后的最佳模型）
            project_root = Path(__file__).parent.parent.parent
            model_path = project_root / "yolo_dataset" / "runs" / "tiles_yolov52" / "weights" / "best.pt"
        
        _global_detector = YOLOTileDetector(str(model_path))
    
    return _global_detector
