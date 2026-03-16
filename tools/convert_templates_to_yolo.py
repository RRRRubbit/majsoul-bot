"""
将模板图片转换为YOLO训练数据集

功能：
1. 扫描 templates/tiles/ 和 templates/buttons/ 目录
2. 为每张图片生成YOLO格式标注
3. 复制图片到 yolo_dataset/images/
4. 生成对应的 .txt 标注文件到 yolo_dataset/labels/
"""
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import random


class YOLODatasetConverter:
    """YOLO数据集转换器"""
    
    # 类别名称到ID的映射（与classes.yaml保持一致）
    CLASS_MAPPING = {
        # 万子
        '1m': 0, '2m': 1, '3m': 2, '4m': 3, '5m': 4,
        '6m': 5, '7m': 6, '8m': 7, '9m': 8,
        # 饼子
        '1p': 9, '2p': 10, '3p': 11, '4p': 12, '5p': 13,
        '6p': 14, '7p': 15, '8p': 16, '9p': 17,
        # 索子
        '1s': 18, '2s': 19, '3s': 20, '4s': 21, '5s': 22,
        '6s': 23, '7s': 24, '8s': 25, '9s': 26,
        # 字牌
        '1z': 27, '2z': 28, '3z': 29, '4z': 30,
        '5z': 31, '6z': 32, '7z': 33,
        # 赤宝牌
        '0m': 34, '0p': 35, '0s': 36,
        # 按钮
        'chi': 37, 'pon': 38, 'kan': 39, 'riichi': 40,
        'tsumo': 41, 'ron': 42, 'skip': 43,
    }
    
    def __init__(
        self,
        templates_root: str = "templates",
        output_root: str = "yolo_dataset",
        train_ratio: float = 0.8,
    ):
        """
        Args:
            templates_root: 模板目录根路径
            output_root: YOLO数据集输出目录
            train_ratio: 训练集比例（0-1）
        """
        self.templates_root = Path(templates_root)
        self.output_root = Path(output_root)
        self.train_ratio = train_ratio
        
        self.stats = {
            'total': 0,
            'train': 0,
            'val': 0,
            'by_class': {},
        }
    
    def convert(self):
        """执行转换"""
        print("=" * 60)
        print("开始转换模板图片为YOLO数据集")
        print("=" * 60)
        
        # 1. 清空输出目录
        self._clean_output_dirs()
        
        # 2. 扫描并转换麻将牌模板
        print("\n[1/3] 转换麻将牌模板...")
        tiles_dir = self.templates_root / "tiles"
        if tiles_dir.exists():
            self._convert_directory(tiles_dir, is_tile=True)
        else:
            print(f"  ⚠ 未找到麻将牌目录: {tiles_dir}")
        
        # 3. 扫描并转换按钮模板
        print("\n[2/3] 转换按钮模板...")
        buttons_dir = self.templates_root / "buttons"
        if buttons_dir.exists():
            self._convert_directory(buttons_dir, is_tile=False)
        else:
            print(f"  ⚠ 未找到按钮目录: {buttons_dir}")
        
        # 4. 输出统计信息
        print("\n[3/3] 转换完成统计")
        self._print_stats()
    
    def _clean_output_dirs(self):
        """清空输出目录"""
        dirs_to_clean = [
            self.output_root / "images" / "train",
            self.output_root / "images" / "val",
            self.output_root / "labels" / "train",
            self.output_root / "labels" / "val",
        ]
        for d in dirs_to_clean:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
    
    def _convert_directory(self, source_dir: Path, is_tile: bool):
        """
        转换一个目录下的所有图片
        
        Args:
            source_dir: 源目录
            is_tile: 是否为麻将牌（True）或按钮（False）
        """
        if is_tile:
            # 麻将牌：每个子目录代表一个类别（如 1m/, 2m/）
            for class_dir in sorted(source_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                class_name = class_dir.name
                if class_name not in self.CLASS_MAPPING:
                    print(f"  ⚠ 跳过未知类别: {class_name}")
                    continue
                
                class_id = self.CLASS_MAPPING[class_name]
                image_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
                
                if not image_files:
                    print(f"  ⚠ {class_name} 目录为空")
                    continue
                
                print(f"  处理 {class_name}: {len(image_files)} 张图片")
                for img_path in image_files:
                    self._convert_image(img_path, class_id, class_name)
        else:
            # 按钮：直接在目录下，文件名即类别（如 chi.png）
            for img_path in source_dir.glob("*.png"):
                class_name = img_path.stem  # 去除扩展名
                if class_name not in self.CLASS_MAPPING:
                    print(f"  ⚠ 跳过未知按钮: {class_name}")
                    continue
                
                class_id = self.CLASS_MAPPING[class_name]
                print(f"  处理按钮 {class_name}")
                self._convert_image(img_path, class_id, class_name)
    
    def _convert_image(self, img_path: Path, class_id: int, class_name: str):
        """
        转换单张图片
        
        Args:
            img_path: 图片路径
            class_id: YOLO类别ID
            class_name: 类别名称
        """
        # 随机分配到训练集或验证集
        is_train = random.random() < self.train_ratio
        split = "train" if is_train else "val"
        
        # 生成唯一文件名（避免重名）
        # 格式：<class_name>_<原文件名>
        new_filename = f"{class_name}_{img_path.name}"
        
        # 读取图片计算实际内容bbox
        import cv2
        import numpy as np
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ⚠ 无法读取图片: {img_path}")
            return
        
        h, w = img.shape[:2]
        
        # 使用边缘检测来找到实际内容区域
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 使用Canny边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 形态学膨胀，连接附近的边缘
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # 找到边缘的边界
        coords = cv2.findNonZero(dilated)
        if coords is not None and len(coords) > 10:  # 至少有10个边缘点
            x, y, bbox_w, bbox_h = cv2.boundingRect(coords)
            
            # 添加边距（10%）
            margin_x = max(2, int(bbox_w * 0.10))
            margin_y = max(2, int(bbox_h * 0.10))
            x = max(0, x - margin_x)
            y = max(0, y - margin_y)
            bbox_w = min(w - x, bbox_w + 2 * margin_x)
            bbox_h = min(h - y, bbox_h + 2 * margin_y)
            
            # 确保bbox不会太小（至少占图片的20%）
            if bbox_w < w * 0.2 or bbox_h < h * 0.2:
                # 太小了，可能检测失败，使用80%的图片区域
                margin_x = int(w * 0.1)
                margin_y = int(h * 0.1)
                x, y = margin_x, margin_y
                bbox_w = w - 2 * margin_x
                bbox_h = h - 2 * margin_y
        else:
            # fallback：使用80%的图片区域（去掉边缘10%）
            margin_x = int(w * 0.1)
            margin_y = int(h * 0.1)
            x, y = margin_x, margin_y
            bbox_w = w - 2 * margin_x
            bbox_h = h - 2 * margin_y
        
        # 计算归一化的YOLO格式坐标
        x_center = (x + bbox_w / 2) / w
        y_center = (y + bbox_h / 2) / h
        norm_w = bbox_w / w
        norm_h = bbox_h / h
        
        # 确保坐标在有效范围内
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        norm_w = max(0.01, min(1.0, norm_w))
        norm_h = max(0.01, min(1.0, norm_h))
        
        # 复制图片
        img_dst = self.output_root / "images" / split / new_filename
        shutil.copy2(img_path, img_dst)
        
        # 生成YOLO标注文件
        # 格式：<class_id> <x_center> <y_center> <width> <height>
        label_dst = self.output_root / "labels" / split / f"{new_filename.rsplit('.', 1)[0]}.txt"
        with open(label_dst, 'w') as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
        
        # 更新统计
        self.stats['total'] += 1
        if is_train:
            self.stats['train'] += 1
        else:
            self.stats['val'] += 1
        
        if class_name not in self.stats['by_class']:
            self.stats['by_class'][class_name] = {'train': 0, 'val': 0}
        self.stats['by_class'][class_name][split] += 1
    
    def _print_stats(self):
        """打印转换统计信息"""
        print(f"\n总计转换: {self.stats['total']} 张图片")
        print(f"  训练集: {self.stats['train']} 张")
        print(f"  验证集: {self.stats['val']} 张")
        print(f"  比例: {self.train_ratio*100:.0f}% / {(1-self.train_ratio)*100:.0f}%")
        
        print("\n各类别统计:")
        for class_name in sorted(self.stats['by_class'].keys()):
            counts = self.stats['by_class'][class_name]
            total = counts['train'] + counts['val']
            print(f"  {class_name:6s}: {total:3d} 张 (训练:{counts['train']:3d}, 验证:{counts['val']:2d})")


def main():
    """主函数"""
    converter = YOLODatasetConverter(
        templates_root="templates",
        output_root="yolo_dataset",
        train_ratio=0.8,
    )
    converter.convert()
    
    print("\n" + "=" * 60)
    print("✅ 数据集转换完成！")
    print("=" * 60)
    print("\n下一步:")
    print("  1. 运行训练脚本: python tools/train_yolo.py")
    print("  2. 或查看数据集: ls yolo_dataset/images/train/")


if __name__ == "__main__":
    main()
