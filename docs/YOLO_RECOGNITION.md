# YOLO麻将牌识别系统

## 概述

本项目已集成YOLOv5模型用于麻将牌识别，相比传统的模板匹配方法，YOLO具有以下优势：
- **更高准确率**: 训练后的模型mAP@50达到99.3%
- **更快速度**: 直接检测，无需逐张遍历模板
- **更好泛化**: 对光照、角度变化更鲁棒
- **支持赤宝牌**: 完整支持37类麻将牌（含0m/0p/0s赤宝牌）

## 模型信息

- **模型类型**: YOLOv5 (ultralytics)
- **训练数据**: 489张标注图像（400训练集，89验证集）
- **类别数量**: 37类
  - 万子: 0m, 1m-9m (10类)
  - 筒子: 0p, 1p-9p (10类)
  - 索子: 0s, 1s-9s (10类)
  - 字牌: 1z-7z (7类)
- **性能指标**:
  - mAP@50: 99.3%
  - mAP@50-95: 99.2%
  - 训练轮数: 49 epochs (early stop at 29)

## 使用方法

### 1. 在TileRecognizer中使用 (推荐)

```python
from majsoul_bot.vision.tile_recognizer import TileRecognizer

# 创建识别器，启用YOLO
recognizer = TileRecognizer(
    yolo_enabled=True,        # 启用YOLO检测
    yolo_priority=True,       # YOLO优先模式
    yolo_conf_threshold=0.5,  # 置信度阈值
)

# 识别手牌
screenshot = capture_screenshot()
results = recognizer.recognize_hand_by_yolo(screenshot, has_drawn_tile=True)

# 结果格式: [(tile_name, (center_x, center_y)), ...]
for tile_name, (x, y) in results:
    print(f"{tile_name} at ({x}, {y})")
```

### 2. 直接使用YOLO检测器

```python
from majsoul_bot.vision.yolo_tile_detector import get_detector

# 获取检测器实例 (单例模式)
detector = get_detector()

# 检测手牌区域
hand_region = screenshot[y1:y2, x1:x2]
detections = detector.detect_tiles(hand_region)

# 结果格式: [(tile_name, confidence, (x1, y1, x2, y2)), ...]
for tile_name, conf, (x1, y1, x2, y2) in detections:
    print(f"{tile_name}: {conf:.3f} at bbox({x1},{y1},{x2},{y2})")
```

### 3. 与传统方法混合使用

```python
# 创建识别器，同时启用YOLO和模板匹配
recognizer = TileRecognizer(
    yolo_enabled=True,
    yolo_priority=True,      # YOLO优先
    nn_enabled=True,         # 也可以启用神经网络
    templates_dir="templates/tiles",
)

# 使用 recognize_hand_by_scan 会自动选择最佳方法
# 如果yolo_priority=True，会优先使用YOLO
results = recognizer.recognize_hand_by_scan(screenshot, has_drawn_tile=True)
```

## 测试脚本

### 运行所有测试
```bash
python tools/test_yolo_recognition.py
```

### 单独测试
```bash
# 测试1: YOLO检测器单独测试
python tools/test_yolo_recognition.py --test 1

# 测试2: TileRecognizer集成测试
python tools/test_yolo_recognition.py --test 2

# 测试3: YOLO vs 模板匹配对比
python tools/test_yolo_recognition.py --test 3

# 测试4: 从图像文件测试
python tools/test_yolo_recognition.py --test 4 --image path/to/screenshot.png
```

## 性能对比

### 识别速度
- **YOLO**: ~50-100ms (单次推理)
- **模板匹配**: ~200-500ms (13-14张牌)
- **神经网络**: ~150-300ms (13-14张牌)

### 识别准确率
- **YOLO**: 99.3% (基于验证集)
- **模板匹配**: 85-95% (取决于模板质量和光照)
- **神经网络**: 90-95% (取决于训练数据)

## 模型路径

默认模型路径: `yolo_dataset/runs/tiles_yolov52/weights/best.pt`

如需使用自定义模型:
```python
recognizer = TileRecognizer(
    yolo_model_path="path/to/your/model.pt"
)
```

## 训练新模型

如需重新训练或微调模型:
```bash
# 1. 准备数据集 (已有工具)
python tools/convert_templates_to_yolo.py

# 2. 训练模型
python tools/train_yolo_tiles.py
```

## 注意事项

1. **首次使用**: 首次加载YOLO模型时会需要下载ultralytics依赖，可能需要几分钟
2. **内存占用**: YOLO模型约占用200-300MB内存
3. **GPU加速**: 如果有NVIDIA GPU，会自动使用CUDA加速
4. **置信度阈值**: 默认0.5，可根据实际情况调整 (0.3-0.7)

## 故障排除

### 模型加载失败
```
错误: 模型文件不存在
解决: 确保已运行训练脚本，或检查模型路径是否正确
```

### 识别结果为空
```
错误: YOLO未检测到任何牌
解决: 
1. 检查手牌区域裁剪是否正确
2. 降低置信度阈值 (yolo_conf_threshold=0.3)
3. 检查截图质量和分辨率
```

### 识别错误率高
```
解决:
1. 使用更多样本重新训练模型
2. 调整置信度阈值
3. 启用花色顺序约束 (enforce_mpsz_order=True)
```

## 后续改进计划

- [ ] 支持副露牌识别（吃/碰/杠）
- [ ] 增加对其他玩家手牌的识别
- [ ] 优化模型以支持移动端部署
- [ ] 添加在线学习功能，根据识别反馈持续改进
