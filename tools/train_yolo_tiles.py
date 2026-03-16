"""
YOLOv5 麻将牌识别模型训练脚本

使用方法：
1. 确保已安装依赖：pip install ultralytics
2. 运行训练：python tools/train_yolo_tiles.py
"""
from pathlib import Path
from ultralytics import YOLO

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "yolo_dataset"
DATA_YAML = DATASET_DIR / "tiles_classes.yaml"


def check_dataset():
    """检查数据集是否完整"""
    if not DATA_YAML.exists():
        print(f"❌ 未找到数据配置文件: {DATA_YAML}")
        return False
    
    train_img_dir = DATASET_DIR / "images" / "train"
    val_img_dir = DATASET_DIR / "images" / "val"
    
    if not train_img_dir.exists() or not val_img_dir.exists():
        print(f"❌ 数据集目录不完整")
        return False
    
    train_count = len(list(train_img_dir.glob("*.png")))
    val_count = len(list(val_img_dir.glob("*.png")))
    
    print(f"✅ 数据集检查通过:")
    print(f"   训练集: {train_count} 张")
    print(f"   验证集: {val_count} 张")
    return True


def train():
    """开始训练"""
    print("=" * 60)
    print("开始训练YOLOv5麻将牌识别模型")
    print("=" * 60)
    print()
    
    # 检查数据集
    if not check_dataset():
        return
    
    # 加载预训练模型
    print("\n📥 加载YOLOv5s预训练模型...")
    model = YOLO('yolov5s.pt')
    
    # 开始训练
    print(f"\n🚀 开始训练...")
    print(f"   数据配置: {DATA_YAML}")
    print(f"   输出目录: {DATASET_DIR}/runs/tiles_yolov5")
    print()
    
    try:
        results = model.train(
            data=str(DATA_YAML),
            epochs=100,              # 训练轮数
            imgsz=640,               # 图片大小
            batch=16,                # 批次大小（如果显存不足会自动调整）
            name='tiles_yolov5',     # 实验名称
            project=str(DATASET_DIR / "runs"),  # 输出目录
            cache=True,              # 缓存图片
            device='cpu',            # 使用CPU（如果有GPU会自动检测）
            patience=20,             # 早停耐心值
            save=True,               # 保存模型
            plots=True,              # 生成训练图表
        )
        
        print("\n" + "=" * 60)
        print("✅ 训练完成！")
        print("=" * 60)
        print(f"\n最佳模型保存在: {DATASET_DIR}/runs/tiles_yolov5/weights/best.pt")
        print(f"最新模型保存在: {DATASET_DIR}/runs/tiles_yolov5/weights/last.pt")
        print(f"\n查看训练结果:")
        print(f"  - 训练曲线: {DATASET_DIR}/runs/tiles_yolov5/results.png")
        print(f"  - 混淆矩阵: {DATASET_DIR}/runs/tiles_yolov5/confusion_matrix.png")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    train()
