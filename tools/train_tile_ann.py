"""
训练麻将牌神经网络分类模型（OpenCV ANN_MLP）。

用途：
  - 从已标注牌图训练 NN 分类器
  - 导出 models/tile_ann.xml 与 labels 映射
  - 供 majsoul_bot.vision.tile_nn_classifier.TileNNClassifier 加载

支持的数据组织：
  1) 文件名包含牌名（推荐）
       templates/tiles/1m.png
       datasets/tiles/1m_0001.png

  2) 按牌名分子目录
       datasets/tiles/1m/xxx.png
       datasets/tiles/7z/xxx.png

注意：
  - 0m/0p/0s（赤宝牌）会自动映射到 5m/5p/5s，
    与当前项目 34 类（ALL_TILE_NAMES）保持兼容。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


# 与 majsoul_bot.vision.tile_recognizer 保持一致的 34 类牌名
ALL_TILE_NAMES: List[str] = (
    [f"{i}m" for i in range(1, 10)]
    + [f"{i}p" for i in range(1, 10)]
    + [f"{i}s" for i in range(1, 10)]
    + [f"{i}z" for i in range(1, 8)]
)


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
RED_DORA_MAP = {
    "0m": "5m",
    "0p": "5p",
    "0s": "5s",
}


def _normalize_label(raw: str) -> Optional[str]:
    name = raw.strip().lower()
    if name in RED_DORA_MAP:
        return RED_DORA_MAP[name]
    if name in ALL_TILE_NAMES:
        return name
    return None


def _infer_label_from_path(path: Path) -> Optional[str]:
    """从文件路径推断牌名标签。"""
    # 优先使用父目录
    parent_label = _normalize_label(path.parent.name)
    if parent_label:
        return parent_label

    stem = path.stem.lower()

    # 尝试按分隔符切 token
    tokens = [tok for tok in re.split(r"[^0-9a-z]+", stem) if tok]
    for tok in tokens:
        label = _normalize_label(tok)
        if label:
            return label
        if len(tok) >= 2:
            label = _normalize_label(tok[:2])
            if label:
                return label

    # 最后尝试全文匹配
    m = re.search(r"(0[psm]|[1-9][psm]|[1-7]z)", stem)
    if m:
        return _normalize_label(m.group(1))

    return None


def _preprocess(img: np.ndarray, input_w: int, input_h: int) -> np.ndarray:
    """与 TileNNClassifier 保持一致的预处理。"""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    resized = cv2.resize(gray, (input_w, input_h), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(resized, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(blur)

    feat = norm.astype(np.float32) / 255.0
    return feat.reshape(-1)


def _augment_image(img: np.ndarray) -> List[np.ndarray]:
    """生成一组轻量增强样本，提升鲁棒性。"""
    variants = [img]
    h, w = img.shape[:2]

    # 平移增强
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(
            img,
            m,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        variants.append(shifted)

    # 亮度/对比度增强
    for alpha, beta in ((0.90, 0), (1.10, 0), (1.0, -12), (1.0, 12)):
        variants.append(cv2.convertScaleAbs(img, alpha=alpha, beta=beta))

    return variants


def _collect_samples(data_dir: Path) -> Tuple[List[np.ndarray], List[str], Counter]:
    imgs: List[np.ndarray] = []
    labels: List[str] = []
    dist: Counter = Counter()

    for p in sorted(data_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue

        label = _infer_label_from_path(p)
        if not label:
            continue

        img = cv2.imread(str(p))
        if img is None or img.size == 0:
            continue

        imgs.append(img)
        labels.append(label)
        dist[label] += 1

    return imgs, labels, dist


def _build_features(
    imgs: Sequence[np.ndarray],
    labels: Sequence[str],
    classes: Sequence[str],
    input_w: int,
    input_h: int,
    use_augment: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    label_to_idx = {name: i for i, name in enumerate(classes)}

    x_rows: List[np.ndarray] = []
    y_idx_rows: List[int] = []

    for img, label in zip(imgs, labels):
        if label not in label_to_idx:
            continue

        variants = _augment_image(img) if use_augment else [img]
        cls_idx = label_to_idx[label]

        for v in variants:
            x_rows.append(_preprocess(v, input_w, input_h))
            y_idx_rows.append(cls_idx)

    x = np.asarray(x_rows, dtype=np.float32)
    y_idx = np.asarray(y_idx_rows, dtype=np.int32)

    y = np.zeros((len(y_idx), len(classes)), dtype=np.float32)
    if len(y_idx) > 0:
        y[np.arange(len(y_idx)), y_idx] = 1.0

    return x, y, y_idx


def _stratified_split(
    y_idx: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    by_class: Dict[int, List[int]] = defaultdict(list)

    for i, cls in enumerate(y_idx.tolist()):
        by_class[int(cls)].append(i)

    train_idx: List[int] = []
    val_idx: List[int] = []

    for cls_indices in by_class.values():
        idx = np.asarray(cls_indices, dtype=np.int32)
        rng.shuffle(idx)

        if val_ratio > 0 and len(idx) >= 5:
            n_val = int(round(len(idx) * val_ratio))
            n_val = max(1, min(n_val, len(idx) - 1))
        else:
            n_val = 0

        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.asarray(train_idx, dtype=np.int32), np.asarray(val_idx, dtype=np.int32)


def _calc_accuracy(model: cv2.ml.ANN_MLP, x: np.ndarray, y_idx: np.ndarray) -> float:
    if x.size == 0 or y_idx.size == 0:
        return float("nan")
    _ret, raw = model.predict(x)
    pred = np.argmax(raw, axis=1)
    return float(np.mean(pred == y_idx))


def train_ann(
    x_train: np.ndarray,
    y_train: np.ndarray,
    class_count: int,
    hidden1: int,
    hidden2: int,
    max_iter: int,
    epsilon: float,
) -> cv2.ml.ANN_MLP:
    feature_dim = int(x_train.shape[1])
    layer_sizes = np.array([feature_dim, hidden1, hidden2, class_count], dtype=np.int32)

    ann = cv2.ml.ANN_MLP_create()
    ann.setLayerSizes(layer_sizes)
    ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 1.0, 1.0)
    ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.001, 0.1)
    ann.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, max_iter, epsilon))

    ok = ann.train(x_train, cv2.ml.ROW_SAMPLE, y_train)
    if not ok:
        raise RuntimeError("ANN_MLP 训练失败")
    return ann


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="训练麻将牌 ANN_MLP 模型")
    p.add_argument("--data-dir", default="templates/tiles", help="训练图片目录（支持递归）")
    p.add_argument("--output-model", default="models/tile_ann.xml", help="模型输出路径")
    p.add_argument("--output-labels", default=None, help="标签映射输出路径（默认同模型名 .labels.json）")

    p.add_argument("--input-w", type=int, default=32, help="输入宽度")
    p.add_argument("--input-h", type=int, default=48, help="输入高度")
    p.add_argument("--hidden1", type=int, default=256, help="第一隐藏层大小")
    p.add_argument("--hidden2", type=int, default=128, help="第二隐藏层大小")

    p.add_argument("--max-iter", type=int, default=1000, help="最大迭代次数")
    p.add_argument("--epsilon", type=float, default=1e-4, help="收敛阈值")
    p.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例（小数据集会自动降为 0）")
    p.add_argument("--seed", type=int, default=42, help="随机种子")

    p.add_argument("--no-augment", action="store_true", help="关闭数据增强")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_model = Path(args.output_model)
    output_labels = Path(args.output_labels) if args.output_labels else output_model.with_suffix(".labels.json")

    if not data_dir.exists():
        print(f"[ERROR] 数据目录不存在: {data_dir}")
        return 1

    raw_imgs, raw_labels, raw_dist = _collect_samples(data_dir)
    if not raw_imgs:
        print(f"[ERROR] 在 {data_dir} 未找到可用标注图像")
        return 1

    classes = [name for name in ALL_TILE_NAMES if name in raw_dist]
    if len(classes) < 2:
        print(
            "[ERROR] 有效类别不足（<2）。请至少提供两个不同牌型样本，"
            "并确保命名符合 1m~9m / 1p~9p / 1s~9s / 1z~7z"
        )
        return 1

    x, y, y_idx = _build_features(
        raw_imgs,
        raw_labels,
        classes,
        input_w=int(args.input_w),
        input_h=int(args.input_h),
        use_augment=not args.no_augment,
    )

    train_idx, val_idx = _stratified_split(y_idx, val_ratio=float(args.val_ratio), seed=int(args.seed))
    x_train, y_train, y_idx_train = x[train_idx], y[train_idx], y_idx[train_idx]
    x_val, y_idx_val = x[val_idx], y_idx[val_idx]

    print("=" * 72)
    print("🧠 训练麻将牌 ANN_MLP 模型")
    print("=" * 72)
    print(f"数据目录: {data_dir.resolve()}")
    print(f"类别数: {len(classes)}")
    print(f"原始样本数: {len(raw_imgs)} | 增强后样本数: {len(x)}")
    print(f"训练集: {len(x_train)} | 验证集: {len(x_val)}")
    print("类别分布（原始样本）:")
    for name in classes:
        print(f"  - {name}: {raw_dist[name]}")

    missing = [name for name in ALL_TILE_NAMES if name not in raw_dist]
    if missing:
        print(f"[WARN] 缺失类别 {len(missing)} 种（模型不会识别这些牌）")

    try:
        model = train_ann(
            x_train=x_train,
            y_train=y_train,
            class_count=len(classes),
            hidden1=int(args.hidden1),
            hidden2=int(args.hidden2),
            max_iter=int(args.max_iter),
            epsilon=float(args.epsilon),
        )
    except Exception as e:
        print(f"[ERROR] 训练失败: {e}")
        return 1

    train_acc = _calc_accuracy(model, x_train, y_idx_train)
    val_acc = _calc_accuracy(model, x_val, y_idx_val) if len(x_val) else float("nan")

    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_labels.parent.mkdir(parents=True, exist_ok=True)

    model.save(str(output_model))

    meta = {
        "labels": classes,
        "input_w": int(args.input_w),
        "input_h": int(args.input_h),
        "feature_dim": int(x.shape[1]),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "samples": {
            "raw": int(len(raw_imgs)),
            "augmented": int(len(x)),
            "train": int(len(x_train)),
            "val": int(len(x_val)),
        },
        "metrics": {
            "train_acc": float(train_acc),
            "val_acc": None if np.isnan(val_acc) else float(val_acc),
        },
    }

    with open(output_labels, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("-" * 72)
    print(f"✅ 模型已保存: {output_model.resolve()}")
    print(f"✅ 标签已保存: {output_labels.resolve()}")
    print(f"训练集准确率: {train_acc:.4f}")
    if np.isnan(val_acc):
        print("验证集准确率: N/A（样本较少，未切分验证集）")
    else:
        print(f"验证集准确率: {val_acc:.4f}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
