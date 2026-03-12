"""
麻将牌神经网络分类器（OpenCV ANN_MLP）

说明：
  - 模型文件默认: models/tile_ann.xml
  - 标签文件默认: models/tile_ann.labels.json
  - 用于在模板匹配之外提供一条神经网络分类通路
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger


class TileNNClassifier:
    """基于 OpenCV ANN_MLP 的牌型分类器"""

    def __init__(
        self,
        model_path: str = "models/tile_ann.xml",
        labels_path: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.labels_path = (
            Path(labels_path) if labels_path else self.model_path.with_suffix(".labels.json")
        )

        self.model: Optional[cv2.ml.ANN_MLP] = None
        self.labels: List[str] = []
        self.input_w: int = 32
        self.input_h: int = 48

        self._load()

    def _load(self):
        """加载模型与标签映射"""
        if not self.model_path.exists() or not self.labels_path.exists():
            logger.info(
                f"神经网络模型未找到（{self.model_path} / {self.labels_path}），"
                "将仅使用模板匹配"
            )
            return

        try:
            self.model = cv2.ml.ANN_MLP_load(str(self.model_path))
            with open(self.labels_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.labels = list(meta.get("labels", []))
            self.input_w = int(meta.get("input_w", 32))
            self.input_h = int(meta.get("input_h", 48))

            if not self.labels:
                logger.warning("NN 标签映射为空，禁用 NN 推理")
                self.model = None
                return

            logger.info(
                f"已加载牌型神经网络: {self.model_path} "
                f"(classes={len(self.labels)}, input={self.input_w}x{self.input_h})"
            )
        except Exception as e:
            logger.warning(f"加载牌型神经网络失败: {e}，将回退模板匹配")
            self.model = None

    def available(self) -> bool:
        return self.model is not None and len(self.labels) > 0

    @staticmethod
    def _preprocess(tile_img: np.ndarray, input_w: int, input_h: int) -> np.ndarray:
        """
        预处理：灰度 + 尺寸归一 + CLAHE + 归一化
        """
        if tile_img.ndim == 3:
            gray = cv2.cvtColor(tile_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = tile_img

        resized = cv2.resize(gray, (input_w, input_h), interpolation=cv2.INTER_AREA)
        blur = cv2.GaussianBlur(resized, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        norm = clahe.apply(blur)

        feat = norm.astype(np.float32) / 255.0
        return feat.reshape(1, -1)

    def predict(
        self,
        tile_img: np.ndarray,
        top_k: int = 3,
    ) -> Tuple[Optional[str], float, List[Tuple[str, float]], Dict[str, float]]:
        """
        预测牌型。

        Returns:
            (best_name, best_prob, top_candidates, all_scores)
        """
        if not self.available() or tile_img.size == 0:
            return None, 0.0, [], {}

        sample = self._preprocess(tile_img, self.input_w, self.input_h)

        try:
            _ret, raw = self.model.predict(sample)
            logits = raw[0].astype(np.float32)

            n = min(len(self.labels), logits.shape[0])
            logits = logits[:n]
            labels = self.labels[:n]

            logits = logits - np.max(logits)
            exp = np.exp(logits)
            probs = exp / (np.sum(exp) + 1e-8)

            score_map: Dict[str, float] = {
                labels[i]: float(probs[i]) for i in range(n)
            }
            sorted_scores = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
            top_candidates = sorted_scores[: max(1, top_k)]

            best_name, best_prob = top_candidates[0]
            return best_name, float(best_prob), top_candidates, score_map
        except Exception as e:
            logger.debug(f"NN 推理失败: {e}")
            return None, 0.0, [], {}
