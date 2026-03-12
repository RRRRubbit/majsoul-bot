"""
测试 TileRecognizer 候选牌型输出能力
"""

import numpy as np

from majsoul_bot.vision.tile_recognizer import TileRecognizer


def test_recognize_tile_with_candidates_returns_topk():
    """应返回 Top-K 候选并按得分降序排列"""
    recognizer = TileRecognizer(templates_dir="templates/tiles", threshold=0.75)
    # 直接使用已有模板作为输入，确保可匹配
    tile_img = recognizer.templates["1m"]

    best_name, best_score, candidates = recognizer.recognize_tile_with_candidates(
        tile_img,
        top_k=3,
    )

    assert best_name is not None
    assert best_score >= 0.0
    assert 0 < len(candidates) <= 3
    # 分数应按降序
    assert all(candidates[i][1] >= candidates[i + 1][1] for i in range(len(candidates) - 1))


def test_recognize_hand_populates_last_recognition_details():
    """recognize_hand 后应写入 last_recognition_details"""
    recognizer = TileRecognizer(templates_dir="templates/tiles", threshold=0.75)

    # 用模板拼一张“截图”，手动设置 hand_count=1，避免依赖真实游戏截图
    fake_screenshot = recognizer.templates["1m"].copy()
    result = recognizer.recognize_hand(
        fake_screenshot,
        hand_count=1,
        has_drawn_tile=False,
    )

    assert len(result) >= 1
    assert len(recognizer.last_recognition_details) == len(result)

    first = recognizer.last_recognition_details[0]
    assert "recognized_name" in first
    assert "best_score" in first
    assert "candidates" in first
    assert isinstance(first["candidates"], list)


class _DummyNNClassifier:
    """测试用假 NN 分类器"""

    def __init__(self, score_map):
        self._score_map = dict(score_map)

    def available(self):
        return True

    def predict(self, _tile_img, top_k=3):
        ranked = sorted(self._score_map.items(), key=lambda x: x[1], reverse=True)
        top = ranked[: max(1, top_k)]
        best_name, best_prob = top[0]
        return best_name, float(best_prob), top, dict(self._score_map)


def test_recognize_tile_with_candidates_supports_nn_fusion():
    """应支持模板分与 NN 概率融合排序"""
    recognizer = TileRecognizer(
        templates_dir="templates/tiles",
        threshold=0.75,
        nn_enabled=False,
    )

    # 构造两张差异较大的模板图
    tile_img = np.zeros((48, 32, 3), dtype=np.uint8)
    tile_img[8:40, 6:26] = (255, 255, 255)

    template_a = tile_img.copy()
    template_b = np.full((48, 32, 3), 30, dtype=np.uint8)

    recognizer.templates = {
        "1m": template_a,
        "2m": template_b,
    }

    # 让 NN 强力偏向 2m，验证融合后结果可被 NN 拉动
    recognizer.nn_classifier = _DummyNNClassifier({"1m": 0.05, "2m": 0.95})
    recognizer.nn_fusion_weight = 0.9

    best_name, best_score, candidates = recognizer.recognize_tile_with_candidates(tile_img, top_k=2)

    assert best_name == "2m"
    assert best_score > 0.5
    assert len(candidates) == 2
    assert candidates[0][0] == "2m"


def test_recognize_tile_can_fallback_to_nn_confidence_when_threshold_not_met():
    """融合分未过阈值时，应允许高置信 NN 兜底通过"""
    recognizer = TileRecognizer(
        templates_dir="templates/tiles",
        threshold=0.95,
        nn_enabled=False,
        nn_min_confidence=0.60,
    )

    # 清空模板，模拟纯 NN 场景
    recognizer.templates = {}
    recognizer.nn_classifier = _DummyNNClassifier({"3p": 0.82, "4p": 0.10})

    tile_img = np.full((48, 32, 3), 120, dtype=np.uint8)
    name, score = recognizer.recognize_tile(tile_img)

    assert name == "3p"
    assert score == 0.82
