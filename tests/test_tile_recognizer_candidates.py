"""
测试 TileRecognizer 候选牌型输出能力
"""

import numpy as np

from majsoul_bot.vision.tile_recognizer import ALL_TILE_NAMES, TileRecognizer


def test_recognize_tile_with_candidates_returns_topk():
    """应返回 Top-K 候选并按得分降序排列"""
    recognizer = TileRecognizer(templates_dir="templates/tiles", threshold=0.75)
    assert recognizer.templates, "测试需要至少一个模板样本"
    # 直接使用已有模板作为输入，确保可匹配
    first_name = next(iter(recognizer.templates.keys()))
    tile_img = recognizer.templates[first_name]

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
    assert recognizer.templates, "测试需要至少一个模板样本"

    # 用模板拼一张“截图”，手动设置 hand_count=1，避免依赖真实游戏截图
    fake_screenshot = recognizer.templates[next(iter(recognizer.templates.keys()))].copy()
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

    # 校准后的 best_score 应不低于原始融合分
    meta = recognizer._last_single_recognition_meta
    assert best_score >= float(meta["raw_best_score"])
    assert 0.0 <= best_score <= 1.0


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


def test_recognize_hand_nn_priority_forces_nn_accept():
    """启用 nn_priority 时，即使低于阈值也应优先采用 NN 结果。"""
    recognizer = TileRecognizer(
        templates_dir="templates/tiles",
        threshold=0.95,
        nn_enabled=False,
        nn_priority=True,
        nn_min_confidence=0.90,
    )

    recognizer.templates = {}
    recognizer.template_samples = {}
    recognizer.nn_classifier = _DummyNNClassifier({"4s": 0.40, "5s": 0.20})

    tile_img = np.full((48, 32, 3), 100, dtype=np.uint8)
    hand = recognizer.recognize_hand(tile_img, hand_count=1, has_drawn_tile=False)

    assert len(hand) == 1
    assert hand[0][0] == "4s"
    detail = recognizer.last_recognition_details[0]
    assert detail["accepted"] is True
    assert detail["accept_reason"] == "nn_priority"


def test_mpsz_order_constraint_can_fix_suit_order():
    """m->p->s->z 约束应能修正逆序花色识别。"""
    recognizer = TileRecognizer(
        templates_dir="templates/tiles",
        nn_enabled=False,
        enforce_mpsz_order=True,
    )

    details = [
        {
            "index": 0,
            "is_drawn": False,
            "accepted": True,
            "best_name": "4p",
            "recognized_name": "4p",
            "best_score": 0.78,
            "candidates": [("4p", 0.78), ("3m", 0.76)],
            "center": (100, 100),
        },
        {
            "index": 1,
            "is_drawn": False,
            "accepted": True,
            "best_name": "2m",
            "recognized_name": "2m",
            "best_score": 0.79,
            "candidates": [("2m", 0.79), ("6p", 0.77)],
            "center": (140, 100),
        },
    ]

    recognizer._apply_mpsz_order_constraint(details, hand_count=13)

    # 原始是 p -> m（逆序），期望被修正为非递减顺序（m<=p<=s<=z）
    suit_rank = {"m": 0, "p": 1, "s": 2, "z": 3}
    s0 = details[0]["recognized_name"][-1]
    s1 = details[1]["recognized_name"][-1]
    assert suit_rank[s0] <= suit_rank[s1]
    # 至少发生一处修正
    assert (
        details[0]["recognized_name"] != "4p"
        or details[1]["recognized_name"] != "2m"
    )
    assert details[0]["accept_reason"] == "mpsz_order_constraint"


def test_mpsz_order_constraint_avoids_cross_suit_insertion_inside_block():
    """应修正类似 6m,2s,8m 的跨花色错插，保持同花色区块连续。"""
    recognizer = TileRecognizer(
        templates_dir="templates/tiles",
        nn_enabled=False,
        enforce_mpsz_order=True,
    )

    details = [
        {
            "index": 0,
            "is_drawn": False,
            "accepted": True,
            "best_name": "6m",
            "recognized_name": "6m",
            "best_score": 0.90,
            "candidates": [("6m", 0.90)],
            "center": (100, 100),
        },
        {
            "index": 1,
            "is_drawn": False,
            "accepted": True,
            "best_name": "2s",
            "recognized_name": "2s",
            "best_score": 0.84,
            "candidates": [("2s", 0.84), ("7m", 0.83)],
            "center": (140, 100),
        },
        {
            "index": 2,
            "is_drawn": False,
            "accepted": True,
            "best_name": "8m",
            "recognized_name": "8m",
            "best_score": 0.88,
            "candidates": [("8m", 0.88)],
            "center": (180, 100),
        },
    ]

    recognizer._apply_mpsz_order_constraint(details, hand_count=13)

    assert details[1]["recognized_name"].endswith("m")
    assert details[1]["accept_reason"] == "mpsz_order_constraint"


def test_all_tile_names_contains_red_dora_labels():
    """识别标签集合应包含红宝牌类别。"""
    assert "0m" in ALL_TILE_NAMES
    assert "0p" in ALL_TILE_NAMES
    assert "0s" in ALL_TILE_NAMES
