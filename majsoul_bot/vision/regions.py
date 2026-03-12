"""
屏幕区域定义模块
基于雀魂游戏画面定义各 UI 元素的坐标区域

所有坐标均为归一化相对坐标（0.0 ~ 1.0），相对于游戏窗口尺寸。
默认值基于 1920x1080 分辨率的雀魂客户端/浏览器。
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple
from loguru import logger


@dataclass
class HandRegion:
    """手牌区域定义"""
    # 13张手牌起始位置（左上角相对坐标）
    x_start: float = 0.240
    y_start: float = 0.847
    # 每张牌的相对尺寸
    tile_width: float = 0.036
    tile_height: float = 0.083
    # 摸牌与手牌末尾之间的相对间隔
    drawn_gap: float = 0.009
    # 手牌上限（不含摸牌）
    max_tiles: int = 13


@dataclass
class ScreenRegions:
    """
    屏幕区域总配置

    包含手牌、按钮、宝牌等 UI 元素的坐标区域定义。
    """
    # 手牌区域
    hand: HandRegion = field(default_factory=HandRegion)

    # 操作按钮扫描区域（碰/吃/杠/立直/自摸/荣和）
    button_scan_x: float = 0.25
    button_scan_y: float = 0.73
    button_scan_w: float = 0.50
    button_scan_h: float = 0.22

    # 宝牌显示区域（右侧）
    dora_x: float = 0.65
    dora_y: float = 0.04
    dora_w: float = 0.30
    dora_h: float = 0.10

    # 中央信息区域（局/本/供托）
    info_x: float = 0.38
    info_y: float = 0.40
    info_w: float = 0.24
    info_h: float = 0.20

    def get_tile_center(
        self,
        tile_index: int,
        hand_count: int = 13,
        is_drawn: bool = False
    ) -> Tuple[float, float]:
        """
        获取手牌中某张牌中心的相对坐标

        Args:
            tile_index: 牌的索引（0~12 为手牌，13 为摸牌）
            hand_count: 当前手牌数量（不含摸牌）
            is_drawn: 是否为摸到的牌

        Returns:
            (rel_x, rel_y): 归一化坐标
        """
        h = self.hand
        if is_drawn or tile_index >= hand_count:
            # 摸牌位于手牌末尾右侧，有一段间隔
            x = (h.x_start
                 + hand_count * h.tile_width
                 + h.drawn_gap
                 + h.tile_width / 2)
        else:
            x = h.x_start + tile_index * h.tile_width + h.tile_width / 2

        y = h.y_start + h.tile_height / 2
        return x, y

    def get_tile_rect(
        self,
        tile_index: int,
        hand_count: int = 13,
        is_drawn: bool = False
    ) -> Tuple[float, float, float, float]:
        """
        获取手牌中某张牌的矩形区域（归一化坐标）

        Returns:
            (x_start, y_start, width, height)
        """
        h = self.hand
        if is_drawn or tile_index >= hand_count:
            x = h.x_start + hand_count * h.tile_width + h.drawn_gap
        else:
            x = h.x_start + tile_index * h.tile_width
        return x, h.y_start, h.tile_width, h.tile_height

    @classmethod
    def load_from_json(cls, path: str = "config/vision_calibration.json") -> "ScreenRegions":
        """
        从 JSON 校准文件加载区域配置

        Args:
            path: 校准文件路径

        Returns:
            ScreenRegions 实例
        """
        regions = cls()
        config_path = Path(path)

        if not config_path.exists():
            logger.debug(f"校准文件不存在: {path}，使用默认区域配置")
            return regions

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            h = regions.hand
            if "hand_x_start" in data:
                h.x_start = data["hand_x_start"]
            if "hand_y_start" in data:
                h.y_start = data["hand_y_start"]
            if "hand_tile_width" in data:
                h.tile_width = data["hand_tile_width"]
            if "hand_tile_height" in data:
                h.tile_height = data["hand_tile_height"]
            if "drawn_gap" in data:
                h.drawn_gap = data["drawn_gap"]

            logger.info(f"已加载视觉校准配置: {path}")
        except Exception as e:
            logger.warning(f"加载校准文件失败: {e}，使用默认值")

        return regions

    def save_to_json(self, path: str = "config/vision_calibration.json"):
        """保存区域配置到 JSON 文件"""
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "hand_x_start": self.hand.x_start,
            "hand_y_start": self.hand.y_start,
            "hand_tile_width": self.hand.tile_width,
            "hand_tile_height": self.hand.tile_height,
            "drawn_gap": self.hand.drawn_gap,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"区域配置已保存: {path}")


# 默认区域配置（单例，程序启动时加载校准文件）
DEFAULT_REGIONS = ScreenRegions.load_from_json()
