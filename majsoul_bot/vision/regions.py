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


# 统一使用项目根目录作为路径起点
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION_PATH = PROJECT_ROOT / "config" / "vision_calibration.json"


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
    drawn_gap: float = 10.0
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
    button_scan_y: float = 0.62
    button_scan_w: float = 0.10
    button_scan_h: float = 0.10

    # 宝牌显示区域（右侧）
    dora_x: float = 0.04
    dora_y: float = 0.25
    dora_w: float = 0.30
    dora_h: float = 0.10

    # 牌堆检测区域（中部偏下）
    wall_x: float = 0.06
    wall_y: float = 0.52
    wall_w: float = 0.88
    wall_h: float = 0.14

    # 四家副露（吃/碰/杠）检测区域
    meld_self_x: float = 0.18
    meld_self_y: float = 0.68
    meld_self_w: float = 0.64
    meld_self_h: float = 0.14

    meld_right_x: float = 0.82
    meld_right_y: float = 0.20
    meld_right_w: float = 0.15
    meld_right_h: float = 0.34

    meld_opposite_x: float = 0.18
    meld_opposite_y: float = 0.08
    meld_opposite_w: float = 0.64
    meld_opposite_h: float = 0.14

    meld_left_x: float = 0.03
    meld_left_y: float = 0.90
    meld_left_w: float = 0.15
    meld_left_h: float = 0.34

    # 中央信息区域（局/本/供托）
    info_x: float = 0.38
    info_y: float = 0.40
    info_w: float = 0.24
    info_h: float = 0.20

    # ── 扫描式识别参数 ──
    # 手牌 x 边界（归一化）：x > hand_x_max 的牌属于副露区，排除在手牌扫描外
    hand_x_max: float = 0.74
    # 摸牌间距倍数：相邻牌之间的像素间隔 > drawn_x_gap_ratio × tile_width 时，
    # 认为该处存在"摸牌间隔"（分隔手牌和摸牌的较大空隙）
    drawn_x_gap_ratio: float = 1.2

    # ── 当前出牌家指示器（中心黄色标识）──
    indicator_center_x: float = 0.50  # 中心点x
    indicator_center_y: float = 0.50  # 中心点y
    indicator_radius: float = 0.08    # 检测区域半径
    indicator_sample_offset: float = 0.10  # 采样点距离中心的偏移

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

    def get_named_rect(self, name: str) -> Tuple[float, float, float, float]:
        """按名称获取归一化矩形区域。"""
        mapping = {
            "dora": (self.dora_x, self.dora_y, self.dora_w, self.dora_h),
            "wall": (self.wall_x, self.wall_y, self.wall_w, self.wall_h),
            "meld_self": (self.meld_self_x, self.meld_self_y, self.meld_self_w, self.meld_self_h),
            "meld_right": (self.meld_right_x, self.meld_right_y, self.meld_right_w, self.meld_right_h),
            "meld_opposite": (self.meld_opposite_x, self.meld_opposite_y, self.meld_opposite_w, self.meld_opposite_h),
            "meld_left": (self.meld_left_x, self.meld_left_y, self.meld_left_w, self.meld_left_h),
        }
        if name not in mapping:
            raise KeyError(f"未知区域名: {name}")
        return mapping[name]

    def set_named_rect(self, name: str, x: float, y: float, w: float, h: float):
        """按名称设置归一化矩形区域。"""
        if name == "dora":
            self.dora_x, self.dora_y, self.dora_w, self.dora_h = x, y, w, h
        elif name == "wall":
            self.wall_x, self.wall_y, self.wall_w, self.wall_h = x, y, w, h
        elif name == "meld_self":
            self.meld_self_x, self.meld_self_y, self.meld_self_w, self.meld_self_h = x, y, w, h
        elif name == "meld_right":
            self.meld_right_x, self.meld_right_y, self.meld_right_w, self.meld_right_h = x, y, w, h
        elif name == "meld_opposite":
            self.meld_opposite_x, self.meld_opposite_y, self.meld_opposite_w, self.meld_opposite_h = x, y, w, h
        elif name == "meld_left":
            self.meld_left_x, self.meld_left_y, self.meld_left_w, self.meld_left_h = x, y, w, h
        else:
            raise KeyError(f"未知区域名: {name}")

    vision_calibration_path: str = str(DEFAULT_CALIBRATION_PATH)

    @classmethod
    def load_from_json(cls, path: str = vision_calibration_path) -> "ScreenRegions":
        """
        从 JSON 校准文件加载区域配置

        Args:
            path: 校准文件路径

        Returns:
            ScreenRegions 实例
        """
        regions = cls()
        config_path = Path(path) if path is not None else cls.vision_calibration_path

        if not config_path.exists():
            logger.debug(f"校准文件不存在: {config_path}，使用默认区域配置")
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

            for key in (
                "dora_x", "dora_y", "dora_w", "dora_h",
                "wall_x", "wall_y", "wall_w", "wall_h",
                "meld_self_x", "meld_self_y", "meld_self_w", "meld_self_h",
                "meld_right_x", "meld_right_y", "meld_right_w", "meld_right_h",
                "meld_opposite_x", "meld_opposite_y", "meld_opposite_w", "meld_opposite_h",
                "meld_left_x", "meld_left_y", "meld_left_w", "meld_left_h",
                "hand_x_max", "drawn_x_gap_ratio",
            ):
                if key in data:
                    setattr(regions, key, data[key])

            logger.info(f"已加载视觉校准配置: {config_path}")
        except Exception as e:
            logger.warning(f"加载校准文件失败: {e}，使用默认值")

        return regions

    def save_to_json(self, path: str | Path | None = None):
        """保存区域配置到 JSON 文件"""
        config_path = Path(path) if path is not None else Path(self.vision_calibration_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "hand_x_start": self.hand.x_start,
            "hand_y_start": self.hand.y_start,
            "hand_tile_width": self.hand.tile_width,
            "hand_tile_height": self.hand.tile_height,
            "drawn_gap": self.hand.drawn_gap,
            "dora_x": self.dora_x,
            "dora_y": self.dora_y,
            "dora_w": self.dora_w,
            "dora_h": self.dora_h,
            "wall_x": self.wall_x,
            "wall_y": self.wall_y,
            "wall_w": self.wall_w,
            "wall_h": self.wall_h,
            "meld_self_x": self.meld_self_x,
            "meld_self_y": self.meld_self_y,
            "meld_self_w": self.meld_self_w,
            "meld_self_h": self.meld_self_h,
            "meld_right_x": self.meld_right_x,
            "meld_right_y": self.meld_right_y,
            "meld_right_w": self.meld_right_w,
            "meld_right_h": self.meld_right_h,
            "meld_opposite_x": self.meld_opposite_x,
            "meld_opposite_y": self.meld_opposite_y,
            "meld_opposite_w": self.meld_opposite_w,
            "meld_opposite_h": self.meld_opposite_h,
            "meld_left_x": self.meld_left_x,
            "meld_left_y": self.meld_left_y,
            "meld_left_w": self.meld_left_w,
            "meld_left_h": self.meld_left_h,
            "hand_x_max": self.hand_x_max,
            "drawn_x_gap_ratio": self.drawn_x_gap_ratio,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"区域配置已保存: {config_path}")


# 默认区域配置（单例，程序启动时加载校准文件）
DEFAULT_REGIONS = ScreenRegions.load_from_json()
