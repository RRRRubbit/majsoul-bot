"""
视觉识别模块
提供屏幕截图、牌面识别、游戏状态检测功能
"""
from .screen_capture import ScreenCapture
from .tile_recognizer import TileRecognizer, ALL_TILE_NAMES
from .game_state_detector import GameStateDetector, GamePhase, DetectedState
from .regions import ScreenRegions, HandRegion, DEFAULT_REGIONS

__all__ = [
    "ScreenCapture",
    "TileRecognizer",
    "ALL_TILE_NAMES",
    "GameStateDetector",
    "GamePhase",
    "DetectedState",
    "ScreenRegions",
    "HandRegion",
    "DEFAULT_REGIONS",
]
