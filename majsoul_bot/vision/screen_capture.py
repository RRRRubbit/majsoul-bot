"""
屏幕截图模块
负责捕获雀魂游戏窗口画面
"""
import time
from typing import Dict, Optional, Tuple

import cv2
import mss
import numpy as np
from loguru import logger

# 尝试导入 pygetwindow（Windows 专用窗口管理库）
try:
    import pygetwindow as gw
    HAS_PYGETWINDOW = True
except ImportError:
    HAS_PYGETWINDOW = False
    logger.warning("pygetwindow 未安装，将使用全屏捕获模式（pip install pygetwindow）")


class ScreenCapture:
    """
    屏幕截图类

    支持自动检测雀魂游戏窗口（独立客户端或浏览器），
    以及区域截图和坐标转换功能。
    """

    # 独立客户端可能使用的窗口标题关键词
    GAME_WINDOW_TITLES = [
        "雀魂麻将",
        "Mahjong Soul",
        "MahjongSoul",
        "mahjongsoul",
    ]

    # 浏览器标签页关键词（当游戏在浏览器中运行时）
    BROWSER_KEYWORDS = [
        "mahjongsoul",
        "雀魂麻将",
        "雀魂 -",
        "Mahjong Soul",
    ]

    def __init__(self):
        self._sct = mss.mss()
        self.game_window = None
        self._last_window_check: float = 0.0
        self._window_check_interval: float = 5.0  # 每 5 秒重新检测一次窗口

    # ------------------------------------------------------------------
    # 窗口检测
    # ------------------------------------------------------------------

    def find_game_window(self) -> bool:
        """
        查找雀魂游戏窗口

        Returns:
            bool: 是否成功找到游戏窗口
        """
        if not HAS_PYGETWINDOW:
            logger.info("跳过窗口检测，使用全屏模式")
            return False

        # 1. 精确匹配游戏客户端窗口标题
        for keyword in self.GAME_WINDOW_TITLES:
            try:
                windows = gw.getWindowsWithTitle(keyword)
                if windows:
                    self.game_window = windows[0]
                    logger.info(
                        f"找到游戏窗口: 「{self.game_window.title}」 "
                        f"({self.game_window.width}×{self.game_window.height})"
                    )
                    return True
            except Exception as e:
                logger.debug(f"查找窗口「{keyword}」时出错: {e}")

        # 2. 在所有窗口中查找浏览器游戏标签
        try:
            for win in gw.getAllWindows():
                title_lower = win.title.lower()
                for kw in self.BROWSER_KEYWORDS:
                    if kw.lower() in title_lower:
                        self.game_window = win
                        logger.info(f"在浏览器中找到游戏: 「{win.title}」")
                        return True
        except Exception as e:
            logger.debug(f"遍历所有窗口时出错: {e}")

        logger.warning("未找到雀魂游戏窗口，将使用全屏捕获")
        return False

    def _refresh_window_if_needed(self):
        """定时重新检测游戏窗口（避免窗口移动后坐标失效）"""
        now = time.time()
        if now - self._last_window_check > self._window_check_interval:
            self.find_game_window()
            self._last_window_check = now

    # ------------------------------------------------------------------
    # 区域获取
    # ------------------------------------------------------------------

    def get_game_region(self) -> Dict[str, int]:
        """
        获取游戏画面在屏幕上的绝对区域

        Returns:
            Dict: {top, left, width, height}（像素值）
        """
        self._refresh_window_if_needed()

        if self.game_window:
            try:
                return {
                    "top": max(0, self.game_window.top),
                    "left": max(0, self.game_window.left),
                    "width": self.game_window.width,
                    "height": self.game_window.height,
                }
            except Exception as e:
                logger.debug(f"获取窗口区域失败: {e}")

        # 回退到主显示器全屏
        m = self._sct.monitors[1]
        return {
            "top": m["top"],
            "left": m["left"],
            "width": m["width"],
            "height": m["height"],
        }

    # ------------------------------------------------------------------
    # 截图
    # ------------------------------------------------------------------

    def capture(self) -> np.ndarray:
        """
        截取游戏窗口画面

        Returns:
            np.ndarray: BGR 格式的图像（OpenCV 可直接使用）
        """
        region = self.get_game_region()
        try:
            shot = self._sct.grab(region)
            img = np.array(shot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img
        except Exception as e:
            logger.error(f"截图失败: {e}")
            return np.zeros((1080, 1920, 3), dtype=np.uint8)

    def capture_region_abs(
        self, abs_x: int, abs_y: int, width: int, height: int
    ) -> np.ndarray:
        """
        截取屏幕上的绝对像素区域

        Args:
            abs_x, abs_y: 区域左上角的绝对屏幕坐标
            width, height: 区域尺寸（像素）

        Returns:
            np.ndarray: BGR 图像
        """
        region = {"top": abs_y, "left": abs_x, "width": width, "height": height}
        try:
            shot = self._sct.grab(region)
            img = np.array(shot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img
        except Exception as e:
            logger.error(f"截取绝对区域失败: {e}")
            return np.zeros((height, width, 3), dtype=np.uint8)

    def capture_region_rel(
        self, rel_x: float, rel_y: float, rel_w: float, rel_h: float
    ) -> np.ndarray:
        """
        截取游戏窗口内的相对区域

        Args:
            rel_x, rel_y: 左上角归一化坐标（0~1）
            rel_w, rel_h: 归一化尺寸（0~1）

        Returns:
            np.ndarray: BGR 图像
        """
        game_region = self.get_game_region()
        gw_w = game_region["width"]
        gw_h = game_region["height"]

        abs_x = game_region["left"] + int(rel_x * gw_w)
        abs_y = game_region["top"] + int(rel_y * gw_h)
        width = max(1, int(rel_w * gw_w))
        height = max(1, int(rel_h * gw_h))

        return self.capture_region_abs(abs_x, abs_y, width, height)

    # ------------------------------------------------------------------
    # 坐标转换
    # ------------------------------------------------------------------

    def rel_to_abs(self, rel_x: float, rel_y: float) -> Tuple[int, int]:
        """
        将游戏窗口内的归一化坐标转换为屏幕绝对坐标

        Args:
            rel_x, rel_y: 归一化坐标（0~1）

        Returns:
            (abs_x, abs_y): 屏幕像素坐标
        """
        region = self.get_game_region()
        abs_x = region["left"] + int(rel_x * region["width"])
        abs_y = region["top"] + int(rel_y * region["height"])
        return abs_x, abs_y

    def pixel_to_abs(
        self, pix_x: int, pix_y: int, screenshot_shape: Optional[Tuple[int, int]] = None
    ) -> Tuple[int, int]:
        """
        将截图内的像素坐标转换为屏幕绝对坐标

        Args:
            pix_x, pix_y: 截图内像素坐标
            screenshot_shape: (height, width)，为 None 时使用当前窗口尺寸

        Returns:
            (abs_x, abs_y): 屏幕像素坐标
        """
        region = self.get_game_region()
        if screenshot_shape:
            sh, sw = screenshot_shape
            rel_x = pix_x / sw
            rel_y = pix_y / sh
        else:
            rel_x = pix_x / region["width"]
            rel_y = pix_y / region["height"]

        return self.rel_to_abs(rel_x, rel_y)

    # ------------------------------------------------------------------
    # 属性 / 工具
    # ------------------------------------------------------------------

    @property
    def window_size(self) -> Tuple[int, int]:
        """返回游戏窗口尺寸 (width, height)"""
        region = self.get_game_region()
        return region["width"], region["height"]

    def save_screenshot(self, path: str = "logs/screenshot.png"):
        """截图并保存到文件（用于调试）"""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        img = self.capture()
        cv2.imwrite(path, img)
        logger.debug(f"截图已保存: {path}")

    def get_monitor_info(self) -> str:
        """返回显示器信息字符串（用于调试）"""
        monitors = self._sct.monitors
        lines = []
        for i, m in enumerate(monitors):
            lines.append(
                f"Monitor {i}: top={m['top']}, left={m['left']}, "
                f"{m['width']}×{m['height']}"
            )
        return "\n".join(lines)
