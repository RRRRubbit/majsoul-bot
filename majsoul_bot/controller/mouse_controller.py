"""
鼠标控制模块
提供人性化的鼠标移动与点击操作，模拟真实玩家行为
"""
import asyncio
import math
import random
import time
from typing import Optional, Tuple

import pyautogui
from loguru import logger

# 保留失败安全（移到屏幕角落会暂停），防止意外操作
pyautogui.FAILSAFE = True
# 禁用 pyautogui 内置的固定延迟，由我们自己控制
pyautogui.PAUSE = 0.0


class MouseController:
    """
    鼠标控制器

    特性：
    - 随机点击偏移：模拟人手抖动
    - 贝塞尔曲线移动：模拟人类鼠标轨迹
    - 可配置操作延迟
    - 异步 API，不阻塞主循环
    """

    def __init__(
        self,
        min_delay: float = 0.8,
        max_delay: float = 2.5,
        click_variance: int = 5,
        move_duration_base: float = 0.3,
    ):
        """
        Args:
            min_delay: 每次操作前的最小等待时间（秒）
            max_delay: 每次操作前的最大等待时间（秒）
            click_variance: 点击位置的随机像素偏移量（±pixels）
            move_duration_base: 鼠标移动基础时长（秒）
        """
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.click_variance = click_variance
        self.move_duration_base = move_duration_base
        self._last_click_time: float = 0.0

    # ------------------------------------------------------------------
    # 核心点击方法
    # ------------------------------------------------------------------

    async def click(
        self,
        x: int,
        y: int,
        delay: bool = True,
        button: str = "left",
        double: bool = False,
    ):
        """
        异步点击指定屏幕坐标

        Args:
            x, y: 绝对屏幕坐标（像素）
            delay: 是否在点击前随机等待（模拟思考时间）
            button: 鼠标按键（"left" / "right" / "middle"）
            double: 是否双击
        """
        if delay:
            wait = random.uniform(self.min_delay, self.max_delay)
            logger.debug(f"等待 {wait:.2f}s 后点击 ({x}, {y})")
            await asyncio.sleep(wait)

        # 加入随机抖动
        jx = random.randint(-self.click_variance, self.click_variance)
        jy = random.randint(-self.click_variance, self.click_variance)
        fx, fy = x + jx, y + jy

        # 人性化移动
        self._move_humanlike(fx, fy)

        # 短暂停顿（落指延迟）
        await asyncio.sleep(random.uniform(0.04, 0.12))

        try:
            if double:
                pyautogui.doubleClick(fx, fy, button=button)
            else:
                pyautogui.click(fx, fy, button=button)
            self._last_click_time = time.time()
            logger.debug(f"{'双击' if double else '点击'} ({fx}, {fy})")
        except Exception as e:
            logger.error(f"鼠标点击失败: {e}")

    async def click_relative(
        self,
        rel_x: float,
        rel_y: float,
        screen_capture,
        delay: bool = True,
        button: str = "left",
    ):
        """
        点击游戏窗口内的归一化坐标位置

        Args:
            rel_x, rel_y: 归一化坐标（0~1）
            screen_capture: ScreenCapture 实例（提供坐标转换）
            delay, button: 同 click()
        """
        abs_x, abs_y = screen_capture.rel_to_abs(rel_x, rel_y)
        await self.click(abs_x, abs_y, delay=delay, button=button)

    async def click_pixel(
        self,
        pix_x: int,
        pix_y: int,
        screen_capture,
        screenshot_shape: Optional[Tuple[int, int]] = None,
        delay: bool = True,
    ):
        """
        点击截图内的像素坐标（自动转换为屏幕绝对坐标）

        Args:
            pix_x, pix_y: 截图内像素坐标
            screen_capture: ScreenCapture 实例
            screenshot_shape: (height, width)，为 None 时用窗口尺寸
            delay: 是否等待
        """
        abs_x, abs_y = screen_capture.pixel_to_abs(pix_x, pix_y, screenshot_shape)
        await self.click(abs_x, abs_y, delay=delay)

    # ------------------------------------------------------------------
    # 鼠标移动
    # ------------------------------------------------------------------

    def _move_humanlike(self, target_x: int, target_y: int):
        """
        使用平滑缓动移动鼠标到目标位置（模拟人类操作）

        移动时间根据距离动态计算，远距离移动更慢。
        """
        try:
            cx, cy = pyautogui.position()
            dist = math.hypot(target_x - cx, target_y - cy)

            # 根据距离计算移动时长：最短 0.08s，每 500px 增加 0.1s
            duration = self.move_duration_base + dist / 5000.0
            duration = max(0.08, min(duration, 0.6))

            # 加入少量随机抖动使轨迹更自然
            duration += random.uniform(-0.02, 0.04)

            pyautogui.moveTo(
                target_x,
                target_y,
                duration=duration,
                tween=pyautogui.easeOutQuad,
            )
        except Exception as e:
            # 平滑移动失败时直接跳到目标
            logger.debug(f"平滑移动失败，直接定位: {e}")
            try:
                pyautogui.moveTo(target_x, target_y)
            except Exception:
                pass

    async def move_to(self, x: int, y: int, smooth: bool = True):
        """异步移动鼠标（不点击）"""
        if smooth:
            self._move_humanlike(x, y)
        else:
            pyautogui.moveTo(x, y)

    # ------------------------------------------------------------------
    # 键盘辅助（备用）
    # ------------------------------------------------------------------

    async def press_key(self, key: str, delay: bool = False):
        """
        按下键盘按键

        Args:
            key: 键名（如 'enter', 'space', 'esc'）
            delay: 是否等待
        """
        if delay:
            await asyncio.sleep(random.uniform(0.3, 0.8))
        try:
            pyautogui.press(key)
            logger.debug(f"按键: {key}")
        except Exception as e:
            logger.error(f"按键失败: {e}")

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def time_since_last_click(self) -> float:
        """距上次点击经过的秒数"""
        return time.time() - self._last_click_time

    def update_delay(self, min_delay: float, max_delay: float):
        """动态更新操作延迟范围"""
        self.min_delay = min_delay
        self.max_delay = max_delay
        logger.debug(f"操作延迟已更新: [{min_delay:.1f}s, {max_delay:.1f}s]")
