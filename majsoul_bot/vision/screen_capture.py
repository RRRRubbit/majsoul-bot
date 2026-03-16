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

# Canvas 检测失败后降级到浏览器窗口区域的容差（占浏览器高度比例）
_CANVAS_MIN_AREA_RATIO = 0.35   # canvas 至少占浏览器截图面积 35%
_CANVAS_ASPECT_MIN = 1.20       # canvas 最小长宽比 (宽/高)
_CANVAS_ASPECT_MAX = 2.40       # canvas 最大长宽比
_CANVAS_RECT_MIN   = 0.55       # 矩形度下限（轮廓面积/外接矩阵面积）
_CANVAS_CACHE_TTL  = 30.0       # canvas 坐标缓存有效期（秒）

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

    def __init__(
        self,
        auto_topmost: bool = True,
        lock_resolution: bool = True,
        lock_width: Optional[int] = None,
        lock_height: Optional[int] = None,
    ):
        self._sct = mss.mss()
        self.game_window = None

        # 窗口策略：置顶 + 分辨率锁定
        self.auto_topmost = bool(auto_topmost)
        self.lock_resolution = bool(lock_resolution)
        self.lock_width = int(lock_width) if lock_width else None
        self.lock_height = int(lock_height) if lock_height else None
        self._locked_size: Optional[Tuple[int, int]] = None

        self._last_window_check: float = 0.0
        self._window_check_interval: float = 5.0  # 每 5 秒重新检测一次窗口
        self._window_found_state: Optional[bool] = None
        self._last_window_signature: Optional[Tuple[str, int, int, int, int]] = None
        self._last_policy_apply_time: float = 0.0
        self._policy_apply_interval: float = 1.5

        # Canvas 边界检测缓存
        # 格式: (offset_x, offset_y, canvas_w, canvas_h) —— 相对于浏览器窗口左上角的像素偏移
        self._canvas_offset: Optional[Tuple[int, int, int, int]] = None
        self._canvas_cache_time: float = 0.0

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
            if self._window_found_state is not False:
                logger.info("跳过窗口检测，使用全屏模式")
            self._window_found_state = False
            self.game_window = None
            self._last_window_signature = None
            return False

        found_window = None

        # 1. 精确匹配游戏客户端窗口标题
        for keyword in self.GAME_WINDOW_TITLES:
            try:
                windows = gw.getWindowsWithTitle(keyword)
                if windows:
                    found_window = windows[0]
                    break
            except Exception as e:
                logger.debug(f"查找窗口「{keyword}」时出错: {e}")

        # 2. 在所有窗口中查找浏览器游戏标签
        if found_window is None:
            try:
                for win in gw.getAllWindows():
                    title = win.title or ""
                    title_lower = title.lower()
                    for kw in self.BROWSER_KEYWORDS:
                        if kw.lower() in title_lower:
                            found_window = win
                            break
                    if found_window is not None:
                        break
            except Exception as e:
                logger.debug(f"遍历所有窗口时出错: {e}")

        if found_window is not None:
            self.game_window = found_window
            self._apply_window_policy(reason="find")
            try:
                signature = (
                    self.game_window.title,
                    int(self.game_window.left),
                    int(self.game_window.top),
                    int(self.game_window.width),
                    int(self.game_window.height),
                )
            except Exception:
                signature = None

            if signature != self._last_window_signature:
                logger.info(
                    f"找到游戏窗口: 「{self.game_window.title}」 "
                    f"({self.game_window.width}×{self.game_window.height})"
                )

            self._last_window_signature = signature
            self._window_found_state = True
            return True

        if self._window_found_state is not False:
            logger.warning("未找到雀魂游戏窗口，将使用全屏捕获")
        self.game_window = None
        self._last_window_signature = None
        self._window_found_state = False
        return False

    def _refresh_window_if_needed(self):
        """定时重新检测游戏窗口（避免窗口移动后坐标失效）"""
        now = time.time()
        if now - self._last_window_check > self._window_check_interval:
            self.find_game_window()
            self._last_window_check = now

        # 即使不重新找窗口，也定时重施加置顶/分辨率策略
        self._apply_window_policy_if_needed()

    def _apply_window_policy_if_needed(self):
        if not self.game_window or not HAS_PYGETWINDOW:
            return
        now = time.time()
        if now - self._last_policy_apply_time < self._policy_apply_interval:
            return
        self._apply_window_policy(reason="periodic")

    def _apply_window_policy(self, reason: str = ""):
        """对已识别窗口应用置顶和分辨率锁定策略。"""
        if not self.game_window or not HAS_PYGETWINDOW:
            return

        # 1) 自动置顶/激活窗口
        if self.auto_topmost:
            try:
                if hasattr(self.game_window, "restore") and getattr(self.game_window, "isMinimized", False):
                    self.game_window.restore()
                if hasattr(self.game_window, "activate"):
                    self.game_window.activate()
            except Exception as e:
                logger.debug(f"窗口置顶失败({reason}): {e}")

        # 2) 分辨率锁定
        if self.lock_resolution:
            try:
                target_w = self.lock_width
                target_h = self.lock_height

                if target_w is None or target_h is None:
                    if self._locked_size is None:
                        self._locked_size = (int(self.game_window.width), int(self.game_window.height))
                        logger.info(f"锁定窗口分辨率为首次尺寸: {self._locked_size[0]}×{self._locked_size[1]}")
                    target_w, target_h = self._locked_size

                cur_w = int(self.game_window.width)
                cur_h = int(self.game_window.height)
                if cur_w != int(target_w) or cur_h != int(target_h):
                    if hasattr(self.game_window, "resizeTo"):
                        self.game_window.resizeTo(int(target_w), int(target_h))
                    logger.info(
                        f"窗口分辨率已纠正: {cur_w}×{cur_h} -> {int(target_w)}×{int(target_h)}"
                    )
            except Exception as e:
                logger.debug(f"窗口分辨率锁定失败({reason}): {e}")

        self._last_policy_apply_time = time.time()

    # ------------------------------------------------------------------
    # 区域获取
    # ------------------------------------------------------------------

    def _raw_browser_region(self) -> Dict[str, int]:
        """返回浏览器（或全屏）的原始像素区域，不做 canvas 修正。"""
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

        m = self._sct.monitors[1]
        return {
            "top": m["top"],
            "left": m["left"],
            "width": m["width"],
            "height": m["height"],
        }

    # ------------------------------------------------------------------
    # Canvas 边界检测
    # ------------------------------------------------------------------

    def _detect_game_canvas(self, screenshot: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        在浏览器窗口截图中检测游戏 canvas 的边界矩形。

        算法：
        1. 灰度化 + 高斯模糊，降低噪声
        2. Canny 边缘检测 + 形态学膨胀，将破碎边缘连通
        3. 找外轮廓，筛选面积最大且形状接近矩形的候选区域
        4. 要求满足：最小面积比例、合理长宽比、矩形度阈值

        Args:
            screenshot: 浏览器窗口截图（BGR，相对于窗口左上角）

        Returns:
            (offset_x, offset_y, canvas_w, canvas_h) 相对于截图左上角的像素坐标，
            或 None（未检测到满足条件的 canvas）
        """
        if screenshot is None or screenshot.size == 0:
            return None

        sh, sw = screenshot.shape[:2]
        screen_area = sh * sw

        # —— 1. 预处理 ——
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # —— 2. Canny 边缘检测 ——
        edges = cv2.Canny(blurred, 20, 80)

        # 膨胀以连通邻近边缘像素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=3)

        # —— 3. 找外轮廓 ——
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # —— 4. 筛选 ——
        best: Optional[Tuple[int, int, int, int]] = None
        best_score: float = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < screen_area * _CANVAS_MIN_AREA_RATIO:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            rect_area = cw * ch
            if rect_area == 0:
                continue

            # 矩形度：轮廓面积与外接矩形面积之比
            rectangularity = area / rect_area
            if rectangularity < _CANVAS_RECT_MIN:
                continue

            # 长宽比检查
            aspect = cw / ch if ch > 0 else 0.0
            if not (_CANVAS_ASPECT_MIN <= aspect <= _CANVAS_ASPECT_MAX):
                continue

            # 综合打分：面积越大、矩形度越高越优先
            score = area * rectangularity
            if score > best_score:
                best_score = score
                best = (x, y, cw, ch)

        # —— 5. 备选策略：若 Canny 路失败，尝试用颜色空间找深色矩形边框 ——
        if best is None:
            best = self._detect_canvas_by_dark_border(screenshot, screen_area)

        return best

    def _detect_canvas_by_dark_border(
        self, screenshot: np.ndarray, screen_area: int
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        备用策略：通过检测图像四周是否存在大面积深色/单色边框来定位 canvas。

        原理：浏览器工具栏通常颜色较浅/单一，游戏画面本身有明显的内容边界。
        通过扫描水平和垂直方向的亮度阶跃（标准差），找到内容区起止行列。
        """
        sh, sw = screenshot.shape[:2]
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # 每行/列的方差，方差高的行说明有复杂内容
        row_var = np.var(gray, axis=1).astype(float)  # shape (sh,)
        col_var = np.var(gray, axis=0).astype(float)  # shape (sw,)

        # 阈值：方差 > 全图方差均值 * 0.3 视为有内容的行/列
        row_thresh = float(np.mean(row_var)) * 0.3
        col_thresh = float(np.mean(col_var)) * 0.3

        content_rows = np.where(row_var > row_thresh)[0]
        content_cols = np.where(col_var > col_thresh)[0]

        if len(content_rows) < 10 or len(content_cols) < 10:
            return None

        y1, y2 = int(content_rows[0]), int(content_rows[-1])
        x1, x2 = int(content_cols[0]), int(content_cols[-1])
        cw, ch = x2 - x1, y2 - y1

        if cw * ch < screen_area * _CANVAS_MIN_AREA_RATIO:
            return None

        aspect = cw / ch if ch > 0 else 0.0
        if not (_CANVAS_ASPECT_MIN <= aspect <= _CANVAS_ASPECT_MAX):
            return None

        return (x1, y1, cw, ch)

    def _get_canvas_offset(self) -> Optional[Tuple[int, int, int, int]]:
        """
        获取（带缓存）游戏 canvas 相对于浏览器窗口的像素偏移。

        Returns:
            (offset_x, offset_y, canvas_w, canvas_h) 或 None
        """
        now = time.time()

        # 缓存未过期则直接返回
        if self._canvas_offset is not None and (now - self._canvas_cache_time) < _CANVAS_CACHE_TTL:
            return self._canvas_offset

        # 需要重新检测
        browser_region = self._raw_browser_region()
        try:
            shot = self._sct.grab(browser_region)
            img = np.array(shot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception as e:
            logger.debug(f"Canvas 检测截图失败: {e}")
            return None

        result = self._detect_game_canvas(img)

        if result is not None:
            ox, oy, cw, ch = result
            self._canvas_offset = (ox, oy, cw, ch)
            self._canvas_cache_time = now
            logger.info(
                f"🎮 游戏 Canvas 已定位: 偏移=({ox}, {oy}), 尺寸={cw}×{ch} "
                f"(浏览器窗口: {browser_region['width']}×{browser_region['height']})"
            )
        else:
            self._canvas_offset = None
            logger.debug("Canvas 边界检测未找到匹配区域，使用完整浏览器窗口")

        return self._canvas_offset

    def force_refresh_canvas(self):
        """强制清除 canvas 缓存，下次调用 get_game_region() 时重新检测。"""
        self._canvas_offset = None
        self._canvas_cache_time = 0.0
        logger.debug("Canvas 缓存已清除，将在下次调用时重新检测")

    def get_game_region(self, use_canvas_detection: bool = True) -> Dict[str, int]:
        """
        获取游戏画面在屏幕上的绝对区域。

        当 use_canvas_detection=True（默认）时，会尝试通过边界检测找到
        浏览器内真实的游戏 canvas 区域；若检测失败则降级到整个浏览器窗口。

        Args:
            use_canvas_detection: 是否启用 canvas 边界检测

        Returns:
            Dict: {top, left, width, height}（像素值）
        """
        self._refresh_window_if_needed()

        browser = self._raw_browser_region()

        if use_canvas_detection and self.game_window is not None:
            offset = self._get_canvas_offset()
            if offset is not None:
                ox, oy, cw, ch = offset
                return {
                    "top":    browser["top"]  + oy,
                    "left":   browser["left"] + ox,
                    "width":  cw,
                    "height": ch,
                }

        return browser

    # ------------------------------------------------------------------
    # 截图
    # ------------------------------------------------------------------

    def capture(self) -> np.ndarray:
        """
        截取游戏窗口画面

        Returns:
            np.ndarray: BGR 格式的图像（OpenCV 可直接使用）
        """
        last_error: Optional[Exception] = None

        # 某些场景（窗口刚切换/浏览器硬件加速帧）可能偶发纯黑帧，做轻量重试
        for attempt in range(3):
            region = self.get_game_region()
            try:
                shot = self._sct.grab(region)
                img = np.array(shot)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                # 纯黑/近纯黑帧判定（避免把无效帧当成有效截图）
                if img.size > 0 and float(np.mean(img)) <= 1.0 and int(np.max(img)) <= 5:
                    if attempt < 2:
                        logger.warning("检测到近纯黑截图，正在重试捕获")
                        # 重新检测窗口，避免窗口坐标或状态瞬时异常
                        self.find_game_window()
                        time.sleep(0.05)
                        continue

                return img
            except Exception as e:
                last_error = e
                if attempt < 2:
                    time.sleep(0.05)
                    continue

        if last_error is not None:
            logger.error(f"截图失败: {last_error}")
        else:
            logger.error("截图失败: 捕获结果持续为近纯黑帧")
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
