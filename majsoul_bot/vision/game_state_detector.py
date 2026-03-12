ba"""
游戏状态检测模块
通过图像识别检测雀魂当前的游戏阶段与可用操作
"""
import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger

from .regions import ScreenRegions, DEFAULT_REGIONS


# ──────────────────────────────────────────────
# 游戏阶段枚举
# ──────────────────────────────────────────────

class GamePhase(Enum):
    """游戏当前所处阶段"""
    UNKNOWN            = "unknown"           # 未知/初始
    LOADING            = "loading"           # 加载/过渡画面
    LOBBY              = "lobby"             # 大厅界面
    WAITING            = "waiting"           # 等待其他玩家操作
    MY_TURN_DISCARD    = "my_turn_discard"   # 轮到我打牌
    OPERATION_AVAILABLE= "operation_available"  # 可执行碰/吃/杠
    RIICHI_AVAILABLE   = "riichi_available"  # 可宣告立直
    WIN_AVAILABLE      = "win_available"     # 可自摸/荣和
    ROUND_RESULT       = "round_result"      # 本局结算画面
    GAME_OVER          = "game_over"         # 游戏结束结算


# ──────────────────────────────────────────────
# 检测结果数据类
# ──────────────────────────────────────────────

@dataclass
class DetectedState:
    """单帧检测结果"""
    phase: GamePhase = GamePhase.UNKNOWN

    # 检测到的操作按钮 → {按钮名: (归一化 rel_x, rel_y)}
    buttons: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # 当前手牌张数（不含摸牌），-1 表示未知
    hand_tile_count: int = 13

    # 是否检测到摸牌（第 14 张）
    has_drawn_tile: bool = False

    # 额外调试信息
    debug_info: Dict = field(default_factory=dict)


# ──────────────────────────────────────────────
# 检测器主类
# ──────────────────────────────────────────────

class GameStateDetector:
    """
    游戏状态检测器

    检测逻辑优先级（由高到低）：
    1. 检测和牌按钮（自摸/荣和） → WIN_AVAILABLE
    2. 检测立直按钮               → RIICHI_AVAILABLE
    3. 检测碰/吃/杠按钮           → OPERATION_AVAILABLE
    4. 检测手牌高亮（我的回合）    → MY_TURN_DISCARD
    5. 否则                       → WAITING

    按钮检测优先使用模板匹配（templates/buttons/），
    若无模板则退为基于 HSV 颜色 + 轮廓面积的颜色检测。
    """

    # 需要识别的按钮名称
    BUTTON_NAMES = ["pon", "chi", "kan", "riichi", "tsumo", "ron", "skip"]

    # ── 颜色检测辅助：各按钮在 HSV 空间的大致范围 (lower, upper) ──
    # 雀魂按钮颜色近似（可根据实际截图调整）
    _HSV_RANGES: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # 手牌"我的回合"亮度阈值（灰度均值）
    MY_TURN_BRIGHTNESS = 148

    def __init__(
        self,
        templates_dir: str = "templates/buttons",
        regions: Optional[ScreenRegions] = None,
        threshold: float = 0.72,
    ):
        """
        Args:
            templates_dir: 按钮模板图片目录
            regions: 屏幕区域配置
            threshold: 模板匹配阈值
        """
        self.templates_dir = Path(templates_dir)
        self.regions = regions or DEFAULT_REGIONS
        self.threshold = threshold
        self.button_templates: Dict[str, np.ndarray] = {}
        self._load_button_templates()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _load_button_templates(self):
        """加载按钮模板图片"""
        if not self.templates_dir.exists():
            logger.info(
                f"按钮模板目录不存在: {self.templates_dir}，将使用颜色检测\n"
                "  → 运行 tools/capture_templates.py 可生成按钮模板"
            )
            return

        for name in self.BUTTON_NAMES:
            for ext in (".png", ".jpg"):
                p = self.templates_dir / f"{name}{ext}"
                if p.exists():
                    img = cv2.imread(str(p))
                    if img is not None:
                        self.button_templates[name] = img
                        logger.debug(f"加载按钮模板: {name}")
                        break

        logger.info(f"已加载 {len(self.button_templates)} 个按钮模板")

    # ------------------------------------------------------------------
    # 主检测入口
    # ------------------------------------------------------------------

    def detect(self, screenshot: np.ndarray) -> DetectedState:
        """
        检测截图中的游戏状态

        Args:
            screenshot: BGR 格式游戏截图

        Returns:
            DetectedState
        """
        state = DetectedState()

        # 1. 检测操作按钮
        buttons = self._detect_buttons(screenshot)
        state.buttons = buttons
        state.debug_info["buttons"] = list(buttons.keys())

        # 2. 根据按钮判断阶段
        if "tsumo" in buttons or "ron" in buttons:
            state.phase = GamePhase.WIN_AVAILABLE
        elif "riichi" in buttons:
            state.phase = GamePhase.RIICHI_AVAILABLE
        elif buttons:
            state.phase = GamePhase.OPERATION_AVAILABLE
        else:
            # 3. 无按钮时：检测是否轮到我打牌
            my_turn, has_drawn = self._detect_my_turn_and_drawn(screenshot)
            if my_turn:
                state.phase = GamePhase.MY_TURN_DISCARD
                state.has_drawn_tile = has_drawn
            else:
                state.phase = GamePhase.WAITING

        logger.debug(
            f"[State] phase={state.phase.value} "
            f"buttons={list(state.buttons.keys())} "
            f"drawn={state.has_drawn_tile}"
        )
        return state

    # ------------------------------------------------------------------
    # 按钮检测
    # ------------------------------------------------------------------

    def _detect_buttons(
        self, screenshot: np.ndarray
    ) -> Dict[str, Tuple[float, float]]:
        """综合检测操作按钮"""
        if self.button_templates:
            # 优先模板匹配
            result = self._detect_by_template(screenshot)
            if result:
                return result
            # 有模板但本次未匹配，才允许颜色检测作为补充
            return self._detect_by_color(screenshot)

        # ── 无模板时：颜色检测误报率极高，直接跳过 ──
        # 用户可运行 tools/capture_templates.py 生成按钮模板后再启用
        return {}

    def _detect_by_template(
        self, screenshot: np.ndarray
    ) -> Dict[str, Tuple[float, float]]:
        """模板匹配检测按钮"""
        found: Dict[str, Tuple[float, float]] = {}
        h, w = screenshot.shape[:2]

        # 只在按钮扫描区域内匹配，提高速度
        sx = int(self.regions.button_scan_x * w)
        sy = int(self.regions.button_scan_y * h)
        sw = int(self.regions.button_scan_w * w)
        sh = int(self.regions.button_scan_h * h)
        scan = screenshot[sy : sy + sh, sx : sx + sw]

        for name, tmpl in self.button_templates.items():
            th, tw = tmpl.shape[:2]
            # 缩放模板至不超过扫描区域
            if th > sh or tw > sw:
                scale = min(sh / th, sw / tw) * 0.9
                tmpl = cv2.resize(tmpl, (int(tw * scale), int(th * scale)))
                th, tw = tmpl.shape[:2]

            if th > scan.shape[0] or tw > scan.shape[1]:
                continue

            res = cv2.matchTemplate(scan, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val >= self.threshold:
                cx = (sx + max_loc[0] + tw // 2) / w
                cy = (sy + max_loc[1] + th // 2) / h
                found[name] = (cx, cy)
                logger.debug(
                    f"模板匹配按钮「{name}」: ({cx:.3f}, {cy:.3f}) 得分={max_val:.3f}"
                )

        return found

    def _detect_by_color(
        self, screenshot: np.ndarray
    ) -> Dict[str, Tuple[float, float]]:
        """
        基于颜色/形状检测操作按钮（无模板时的备用方法）

        思路：
        - 在按钮扫描区域内转为 HSV
        - 筛选高饱和度、高亮度的像素（按钮通常色彩鲜艳）
        - 用轮廓分析找到矩形色块（按钮候选）
        - 按 x 坐标从左到右排列，并匹配预设顺序
        """
        found: Dict[str, Tuple[float, float]] = {}
        h, w = screenshot.shape[:2]

        sx = int(self.regions.button_scan_x * w)
        sy = int(self.regions.button_scan_y * h)
        sw = int(self.regions.button_scan_w * w)
        sh = int(self.regions.button_scan_h * h)

        if sx + sw > w or sy + sh > h:
            return found

        scan = screenshot[sy : sy + sh, sx : sx + sw]
        hsv = cv2.cvtColor(scan, cv2.COLOR_BGR2HSV)

        # ── 筛选高饱和度 + 高亮度区域（雀魂按钮色彩非常鲜艳）──
        # 阈值收紧到 sat>130, val>160，避免把游戏背景误识别为按钮
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        mask = ((sat > 130) & (val > 160)).astype(np.uint8) * 255

        # 膨胀操作，将相邻像素连接为完整按钮轮廓
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 面积过滤：按钮约占扫描区域 4%~20%（稍微提高下限，减少小噪点）
        min_area = sw * sh * 0.04
        max_area = sw * sh * 0.22

        candidates: List[Tuple[float, float, float]] = []  # (cx, cy, area)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area <= area <= max_area):
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            # 按钮宽高比约 1.2 ~ 3.5（再收紧，排除宽条）
            ratio = bw / bh if bh > 0 else 0
            if not (1.2 <= ratio <= 3.5):
                continue
            cx = (sx + bx + bw / 2) / w
            cy = (sy + by + bh / 2) / h
            candidates.append((cx, cy, area))

        # 候选数超过 5 个说明大概率是误检测（背景干扰），直接放弃
        if len(candidates) > 5:
            logger.debug(
                f"颜色检测到 {len(candidates)} 个候选，疑似误检测，忽略"
            )
            return found

        # 按 x 坐标从左到右排序
        candidates.sort(key=lambda c: c[0])

        # 雀魂操作按钮常见顺序（左 → 右）：跳过 / 吃 / 碰 / 杠
        # 和牌按钮通常单独出现在中央偏右
        default_order = ["skip", "chi", "pon", "kan", "riichi"]
        for i, (cx, cy, _) in enumerate(candidates[:5]):
            if i < len(default_order):
                found[default_order[i]] = (cx, cy)
                logger.debug(f"颜色检测按钮候选「{default_order[i]}」: ({cx:.3f}, {cy:.3f})")

        return found

    # ------------------------------------------------------------------
    # 回合检测
    # ------------------------------------------------------------------

    def _detect_my_turn_and_drawn(
        self, screenshot: np.ndarray
    ) -> Tuple[bool, bool]:
        """
        检测是否轮到我打牌，以及是否有摸牌

        Returns:
            (is_my_turn, has_drawn_tile)
        """
        h, w = screenshot.shape[:2]
        reg = self.regions.hand

        # ── 1. 检测手牌区域亮度 ──
        x1 = int(reg.x_start * w)
        y1 = int(reg.y_start * h)
        x2 = int((reg.x_start + reg.max_tiles * reg.tile_width) * w)
        y2 = int((reg.y_start + reg.tile_height) * h)

        x2 = min(x2, w)
        y2 = min(y2, h)

        if x1 >= w or y1 >= h or x2 <= x1 or y2 <= y1:
            return False, False

        hand_region = screenshot[y1:y2, x1:x2]
        if hand_region.size == 0:
            return False, False

        gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))

        is_my_turn = mean_brightness > self.MY_TURN_BRIGHTNESS

        # ── 2. 检测摸牌位置是否有牌 ──
        has_drawn = False
        if is_my_turn:
            has_drawn = self._detect_drawn_tile(screenshot)

        logger.debug(
            f"手牌亮度={mean_brightness:.1f} "
            f"my_turn={is_my_turn} has_drawn={has_drawn}"
        )
        return is_my_turn, has_drawn

    def _detect_drawn_tile(self, screenshot: np.ndarray) -> bool:
        """
        检测摸牌区域（第 14 张位置）是否有牌

        通过检测该区域的亮度+纹理来判断
        """
        h, w = screenshot.shape[:2]
        reg = self.regions.hand

        drawn_x = int(
            (reg.x_start + reg.max_tiles * reg.tile_width + reg.drawn_gap) * w
        )
        drawn_y = int(reg.y_start * h)
        tile_w = max(1, int(reg.tile_width * w))
        tile_h = max(1, int(reg.tile_height * h))

        x_end = min(drawn_x + tile_w, w)
        y_end = min(drawn_y + tile_h, h)

        if drawn_x >= w or drawn_y >= h:
            return False

        region = screenshot[drawn_y:y_end, drawn_x:x_end]
        if region.size == 0:
            return False

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        texture = float(np.std(gray))

        # 有牌：亮度中等（>100），且有一定纹理（std>15）
        return brightness > 100 and texture > 15

    # ------------------------------------------------------------------
    # 调试可视化
    # ------------------------------------------------------------------

    def visualize(
        self, screenshot: np.ndarray, state: DetectedState
    ) -> np.ndarray:
        """
        在截图上叠加可视化标注（调试用）

        标注内容：
        - 绿框：手牌扫描区域
        - 红圈+文字：检测到的操作按钮
        - 橙框：按钮扫描区域
        - 顶部文字：当前游戏阶段
        """
        debug = screenshot.copy()
        h, w = debug.shape[:2]
        reg = self.regions.hand

        # 手牌区域（绿框）
        hx1 = int(reg.x_start * w)
        hy1 = int(reg.y_start * h)
        hx2 = int((reg.x_start + reg.max_tiles * reg.tile_width) * w)
        hy2 = int((reg.y_start + reg.tile_height) * h)
        cv2.rectangle(debug, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)

        # 摸牌区域（蓝框）
        dx = int(
            (reg.x_start + reg.max_tiles * reg.tile_width + reg.drawn_gap) * w
        )
        dw = int(reg.tile_width * w)
        cv2.rectangle(
            debug,
            (dx, hy1),
            (dx + dw, hy2),
            (255, 180, 0), 2,
        )

        # 按钮扫描区域（橙框）
        bx1 = int(self.regions.button_scan_x * w)
        by1 = int(self.regions.button_scan_y * h)
        bx2 = int((self.regions.button_scan_x + self.regions.button_scan_w) * w)
        by2 = int((self.regions.button_scan_y + self.regions.button_scan_h) * h)
        cv2.rectangle(debug, (bx1, by1), (bx2, by2), (0, 165, 255), 1)

        # 检测到的按钮（红圈）
        for btn_name, (rx, ry) in state.buttons.items():
            px, py = int(rx * w), int(ry * h)
            cv2.circle(debug, (px, py), 22, (0, 0, 220), 3)
            cv2.putText(
                debug, btn_name,
                (px - 22, py - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 220), 2,
            )

        # 摸牌状态
        if state.has_drawn_tile:
            cv2.putText(
                debug, "DRAWN",
                (dx, hy1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2,
            )

        # 顶部状态栏
        phase_text = f"Phase: {state.phase.value}"
        cv2.rectangle(debug, (0, 0), (w, 44), (30, 30, 30), -1)
        cv2.putText(
            debug, phase_text,
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 230, 50), 2,
        )

        return debug
