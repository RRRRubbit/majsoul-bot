"""
游戏状态检测模块
通过图像识别检测雀魂当前的游戏阶段与可用操作
"""
import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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
    READY_SCREEN       = "ready_screen"      # 准备界面（可点击准备按钮）
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

    # 按钮置信度得分 → {按钮名: score}
    button_scores: Dict[str, float] = field(default_factory=dict)

    # 当前手牌张数（不含摸牌），-1 表示未知
    hand_tile_count: int = 13

    # 是否检测到摸牌（第 14 张）
    has_drawn_tile: bool = False

    # 牌堆区域亮度/纹理度量（用于判断游戏中/结算）
    wall_metrics: Dict[str, float] = field(default_factory=dict)

    # 宝牌区域图像切片（可选，调试用）
    dora_img: Optional[Any] = None

    # 四家副露变化度（亮度/纹理变化，>0 说明有副露出现）
    meld_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # ── 新增：游戏阶段信息 ──
    # 当前出牌家（0=自己, 1=下家右, 2=对家, 3=上家左，-1=未知）
    current_player: int = -1
    
    # 场风（东/南/西/北，1~4，-1=未知）
    round_wind: int = -1
    
    # 自风（东/南/西/北，1~4，-1=未知）
    seat_wind: int = -1
    
    # 四家手牌亮度 {seat: brightness}，seat in ['self', 'right', 'opposite', 'left']
    player_brightness: Dict[str, float] = field(default_factory=dict)

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

    # 手牌"我的回合"检测阈值
    # 说明：不同分辨率/浏览器缩放下，绝对亮度会波动，
    # 因此采用“绝对阈值 + 相对亮度差 + 摸牌位检测”联合判定。
    MY_TURN_BRIGHTNESS = 148
    MY_TURN_MIN_BRIGHTNESS = 118
    MY_TURN_MIN_DELTA = 10
    MY_TURN_MIN_TEXTURE = 16

    # 小窗口/浏览器边框场景下，手牌区域可能出现轻微偏移，
    # 通过候选偏移搜索增强鲁棒性（单位：相对屏幕宽高）
    _HAND_OFFSET_X_CANDIDATES = (0.0, -0.02, 0.02)
    _HAND_OFFSET_Y_CANDIDATES = (0.0, -0.08, -0.05, -0.03, 0.02)

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

        # 1. 检测操作按钮（返回 {name: (cx,cy)} 和 {name: score}）
        buttons, button_scores = self._detect_buttons_with_scores(screenshot)
        state.buttons = buttons
        state.button_scores = button_scores
        state.debug_info["buttons"] = list(buttons.keys())

        # 2. 根据按钮判断阶段
        # 当 ron 与其他操作按钮同时出现时，优先交由策略层决定；
        # 仅在有 tsumo/ron 且无其他操作类按钮时才直接进入 WIN_AVAILABLE。
        op_btns = {k for k in buttons if k not in {"tsumo", "ron", "riichi"}}
        win_btns = {k for k in buttons if k in {"tsumo", "ron"}}

        if win_btns and not op_btns:
            # 纯和牌场景
            state.phase = GamePhase.WIN_AVAILABLE
        elif win_btns and op_btns:
            # 荣和与吃/碰/杠并存（把 ron 作为操作选项一并传入 operation 处理）
            state.phase = GamePhase.OPERATION_AVAILABLE
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

        # 4. 附加区域检测（牌堆/副露/宝牌）
        state.wall_metrics = self._detect_wall_metrics(screenshot)
        state.meld_metrics = self._detect_meld_metrics(screenshot)

        # 5. 检测当前出牌家（通过中心黄色指示器）
        state.current_player = self._detect_current_player(screenshot)

        logger.debug(
            f"[State] phase={state.phase.value} "
            f"buttons={list(state.buttons.keys())} "
            f"drawn={state.has_drawn_tile} "
            f"current_player={state.current_player}"
        )
        return state

    # ------------------------------------------------------------------
    # 按钮检测
    # ------------------------------------------------------------------

    def _detect_buttons(
        self, screenshot: np.ndarray
    ) -> Dict[str, Tuple[float, float]]:
        """综合检测操作按钮（仅返回坐标字典，兼容旧接口）"""
        buttons, _ = self._detect_buttons_with_scores(screenshot)
        return buttons

    def _detect_buttons_with_scores(
        self, screenshot: np.ndarray
    ) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float]]:
        """综合检测操作按钮，返回 (坐标字典, 置信度字典)"""
        if self.button_templates:
            result, scores = self._detect_by_template_with_scores(screenshot)
            if result:
                return result, scores
            # 有模板但本次未匹配，才允许颜色检测作为补充
            color_result = self._detect_by_color(screenshot)
            return color_result, {k: 0.0 for k in color_result}

        # ── 无模板时：颜色检测误报率极高，直接跳过 ──
        return {}, {}

    def _detect_by_template(
        self, screenshot: np.ndarray
    ) -> Dict[str, Tuple[float, float]]:
        """模板匹配检测按钮（仅返回坐标字典）"""
        result, _ = self._detect_by_template_with_scores(screenshot)
        return result

    def _detect_by_template_with_scores(
        self, screenshot: np.ndarray
    ) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float]]:
        """模板匹配检测按钮，返回 (坐标字典, 置信度字典)。

        多按钮并发友好策略：
        - 每个按钮独立匹配，互不干扰（避免一个按钮的高分遮蔽其他按钮）。
        - 多尺度 × 双通道（BGR + 灰度）取最高分。
        - 操作类按钮（skip/chi/pon/kan/ron）使用较低阈值，提高召回。
        - 对位置相近的按钮不做 NMS 压制（允许同时命中）。
        """
        found: Dict[str, Tuple[float, float]] = {}
        scores: Dict[str, float] = {}
        h, w = screenshot.shape[:2]

        # 只在按钮扫描区域内匹配，提高速度
        sx = int(self.regions.button_scan_x * w)
        sy = int(self.regions.button_scan_y * h)
        sw = int(self.regions.button_scan_w * w)
        sh = int(self.regions.button_scan_h * h)
        scan = screenshot[sy : sy + sh, sx : sx + sw]

        # 多尺度 + 灰度双通道匹配，增强在缩放/亮度波动下的召回
        scan_gray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
        scales = (0.88, 0.94, 1.00, 1.06, 1.12)

        for name, tmpl_src in self.button_templates.items():
            best_score = -1.0
            best_loc = None
            best_wh = None

            for scale in scales:
                tmpl = tmpl_src
                th0, tw0 = tmpl.shape[:2]
                tw = max(8, int(tw0 * scale))
                th = max(8, int(th0 * scale))
                if tw != tw0 or th != th0:
                    tmpl = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_AREA)

                th, tw = tmpl.shape[:2]
                if th > sh or tw > sw:
                    continue
                if th > scan.shape[0] or tw > scan.shape[1]:
                    continue

                # BGR 匹配
                res_bgr = cv2.matchTemplate(scan, tmpl, cv2.TM_CCOEFF_NORMED)
                _, score_bgr, _, loc_bgr = cv2.minMaxLoc(res_bgr)

                # 灰度匹配（对色偏更稳）
                tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
                res_gray = cv2.matchTemplate(scan_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
                _, score_gray, _, loc_gray = cv2.minMaxLoc(res_gray)

                if score_gray >= score_bgr:
                    cur_score = float(score_gray)
                    cur_loc = loc_gray
                else:
                    cur_score = float(score_bgr)
                    cur_loc = loc_bgr

                if cur_score > best_score:
                    best_score = cur_score
                    best_loc = cur_loc
                    best_wh = (tw, th)

            # 操作按钮放宽阈值，荣和放宽幅度更小（保证精度）
            threshold = self.threshold
            if name in {"skip", "chi", "pon", "kan"}:
                threshold = max(0.60, self.threshold - 0.06)
            elif name == "ron":
                threshold = max(0.63, self.threshold - 0.04)

            if best_loc is not None and best_wh is not None and best_score >= threshold:
                tw, th = best_wh
                cx = (sx + best_loc[0] + tw // 2) / w
                cy = (sy + best_loc[1] + th // 2) / h
                found[name] = (cx, cy)
                scores[name] = float(best_score)
                logger.debug(
                    f"模板匹配按钮「{name}」: ({cx:.3f}, {cy:.3f}) 得分={best_score:.3f}"
                )

        return found, scores

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
    # 附加区域检测：牌堆 / 副露 / 宝牌
    # ------------------------------------------------------------------

    def _region_metrics(self, screenshot: np.ndarray, rx: float, ry: float,
                        rw: float, rh: float) -> Dict[str, float]:
        """提取归一化区域的亮度/纹理/边缘密度指标。"""
        ih, iw = screenshot.shape[:2]
        x1 = max(0, int(rx * iw))
        y1 = max(0, int(ry * ih))
        x2 = min(iw, int((rx + rw) * iw))
        y2 = min(ih, int((ry + rh) * ih))
        if x2 <= x1 or y2 <= y1:
            return {"brightness": 0.0, "texture": 0.0, "edge_density": 0.0}
        region = screenshot[y1:y2, x1:x2]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        return {
            "brightness": float(np.mean(gray)),
            "texture": float(np.std(gray)),
            "edge_density": float(np.mean(edges > 0)),
        }

    def _detect_wall_metrics(self, screenshot: np.ndarray) -> Dict[str, float]:
        """检测牌堆区域指标（亮/有纹理说明牌堆存在）。"""
        try:
            reg = self.regions
            return self._region_metrics(
                screenshot, reg.wall_x, reg.wall_y, reg.wall_w, reg.wall_h
            )
        except Exception as e:
            logger.debug(f"牌堆检测失败: {e}")
            return {}

    def _detect_meld_metrics(self, screenshot: np.ndarray) -> Dict[str, Dict[str, float]]:
        """检测四家副露区域指标。"""
        result: Dict[str, Dict[str, float]] = {}
        reg = self.regions
        meld_defs = {
            "self": (reg.meld_self_x, reg.meld_self_y, reg.meld_self_w, reg.meld_self_h),
            "right": (reg.meld_right_x, reg.meld_right_y, reg.meld_right_w, reg.meld_right_h),
            "opposite": (reg.meld_opposite_x, reg.meld_opposite_y, reg.meld_opposite_w, reg.meld_opposite_h),
            "left": (reg.meld_left_x, reg.meld_left_y, reg.meld_left_w, reg.meld_left_h),
        }
        for seat, (rx, ry, rw, rh) in meld_defs.items():
            try:
                result[seat] = self._region_metrics(screenshot, rx, ry, rw, rh)
            except Exception as e:
                logger.debug(f"副露区域检测失败({seat}): {e}")
                result[seat] = {}
        return result

    def _detect_current_player(self, screenshot: np.ndarray) -> int:
        """
        检测当前出牌家（通过中心黄色指示器）
        
        检测策略：在屏幕中心四个方向采样黄色亮度
        - 下方（6点钟方向）：自己 (0)
        - 右方（3点钟方向）：下家 (1) 
        - 上方（12点钟方向）：对家 (2)
        - 左方（9点钟方向）：上家 (3)
        
        Returns:
            int: 当前出牌家 (0-3)，-1表示未检测到
        """
        try:
            h, w = screenshot.shape[:2]
            reg = self.regions
            
            # 中心点和采样偏移
            cx = int(reg.indicator_center_x * w)
            cy = int(reg.indicator_center_y * h)
            offset = int(reg.indicator_sample_offset * w)  # 采样点距离
            sample_size = max(8, int(0.02 * w))  # 采样区域大小
            
            # 四个方向的采样点坐标（下、右、上、左）
            directions = {
                0: (cx, cy + offset),              # 下 - 自己
                1: (cx + offset, cy),              # 右 - 下家
                2: (cx, cy - offset),              # 上 - 对家  
                3: (cx - offset, cy),              # 左 - 上家
            }
            
            hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
            yellow_scores = {}
            
            for player_id, (px, py) in directions.items():
                # 裁剪采样区域
                x1 = max(0, px - sample_size // 2)
                y1 = max(0, py - sample_size // 2)
                x2 = min(w, px + sample_size // 2)
                y2 = min(h, py + sample_size // 2)
                
                if x2 <= x1 or y2 <= y1:
                    yellow_scores[player_id] = 0.0
                    continue
                
                region_hsv = hsv[y1:y2, x1:x2]
                
                # 黄色HSV范围：H在20-40（黄色），S>100，V>150
                h_channel = region_hsv[:, :, 0]
                s_channel = region_hsv[:, :, 1]
                v_channel = region_hsv[:, :, 2]
                
                # 黄色像素掩码
                yellow_mask = (
                    ((h_channel >= 20) & (h_channel <= 40)) &
                    (s_channel > 100) &
                    (v_channel > 150)
                )
                
                # 黄色得分：黄色像素占比 × 平均亮度
                yellow_ratio = float(np.mean(yellow_mask))
                avg_value = float(np.mean(v_channel[yellow_mask])) if np.any(yellow_mask) else 0.0
                yellow_scores[player_id] = yellow_ratio * avg_value
                
                logger.debug(
                    f"方向{player_id} 黄色检测: ratio={yellow_ratio:.3f} "
                    f"avg_v={avg_value:.1f} score={yellow_scores[player_id]:.1f}"
                )
            
            # 选择得分最高的方向
            if not yellow_scores or max(yellow_scores.values()) < 10.0:
                return -1  # 未检测到明显黄色
            
            current_player = max(yellow_scores, key=yellow_scores.get)
            logger.debug(f"当前出牌家: {current_player}")
            return current_player
            
        except Exception as e:
            logger.debug(f"检测当前出牌家失败: {e}")
            return -1

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

        # ── 1. 在“校准点附近”搜索最像手牌的区域（避免窗口边框/缩放造成偏移）──
        hand_w = max(1, int(reg.max_tiles * reg.tile_width * w))
        hand_h = max(1, int(reg.tile_height * h))

        best_metrics = None  # (score, x1, y1, mean_brightness, texture, ref_brightness, delta)
        for ox in self._HAND_OFFSET_X_CANDIDATES:
            for oy in self._HAND_OFFSET_Y_CANDIDATES:
                x1 = int((reg.x_start + ox) * w)
                y1 = int((reg.y_start + oy) * h)
                x2 = x1 + hand_w
                y2 = y1 + hand_h

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                hand_region = screenshot[y1:y2, x1:x2]
                if hand_region.size == 0:
                    continue

                gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
                mean_brightness = float(np.mean(gray))
                texture = float(np.std(gray))

                # 上方参考区域（同宽同高）
                ref_y1 = max(0, y1 - (y2 - y1))
                ref_y2 = y1
                if ref_y2 > ref_y1:
                    ref_region = screenshot[ref_y1:ref_y2, x1:x2]
                    ref_gray = cv2.cvtColor(ref_region, cv2.COLOR_BGR2GRAY)
                    ref_brightness = float(np.mean(ref_gray))
                else:
                    ref_brightness = mean_brightness

                brightness_delta = mean_brightness - ref_brightness

                # 倾向“亮且有纹理且相对上方更亮”的区域
                score = texture * 1.2 + brightness_delta * 0.9 + max(0.0, mean_brightness - 85.0) * 0.1

                if best_metrics is None or score > best_metrics[0]:
                    best_metrics = (
                        score,
                        x1,
                        y1,
                        mean_brightness,
                        texture,
                        ref_brightness,
                        brightness_delta,
                    )

        if best_metrics is None:
            return False, False

        _, best_x1, best_y1, mean_brightness, texture, ref_brightness, brightness_delta = best_metrics

        # 将像素偏移转换为相对偏移，供摸牌检测复用同一锚点
        x_offset_rel = (best_x1 / w) - reg.x_start
        y_offset_rel = (best_y1 / h) - reg.y_start

        # ── 2. 检测摸牌位置是否有牌（使用同一偏移锚点）──
        has_drawn = self._detect_drawn_tile(
            screenshot,
            x_offset_rel=x_offset_rel,
            y_offset_rel=y_offset_rel,
        )

        # 判定策略：
        # A) 摸牌位有牌 => 高优先判定为我的回合
        # B) 手牌绝对亮度高（历史规则）
        # C) 手牌亮度相对上方区域明显更亮 + 有足够纹理（自适应规则）
        is_my_turn = (
            has_drawn
            or mean_brightness > self.MY_TURN_BRIGHTNESS
            or (
                mean_brightness > self.MY_TURN_MIN_BRIGHTNESS
                and brightness_delta > self.MY_TURN_MIN_DELTA
                and texture > self.MY_TURN_MIN_TEXTURE
            )
        )

        logger.debug(
            f"手牌亮度={mean_brightness:.1f} 参考亮度={ref_brightness:.1f} "
            f"delta={brightness_delta:.1f} texture={texture:.1f} "
            f"offset=({x_offset_rel:+.3f},{y_offset_rel:+.3f}) "
            f"my_turn={is_my_turn} has_drawn={has_drawn}"
        )
        return is_my_turn, has_drawn

    def _detect_drawn_tile(
        self,
        screenshot: np.ndarray,
        x_offset_rel: float = 0.0,
        y_offset_rel: float = 0.0,
    ) -> bool:
        """
        检测摸牌区域（第 14 张位置）是否有牌

        通过检测该区域的亮度+纹理来判断
        """
        h, w = screenshot.shape[:2]
        reg = self.regions.hand

        drawn_y = int((reg.y_start + y_offset_rel) * h)
        tile_w = max(1, int(reg.tile_width * w))
        tile_h = max(1, int(reg.tile_height * h))

        if drawn_y >= h:
            return False

        # 兼容不同 drawn_gap 标定：
        # - 优先使用校准值
        # - 再尝试两个经验偏移（避免旧配置/异常配置导致完全失效）
        candidate_gap_rel = [
            float(reg.drawn_gap),
            float(reg.tile_width * 0.22),
            float(reg.tile_width * 0.45),
        ]

        # 参考：第13张手牌区域，用于与摸牌候选做“牌面相似度”比较
        ref_x = int((reg.x_start + x_offset_rel + (reg.max_tiles - 1) * reg.tile_width) * w)
        ref_x2 = min(ref_x + tile_w, w)
        ref_y2 = min(drawn_y + tile_h, h)
        ref_gray = None
        if ref_x < w and ref_x2 > ref_x and ref_y2 > drawn_y:
            ref_region = screenshot[drawn_y:ref_y2, ref_x:ref_x2]
            if ref_region.size > 0:
                ref_gray = cv2.cvtColor(ref_region, cv2.COLOR_BGR2GRAY)

        for gap_rel in candidate_gap_rel:
            drawn_x = int((reg.x_start + x_offset_rel + reg.max_tiles * reg.tile_width + gap_rel) * w)
            if drawn_x >= w:
                continue

            x_end = min(drawn_x + tile_w, w)
            y_end = min(drawn_y + tile_h, h)
            if x_end <= drawn_x or y_end <= drawn_y:
                continue

            region = screenshot[drawn_y:y_end, drawn_x:x_end]
            if region.size == 0:
                continue

            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))
            texture = float(np.std(gray))
            edges = cv2.Canny(gray, 60, 140)
            edge_ratio = float(np.mean(edges > 0))

            hist_corr = 0.0
            if ref_gray is not None and ref_gray.shape == gray.shape:
                hist_a = cv2.calcHist([ref_gray], [0], None, [32], [0, 256])
                hist_b = cv2.calcHist([gray], [0], None, [32], [0, 256])
                hist_a = cv2.normalize(hist_a, None).flatten()
                hist_b = cv2.normalize(hist_b, None).flatten()
                hist_corr = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))

            # 有牌判定：融合亮度/纹理/边缘/与第13张的分布相似度
            tile_like_votes = 0
            if brightness > 70:
                tile_like_votes += 1
            if texture > 10:
                tile_like_votes += 1
            if edge_ratio > 0.03:
                tile_like_votes += 1
            if hist_corr > 0.45:
                tile_like_votes += 1

            logger.debug(
                "摸牌候选检测: "
                f"gap_rel={gap_rel:.4f} brightness={brightness:.1f} "
                f"texture={texture:.1f} edge_ratio={edge_ratio:.3f} "
                f"hist_corr={hist_corr:.3f} votes={tile_like_votes}"
            )

            if tile_like_votes >= 3:
                return True

        return False

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
        cv2.putText(debug, "Hand", (hx1 + 2, hy1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

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
        cv2.putText(debug, "Buttons", (bx1 + 2, by1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 165, 255), 1)

        # 检测到的按钮（红圈，附带置信度）
        for btn_name, (rx, ry) in state.buttons.items():
            px, py = int(rx * w), int(ry * h)
            score = state.button_scores.get(btn_name, 0.0)
            cv2.circle(debug, (px, py), 22, (0, 0, 220), 3)
            cv2.putText(
                debug, f"{btn_name}({score:.2f})",
                (px - 28, py - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2,
            )

        # ── 新增区域框 ──
        def _draw_region(name: str, color: Tuple[int, int, int], label: str):
            try:
                rx, ry, rw, rh = self.regions.get_named_rect(name)
                rx1, ry1 = int(rx * w), int(ry * h)
                rx2, ry2 = int((rx + rw) * w), int((ry + rh) * h)
                cv2.rectangle(debug, (rx1, ry1), (rx2, ry2), color, 1)
                cv2.putText(debug, label, (rx1 + 2, ry1 + 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
            except Exception:
                pass

        _draw_region("dora",          (0, 220, 255),  "Dora")
        _draw_region("wall",          (180, 180, 60), "Wall")
        _draw_region("meld_self",     (200, 80, 255),  "Meld-S")
        _draw_region("meld_right",    (200, 80, 255),  "Meld-R")
        _draw_region("meld_opposite", (200, 80, 255),  "Meld-O")
        _draw_region("meld_left",     (200, 80, 255),  "Meld-L")

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
