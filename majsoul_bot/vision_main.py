"""
雀魂机器人 - 机器视觉版主程序

工作原理：
  1. 持续截取游戏窗口画面
  2. 用 OpenCV 模板匹配识别手牌与 UI 按钮
  3. 用原有 AI 逻辑决定打牌/操作
  4. 用 pyautogui 模拟鼠标点击执行操作

启动方式：
  python -m majsoul_bot.vision_main
  python majsoul_bot/vision_main.py
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from majsoul_bot.ai import SimpleAI
from majsoul_bot.config import Settings
from majsoul_bot.controller import MouseController
from majsoul_bot.game_logic import Hand, MahjongRules
from majsoul_bot.game_logic.tile import Tile
from majsoul_bot.utils import setup_logger, get_logger
from majsoul_bot.vision import (
    DEFAULT_REGIONS,
    DetectedState,
    GamePhase,
    GameStateDetector,
    ScreenCapture,
    TileRecognizer,
)


# ──────────────────────────────────────────────────────────────────────
# 机器人主类
# ──────────────────────────────────────────────────────────────────────

class VisionBot:
    """
    基于机器视觉的雀魂自动打牌机器人

    状态机：
    ┌─────────────────────────────────────────────────────┐
    │  WAITING  ←──────────────────────────────────────┐  │
    │     │                                             │  │
    │     ▼                                             │  │
    │  MY_TURN_DISCARD  → 识别手牌 → AI决策 → 点击打牌 │  │
    │                                                   │  │
    │  OPERATION_AVAILABLE  → AI决策碰/吃/杠/跳过     ─┘  │
    │                                                      │
    │  RIICHI_AVAILABLE  → AI决策是否立直                  │
    │                                                      │
    │  WIN_AVAILABLE  → 自动和牌                           │
    └─────────────────────────────────────────────────────┘
    """

    # 两次操作之间的最小冷却时间（秒），避免重复触发
    ACTION_COOLDOWN = 4.0

    # 打牌之后的额外等待时间（秒），等游戏状态切换才允许再次打牌
    # 防止：点击打牌 → 状态未变 → 反复点击同一张牌
    DISCARD_LOCK_TIMEOUT = 3.0

    # 主循环截图间隔（秒）
    CAPTURE_INTERVAL = 0.5

    def __init__(
        self,
        debug: bool = False,
        templates_dir: str = "templates",
        min_delay: float = 1.0,
        max_delay: float = 2.5,
        click_variance: int = 6,
        capture_interval: float = 0.5,
        tile_threshold: float = 0.75,
        button_threshold: float = 0.72,
        action_cooldown: float = 4.0,
        discard_lock_timeout: float = 3.0,
        nn_enabled: bool = True,
        nn_model_path: str = "models/tile_ann.xml",
        nn_labels_path: Optional[str] = None,
        nn_fusion_weight: float = 0.65,
        nn_min_confidence: float = 0.58,
        nn_top_k: int = 5,
        log_level: str = "INFO",
        log_file: str = "logs/vision_bot.log",
    ):
        """
        Args:
            debug: 是否开启调试模式（保存带标注的截图到 logs/）
            templates_dir: 模板根目录
            min_delay, max_delay: 操作延迟范围（秒）
            click_variance: 点击随机偏移像素
            capture_interval: 主循环截图间隔（秒）
            tile_threshold: 牌面模板匹配阈值
            button_threshold: 按钮模板匹配阈值
            action_cooldown: 两次操作间冷却时间（秒）
            discard_lock_timeout: 出牌后等待状态切换超时（秒）
            nn_enabled: 是否启用神经网络辅助识别
            nn_model_path: 神经网络模型路径
            nn_labels_path: 神经网络标签路径（None 时自动推断）
            nn_fusion_weight: 融合权重（越高越偏向 NN）
            nn_min_confidence: NN 兜底最小置信度
            nn_top_k: NN 候选数量
            log_level: 日志等级
            log_file: 日志文件路径
        """
        # ── 日志 ──
        os.makedirs("logs", exist_ok=True)
        setup_logger(log_level=log_level, log_file=log_file)
        self.logger = get_logger()

        self.debug = debug
        self.CAPTURE_INTERVAL = capture_interval
        self.ACTION_COOLDOWN = action_cooldown
        self.DISCARD_LOCK_TIMEOUT = discard_lock_timeout

        # ── 视觉组件 ──
        self.screen_capture = ScreenCapture()
        self.tile_recognizer = TileRecognizer(
            templates_dir=f"{templates_dir}/tiles",
            threshold=tile_threshold,
            nn_enabled=nn_enabled,
            nn_model_path=nn_model_path,
            nn_labels_path=nn_labels_path,
            nn_fusion_weight=nn_fusion_weight,
            nn_min_confidence=nn_min_confidence,
            nn_top_k=nn_top_k,
        )
        self.game_state_detector = GameStateDetector(
            templates_dir=f"{templates_dir}/buttons",
            threshold=button_threshold,
        )

        # ── 控制器 ──
        self.mouse = MouseController(
            min_delay=min_delay,
            max_delay=max_delay,
            click_variance=click_variance,
        )

        # ── 游戏逻辑 ──
        self.ai = SimpleAI()
        self.hand = Hand()

        # ── 运行状态 ──
        self.is_running = False
        self._last_action_time: float = 0.0
        self._last_phase: GamePhase = GamePhase.UNKNOWN
        self._debug_frame_count: int = 0

        # 打牌锁：True 表示已点击打牌，等待游戏状态切换为非 MY_TURN_DISCARD 后才允许再次出牌
        # 防止：手牌区域亮度始终 > 阈值 → 反复触发 my_turn_discard → 重复点击
        self._discard_pending: bool = False
        self._discard_pending_since: float = 0.0

        # 识别到的手牌信息 [(tile_name, (pix_x, pix_y)), ...]
        self._recognized_tiles: List[Tuple[str, Tuple[int, int]]] = []

    # ------------------------------------------------------------------
    # 启动 / 停止
    # ------------------------------------------------------------------

    async def start(self):
        """启动视觉机器人"""
        self.logger.info("=" * 60)
        self.logger.info("🀄 雀魂视觉机器人  (machine-vision mode)")
        self.logger.info("=" * 60)

        if not self.tile_recognizer.has_recognition_backend():
            self.logger.warning(
                "⚠  未检测到可用识别后端（模板 / NN）！\n"
                "   机器人将以「仅位置」模式运行（无法识别具体牌面）。\n"
                "   → 可先运行 python tools/capture_templates.py 捕获模板，\n"
                "     或训练并放置 models/tile_ann.xml 模型。"
            )
        elif not self.tile_recognizer.has_templates() and self.tile_recognizer.has_nn_model():
            self.logger.warning("⚠  未检测到模板，当前使用 NN-only 识别模式")

        # 尝试定位游戏窗口
        found = self.screen_capture.find_game_window()
        if found:
            w, h = self.screen_capture.window_size
            self.logger.info(f"✅ 游戏窗口已定位，尺寸 {w}×{h}")
        else:
            self.logger.warning("⚠  未找到游戏窗口，将捕获整个主显示器")

        self.logger.info(
            f"调试模式: {'ON' if self.debug else 'OFF'} | "
            f"模板: {len(self.tile_recognizer.templates)} 张牌 | "
            f"NN: {'ON' if self.tile_recognizer.has_nn_model() else 'OFF'}"
        )
        self.logger.info(
            "运行参数: "
            f"capture_interval={self.CAPTURE_INTERVAL:.2f}s, "
            f"tile_threshold={self.tile_recognizer.threshold:.2f}, "
            f"button_threshold={self.game_state_detector.threshold:.2f}, "
            f"nn_fusion={self.tile_recognizer.nn_fusion_weight:.2f}, "
            f"nn_min_conf={self.tile_recognizer.nn_min_confidence:.2f}, "
            f"cooldown={self.ACTION_COOLDOWN:.2f}s"
        )
        self.logger.info("按 Ctrl+C 停止机器人")
        self.logger.info("-" * 60)

        self.is_running = True
        self.ai.on_game_start()

        try:
            await self._main_loop()
        except KeyboardInterrupt:
            self.logger.info("收到停止信号（Ctrl+C）")
        except Exception as e:
            self.logger.error(f"机器人异常退出: {e}", exc_info=True)
        finally:
            self.is_running = False
            self.logger.info("机器人已停止。")

    async def stop(self):
        """请求停止"""
        self.is_running = False

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    async def _main_loop(self):
        """主循环：截图 → 检测 → 决策 → 操作"""
        while self.is_running:
            try:
                screenshot = self.screen_capture.capture()
                state = self.game_state_detector.detect(screenshot)

                # 记录状态变化
                if state.phase != self._last_phase:
                    self.logger.info(f"[状态] {self._last_phase.value} → {state.phase.value}")
                    self._last_phase = state.phase

                # 调试：每 20 帧保存一次标注截图
                if self.debug:
                    self._debug_frame_count += 1
                    if self._debug_frame_count % 20 == 0:
                        self._save_debug_frame(screenshot, state)

                # 根据状态执行操作（有冷却时间保护）
                await self._dispatch(screenshot, state)

            except Exception as e:
                self.logger.error(f"主循环异常: {e}", exc_info=False)

            await asyncio.sleep(self.CAPTURE_INTERVAL)

    # ------------------------------------------------------------------
    # 状态分发
    # ------------------------------------------------------------------

    async def _dispatch(self, screenshot, state: DetectedState):
        """根据当前游戏阶段分发处理函数"""
        phase = state.phase

        # ── 打牌锁管理（每帧运行，不受冷却限制）──
        # 游戏切换到非 MY_TURN_DISCARD 状态时立即解锁，
        # 确保下次轮到我出牌时能正常响应
        if self._discard_pending and phase != GamePhase.MY_TURN_DISCARD:
            self.logger.debug(f"[打牌锁] 状态切换为 {phase.value}，解锁")
            self._discard_pending = False

        # ── 全局冷却保护 ──
        if time.time() - self._last_action_time < self.ACTION_COOLDOWN:
            return

        if phase == GamePhase.WIN_AVAILABLE:
            await self._on_win(state)
        elif phase == GamePhase.RIICHI_AVAILABLE:
            await self._on_riichi(screenshot, state)
        elif phase == GamePhase.OPERATION_AVAILABLE:
            await self._on_operation(screenshot, state)
        elif phase == GamePhase.MY_TURN_DISCARD:
            # ── 打牌锁：已出牌但状态未切换时，跳过重复触发 ──
            if self._discard_pending:
                elapsed = time.time() - self._discard_pending_since
                if elapsed < self.DISCARD_LOCK_TIMEOUT:
                    self.logger.debug(
                        f"[打牌锁] 等待状态切换 ({elapsed:.1f}/{self.DISCARD_LOCK_TIMEOUT}s)"
                    )
                    return
                # 超时则强制解锁（游戏可能卡住或检测出错）
                self.logger.warning(
                    f"[打牌锁] 超时 {elapsed:.1f}s，强制解锁重试"
                )
                self._discard_pending = False
            await self._on_discard(screenshot, state)
        # WAITING / UNKNOWN / 其他阶段：静默等待

    # ------------------------------------------------------------------
    # 各阶段处理
    # ------------------------------------------------------------------

    async def _on_win(self, state: DetectedState):
        """处理和牌机会（自摸 > 荣和）"""
        self.logger.info("🎉 检测到和牌机会！")

        for btn in ("tsumo", "ron"):
            if btn in state.buttons:
                rel_x, rel_y = state.buttons[btn]
                label = "自摸" if btn == "tsumo" else "荣和"
                self.logger.info(f"  → 执行{label}")
                await self.mouse.click_relative(rel_x, rel_y, self.screen_capture)
                self._mark_action()
                self.ai.on_round_start()  # 重置 AI 立直状态
                return

    async def _on_riichi(self, screenshot, state: DetectedState):
        """处理立直机会"""
        # 先识别手牌（为了 AI 判断）
        self._update_hand(screenshot, state)

        if self.ai.decide_riichi(self.hand) and "riichi" in state.buttons:
            rel_x, rel_y = state.buttons["riichi"]
            self.logger.info("⚡ 立直！")
            await self.mouse.click_relative(rel_x, rel_y, self.screen_capture)
            self._mark_action()
        else:
            # 不立直则正常打牌
            await self._on_discard(screenshot, state)

    async def _on_operation(self, screenshot, state: DetectedState):
        """处理碰/吃/杠操作"""
        # 和牌机会优先
        if "ron" in state.buttons:
            await self._on_win(state)
            return

        self._update_hand(screenshot, state)
        buttons = state.buttons

        # ── 简单策略：除非有明显价值否则跳过 ──
        # 碰牌：如果手牌中有 2 张相同的字牌或三元牌，可能值得碰
        # 目前简化为：始终跳过（保持门前清，优先立直）
        self.logger.debug(f"可用操作: {list(buttons.keys())}，策略：跳过")

        # 点击跳过按钮
        skip_btn = self._find_skip_button(buttons)
        if skip_btn:
            rel_x, rel_y = skip_btn
            self.logger.info("⏭ 跳过操作")
            await self.mouse.click_relative(rel_x, rel_y, self.screen_capture)
            self._mark_action()
        else:
            self.logger.warning("找不到跳过按钮，等待超时")

    async def _on_discard(self, screenshot, state: DetectedState):
        """处理打牌阶段（核心决策）"""
        # 1. 识别手牌
        recognized = self.tile_recognizer.recognize_hand(
            screenshot,
            hand_count=13,
            has_drawn_tile=state.has_drawn_tile,
        )
        self._recognized_tiles = recognized

        if not recognized:
            self.logger.warning("手牌识别失败，等待下一帧")
            # 不重置冷却，允许下一帧立即重试
            return
        else:
            self.logger.info(f"识别到手牌: {[name for name, _ in recognized]}")
            self._log_candidate_tiles_for_unknown()
        # 2. 根据识别结果更新 AI 手牌
        drawn_tile = self._build_hand_from_recognized(recognized, state.has_drawn_tile)

        if self.hand.get_tile_count() == 0:
            # ── 无模板「位置模式」回退 ──
            # 没有牌面识别信息，但有像素位置，直接点击
            if self.tile_recognizer.has_recognition_backend():
                self.logger.warning(
                    "⚠  手牌未识别到可用牌型（模板/NN均未通过阈值，已输出候选牌型），"
                    "使用视觉聚类/位置模式打牌"
                )
            else:
                self.logger.info("⚠  无模板且无NN模型：使用视觉聚类/位置模式打牌")
            pos = self._pick_position_fallback(recognized, state.has_drawn_tile, screenshot)
            if pos:
                pix_x, pix_y = pos
                sh, sw = screenshot.shape[:2]
                await self.mouse.click_pixel(
                    pix_x, pix_y,
                    self.screen_capture,
                    screenshot_shape=(sh, sw),
                )
                self.logger.info(f"   → 位置模式点击: ({pix_x}, {pix_y})")
            else:
                self.logger.warning("   位置模式：找不到可点击的牌位，跳过")
            # 设置打牌锁：等待游戏状态切换才允许再次出牌
            self._discard_pending = True
            self._discard_pending_since = time.time()
            self._mark_action()
            return

        # 3. AI 决定打哪张牌
        tile_to_discard = self.ai.decide_discard(self.hand, drawn_tile)
        self.logger.info(
            f"🀄 打出: {tile_to_discard.get_display_name()} "
            f"（{tile_to_discard}） | 手牌 {self.hand.get_tile_count()}张"
        )

        # 4. 在识别结果中找到该牌的像素坐标并点击
        pos = self._find_tile_position(tile_to_discard, recognized)
        if pos is None:
            self.logger.warning(f"未找到牌 {tile_to_discard} 的屏幕位置，点击第一张")
            if recognized:
                pos = recognized[0][1]

        if pos:
            pix_x, pix_y = pos
            sh, sw = screenshot.shape[:2]
            await self.mouse.click_pixel(
                pix_x, pix_y,
                self.screen_capture,
                screenshot_shape=(sh, sw),
            )
            # 设置打牌锁：等待游戏状态切换才允许再次出牌
            self._discard_pending = True
            self._discard_pending_since = time.time()
            self._mark_action()

            # 5. 从 AI 手牌中移除打出的牌
            self.hand.remove_tile(tile_to_discard)
            shanten = self.hand.calculate_shanten()
            self.logger.info(f"   向听数: {shanten}")
        else:
            self.logger.error("无法确定点击位置")
            self._mark_action()  # 防止卡死

    def _log_candidate_tiles_for_unknown(self):
        """
        输出 unknown 牌位的候选牌型。

        依赖 TileRecognizer 在 recognize_hand() 后写入
        self.tile_recognizer.last_recognition_details。
        """
        details = getattr(self.tile_recognizer, "last_recognition_details", None)
        if not details:
            return

        unknown_details = [
            d for d in details if str(d.get("recognized_name", "")).startswith("unknown_")
        ]
        if not unknown_details:
            return

        self.logger.info(
            f"候选牌型（阈值={self.tile_recognizer.threshold:.2f}，仅展示未通过阈值的牌位）:"
        )
        for item in unknown_details:
            idx = int(item.get("index", -1))
            is_drawn = bool(item.get("is_drawn", False))
            best_score = float(item.get("best_score", 0.0))
            candidates = item.get("candidates", []) or []
            accept_reason = str(item.get("accept_reason", "below_threshold"))
            tpl_name = item.get("template_best_name")
            tpl_score = float(item.get("template_best_score", 0.0))
            nn_name = item.get("nn_best_name")
            nn_score = float(item.get("nn_best_score", 0.0))

            slot = "摸牌位" if is_drawn else f"手牌[{idx}]"
            if candidates:
                cand_text = ", ".join(
                    f"{name}({score:.3f})" for name, score in candidates
                )
            else:
                cand_text = "无可用候选"

            backend_info = (
                f"tmpl={tpl_name or '-'}({tpl_score:.3f}) "
                f"nn={nn_name or '-'}({nn_score:.3f})"
            )

            self.logger.info(
                f"   - {slot}: {cand_text} | best={best_score:.3f} | {backend_info} | reason={accept_reason}"
            )

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _update_hand(self, screenshot, state: DetectedState):
        """从截图更新 AI 手牌（不返回 drawn_tile）"""
        recognized = self.tile_recognizer.recognize_hand(
            screenshot,
            hand_count=13,
            has_drawn_tile=state.has_drawn_tile,
        )
        self._recognized_tiles = recognized
        self._build_hand_from_recognized(recognized, state.has_drawn_tile)

    def _build_hand_from_recognized(
        self,
        recognized: List[Tuple[str, Tuple[int, int]]],
        has_drawn_tile: bool,
    ) -> Optional[Tile]:
        """
        从识别结果构建 Hand 对象，并返回摸牌（若有）

        Returns:
            Optional[Tile]: 摸牌对象，无法识别时为 None
        """
        new_hand = Hand()
        drawn_tile: Optional[Tile] = None

        hand_count = 13
        for i, (name, _pos) in enumerate(recognized):
            # 跳过位置占位名
            if name.startswith(("unknown_", "pos_")):
                continue
            try:
                tile = Tile.from_string(name)
                if i >= hand_count:
                    drawn_tile = tile
                new_hand.tiles.append(tile)
            except (ValueError, AttributeError):
                self.logger.debug(f"跳过无效牌名: {name}")

        if new_hand.get_tile_count() > 0:
            self.hand = new_hand
        elif not self.tile_recognizer.has_templates():
            # 无模板时使用位置占位；手牌对象保持不变
            pass

        return drawn_tile

    def _find_tile_position(
        self,
        tile: Tile,
        recognized: List[Tuple[str, Tuple[int, int]]],
    ) -> Optional[Tuple[int, int]]:
        """
        在识别结果中查找指定牌的像素坐标

        匹配顺序：
        1. 完整匹配（含赤宝牌标记）
        2. 基础匹配（忽略赤宝标记）
        3. 无模板时按 AI 决策的索引估算位置
        """
        tile_str = str(tile)
        tile_base = f"{tile.value}{tile.tile_type.value}"

        # 精确匹配
        for name, pos in recognized:
            if name == tile_str:
                return pos

        # 宽松匹配（忽略 'r' 赤宝标记）
        for name, pos in recognized:
            if name.replace("r", "") == tile_base:
                return pos

        # 无法识别但有位置信息时，返回第一个非占位项
        for name, pos in recognized:
            if not name.startswith(("unknown_", "pos_")):
                return pos

        # 最后兜底：返回第一个位置
        if recognized:
            return recognized[0][1]

        return None

    def _pick_position_fallback(
        self,
        recognized: List[Tuple[str, Tuple[int, int]]],
        has_drawn_tile: bool,
        screenshot: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        无命名模板时的位置选牌策略：

        优先路径（有截图时）：
          使用视觉孤立度聚类，找出与其他牌最不相似的牌出打
          （近似"打孤张/没有搭子的牌"的效果）

        回退路径（无截图时）：
          - 有摸牌：摸切（打最后一张）
          - 无摸牌：打中间一张

        Args:
            recognized: tile_recognizer.recognize_hand() 返回的列表
            has_drawn_tile: 是否检测到了摸牌（第14张）
            screenshot: 原始截图（可选，用于视觉聚类）
        """
        if not recognized:
            return None

        # ── 视觉聚类路径 ──────────────────────────────────────
        if screenshot is not None:
            tile_imgs = self.tile_recognizer.extract_tile_images(
                screenshot,
                hand_count=13,
                has_drawn_tile=has_drawn_tile,
            )
            if tile_imgs:
                best_idx = self.tile_recognizer.find_best_discard_index(
                    tile_imgs, has_drawn_tile
                )
                if best_idx < len(recognized):
                    tag = "摸牌" if best_idx >= 13 else f"第{best_idx}张"
                    self.logger.info(
                        f"   🔍 视觉聚类选牌: {tag}（位置 {best_idx}）"
                    )
                    return recognized[best_idx][1]

        # ── 回退路径 ──────────────────────────────────────────
        if has_drawn_tile:
            return recognized[-1][1]
        else:
            mid = len(recognized) // 2
            return recognized[mid][1]

    def _find_skip_button(
        self, buttons: dict
    ) -> Optional[Tuple[float, float]]:
        """从按钮字典中找到"跳过"按钮坐标"""
        # 优先找明确的 skip 按钮
        if "skip" in buttons:
            return buttons["skip"]

        # 否则取最右边的按钮（通常是跳过/过）
        if buttons:
            rightmost = max(buttons.items(), key=lambda b: b[1][0])
            return rightmost[1]

        return None

    def _mark_action(self):
        """记录本次操作时间（冷却计时）"""
        self._last_action_time = time.time()

    def _save_debug_frame(self, screenshot, state: DetectedState):
        """保存带标注的调试截图"""
        try:
            debug_img = self.game_state_detector.visualize(screenshot, state)
            # 同时标注手牌识别区域
            debug_img = self.tile_recognizer.draw_hand_regions(
                debug_img, hand_count=13, has_drawn=state.has_drawn_tile
            )
            os.makedirs("logs", exist_ok=True)
            cv2.imwrite("logs/debug_latest.png", debug_img)
        except Exception as e:
            self.logger.debug(f"保存调试帧失败: {e}")


# ──────────────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="雀魂机器视觉机器人")
    p.add_argument(
        "--config",
        default="config/config.yaml",
        help="配置文件路径（不存在时自动使用内置默认值）",
    )

    debug_group = p.add_mutually_exclusive_group()
    debug_group.add_argument("--debug", dest="debug", action="store_true", help="开启调试模式（覆盖配置）")
    debug_group.add_argument("--no-debug", dest="debug", action="store_false", help="关闭调试模式（覆盖配置）")
    p.set_defaults(debug=None)

    p.add_argument("--templates", default=None, help="模板根目录（覆盖配置）")
    p.add_argument("--min-delay", type=float, default=None, help="操作最小延迟（秒，覆盖配置）")
    p.add_argument("--max-delay", type=float, default=None, help="操作最大延迟（秒，覆盖配置）")
    p.add_argument("--click-variance", type=int, default=None, help="点击偏移像素（覆盖配置）")

    p.add_argument("--capture-interval", type=float, default=None, help="主循环截图间隔（秒，覆盖配置）")
    p.add_argument("--tile-threshold", type=float, default=None, help="牌面模板匹配阈值（0~1，覆盖配置）")
    p.add_argument("--button-threshold", type=float, default=None, help="按钮模板匹配阈值（0~1，覆盖配置）")
    p.add_argument("--action-cooldown", type=float, default=None, help="操作冷却时长（秒，覆盖配置）")
    p.add_argument("--discard-lock-timeout", type=float, default=None, help="出牌锁超时时长（秒，覆盖配置）")

    nn_group = p.add_mutually_exclusive_group()
    nn_group.add_argument("--nn", dest="nn_enabled", action="store_true", help="启用 NN 识别（覆盖配置）")
    nn_group.add_argument("--no-nn", dest="nn_enabled", action="store_false", help="禁用 NN 识别（覆盖配置）")
    p.set_defaults(nn_enabled=None)

    p.add_argument("--nn-model-path", type=str, default=None, help="NN 模型路径（覆盖配置）")
    p.add_argument("--nn-labels-path", type=str, default=None, help="NN 标签路径（覆盖配置）")
    p.add_argument("--nn-fusion-weight", type=float, default=None, help="NN 融合权重（0~1，覆盖配置）")
    p.add_argument("--nn-min-confidence", type=float, default=None, help="NN 兜底最小置信度（0~1，覆盖配置）")
    p.add_argument("--nn-top-k", type=int, default=None, help="NN 候选数量（>=1，覆盖配置）")

    return p.parse_args()


def _load_settings_or_default(config_path: str) -> Settings:
    """
    加载配置文件；若不存在或格式错误则回退为默认配置。
    """
    try:
        settings = Settings.load_from_yaml(config_path)
        print(f"已加载配置文件: {config_path}")
        return settings
    except FileNotFoundError:
        print(f"未找到配置文件 {config_path}，使用内置默认参数运行。")
        return Settings()
    except ValueError as e:
        print(f"配置文件无效（{e}），使用内置默认参数运行。")
        return Settings()


def _resolve_option(cli_value, config_value):
    """命令行参数优先，其次配置文件值。"""
    return config_value if cli_value is None else cli_value


def _build_bot_from_args(args) -> VisionBot:
    settings = _load_settings_or_default(args.config)
    vision_cfg = settings.vision
    controller_cfg = settings.controller

    debug = _resolve_option(args.debug, vision_cfg.debug_mode)
    templates_dir = _resolve_option(args.templates, vision_cfg.templates_dir)
    min_delay = _resolve_option(args.min_delay, controller_cfg.min_delay)
    max_delay = _resolve_option(args.max_delay, controller_cfg.max_delay)
    click_variance = _resolve_option(args.click_variance, controller_cfg.click_variance)
    capture_interval = _resolve_option(args.capture_interval, vision_cfg.capture_interval)
    tile_threshold = _resolve_option(args.tile_threshold, vision_cfg.template_threshold)
    button_threshold = _resolve_option(args.button_threshold, vision_cfg.button_threshold)
    action_cooldown = _resolve_option(args.action_cooldown, vision_cfg.action_cooldown)
    discard_lock_timeout = _resolve_option(
        args.discard_lock_timeout,
        vision_cfg.discard_lock_timeout,
    )
    nn_enabled = _resolve_option(args.nn_enabled, vision_cfg.nn_enabled)
    nn_model_path = _resolve_option(args.nn_model_path, vision_cfg.nn_model_path)
    nn_labels_path = _resolve_option(args.nn_labels_path, vision_cfg.nn_labels_path)
    nn_fusion_weight = _resolve_option(args.nn_fusion_weight, vision_cfg.nn_fusion_weight)
    nn_min_confidence = _resolve_option(args.nn_min_confidence, vision_cfg.nn_min_confidence)
    nn_top_k = _resolve_option(args.nn_top_k, vision_cfg.nn_top_k)

    if isinstance(nn_labels_path, str) and not nn_labels_path.strip():
        nn_labels_path = None

    if min_delay > max_delay:
        raise ValueError("参数错误：min_delay 不能大于 max_delay")
    if click_variance < 0:
        raise ValueError("参数错误：click_variance 不能小于 0")
    if capture_interval <= 0:
        raise ValueError("参数错误：capture_interval 必须大于 0")
    if not (0.0 <= tile_threshold <= 1.0):
        raise ValueError("参数错误：tile_threshold 必须在 [0, 1] 区间")
    if not (0.0 <= button_threshold <= 1.0):
        raise ValueError("参数错误：button_threshold 必须在 [0, 1] 区间")
    if not (0.0 <= nn_fusion_weight <= 1.0):
        raise ValueError("参数错误：nn_fusion_weight 必须在 [0, 1] 区间")
    if not (0.0 <= nn_min_confidence <= 1.0):
        raise ValueError("参数错误：nn_min_confidence 必须在 [0, 1] 区间")
    if int(nn_top_k) < 1:
        raise ValueError("参数错误：nn_top_k 必须 >= 1")
    if action_cooldown < 0 or discard_lock_timeout < 0:
        raise ValueError("参数错误：冷却时间参数不能小于 0")

    return VisionBot(
        debug=debug,
        templates_dir=templates_dir,
        min_delay=min_delay,
        max_delay=max_delay,
        click_variance=click_variance,
        capture_interval=capture_interval,
        tile_threshold=tile_threshold,
        button_threshold=button_threshold,
        action_cooldown=action_cooldown,
        discard_lock_timeout=discard_lock_timeout,
        nn_enabled=bool(nn_enabled),
        nn_model_path=nn_model_path,
        nn_labels_path=nn_labels_path,
        nn_fusion_weight=float(nn_fusion_weight),
        nn_min_confidence=float(nn_min_confidence),
        nn_top_k=int(nn_top_k),
        log_level=settings.logging.level,
        log_file=settings.logging.file,
    )


def main():
    args = parse_args()

    try:
        bot = _build_bot_from_args(args)
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        print("\n已停止")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
