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
    ):
        """
        Args:
            debug: 是否开启调试模式（保存带标注的截图到 logs/）
            templates_dir: 模板根目录
            min_delay, max_delay: 操作延迟范围（秒）
        """
        # ── 日志 ──
        os.makedirs("logs", exist_ok=True)
        setup_logger(log_level="INFO", log_file="logs/vision_bot.log")
        self.logger = get_logger()

        self.debug = debug

        # ── 视觉组件 ──
        self.screen_capture = ScreenCapture()
        self.tile_recognizer = TileRecognizer(
            templates_dir=f"{templates_dir}/tiles"
        )
        self.game_state_detector = GameStateDetector(
            templates_dir=f"{templates_dir}/buttons"
        )

        # ── 控制器 ──
        self.mouse = MouseController(
            min_delay=min_delay,
            max_delay=max_delay,
            click_variance=6,
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

        if not self.tile_recognizer.has_templates():
            self.logger.warning(
                "⚠  未检测到牌型模板！\n"
                "   机器人将以「仅位置」模式运行（无法识别具体牌面）。\n"
                "   → 运行 python tools/capture_templates.py 来生成模板。"
            )

        # 尝试定位游戏窗口
        found = self.screen_capture.find_game_window()
        if found:
            w, h = self.screen_capture.window_size
            self.logger.info(f"✅ 游戏窗口已定位，尺寸 {w}×{h}")
        else:
            self.logger.warning("⚠  未找到游戏窗口，将捕获整个主显示器")

        self.logger.info(
            f"调试模式: {'ON' if self.debug else 'OFF'} | "
            f"模板: {len(self.tile_recognizer.templates)} 张牌"
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
        # 2. 根据识别结果更新 AI 手牌
        drawn_tile = self._build_hand_from_recognized(recognized, state.has_drawn_tile)

        if self.hand.get_tile_count() == 0:
            # ── 无模板「位置模式」回退 ──
            # 没有牌面识别信息，但有像素位置，直接点击
            self.logger.info("⚠  无模板：使用视觉聚类/位置模式打牌")
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
    p.add_argument("--debug", action="store_true", help="开启调试模式（保存标注截图）")
    p.add_argument("--templates", default="templates", help="模板根目录（默认：templates）")
    p.add_argument("--min-delay", type=float, default=1.0, help="操作最小延迟（秒）")
    p.add_argument("--max-delay", type=float, default=2.5, help="操作最大延迟（秒）")
    return p.parse_args()


def main():
    args = parse_args()
    bot = VisionBot(
        debug=args.debug,
        templates_dir=args.templates,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
    )
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        print("\n已停止")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
