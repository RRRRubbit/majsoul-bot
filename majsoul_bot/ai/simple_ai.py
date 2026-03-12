"""
简单 AI 实现模块
提供基础的打牌策略
"""
import random
from typing import List, Optional, Tuple
from loguru import logger
from .strategy import Strategy
from ..game_logic import Tile, Hand, MahjongRules


class SimpleAI(Strategy):
    """
    简单 AI 策略

    基础策略：
    1. 优先打字牌
    2. 打边张和孤张
    3. 听牌时自动和牌
    4. 不吃牌（保持门前清）
    5. 有刻子时考虑碰牌
    """

    def __init__(self):
        """初始化简单 AI"""
        super().__init__(name="SimpleAI")
        self.is_riichi = False  # 是否已立直
        self.discarded_tiles: List[Tile] = []  # 已打出的牌

    def decide_discard(self, hand: Hand, drawn_tile: Optional[Tile] = None) -> Tile:
        """
        决定打出哪张牌

        策略：
        1. 如果已立直，打刚摸的牌
        2. 优先打字牌（风牌和三元牌）
        3. 打边张牌（1, 9）
        4. 打孤张（周围没有相邻牌的牌）
        5. 随机打一张

        Args:
            hand: 当前手牌
            drawn_tile: 刚摸到的牌

        Returns:
            Tile: 决定打出的牌
        """
        if self.is_riichi and drawn_tile:
            logger.info(f"立直状态，打刚摸的牌: {drawn_tile}")
            return drawn_tile

        # 优先打字牌
        honor_tiles = [tile for tile in hand.tiles if tile.is_honor()]
        if honor_tiles:
            # 打出数量最少的字牌
            tile_to_discard = self._find_least_useful_tile(hand, honor_tiles)
            logger.info(f"打字牌: {tile_to_discard}")
            return tile_to_discard

        # 打边张牌
        terminal_tiles = [tile for tile in hand.tiles if tile.is_terminal()]
        if terminal_tiles:
            tile_to_discard = self._find_least_useful_tile(hand, terminal_tiles)
            logger.info(f"打边张: {tile_to_discard}")
            return tile_to_discard

        # 打孤张
        isolated_tiles = self._find_isolated_tiles(hand)
        if isolated_tiles:
            tile_to_discard = random.choice(isolated_tiles)
            logger.info(f"打孤张: {tile_to_discard}")
            return tile_to_discard

        # 如果有刚摸的牌，优先考虑打出
        if drawn_tile and drawn_tile in hand.tiles:
            logger.info(f"打刚摸的牌: {drawn_tile}")
            return drawn_tile

        # 随机打一张
        tile_to_discard = random.choice(hand.tiles)
        logger.info(f"随机打牌: {tile_to_discard}")
        return tile_to_discard

    def _find_least_useful_tile(self, hand: Hand, candidates: List[Tile]) -> Tile:
        """
        从候选牌中找出最没用的一张

        Args:
            hand: 当前手牌
            candidates: 候选牌列表

        Returns:
            Tile: 最没用的牌
        """
        tile_counter = hand.get_tile_counter()

        # 找出数量最少的牌
        min_count = float('inf')
        least_useful = candidates[0]

        for tile in candidates:
            key = (tile.tile_type, tile.value)
            count = tile_counter.get(key, 0)
            if count < min_count:
                min_count = count
                least_useful = tile

        return least_useful

    def _find_isolated_tiles(self, hand: Hand) -> List[Tile]:
        """
        找出手牌中的孤张（周围没有相邻牌的牌）

        Args:
            hand: 当前手牌

        Returns:
            List[Tile]: 孤张列表
        """
        isolated = []
        groups = hand.get_groups()

        for tile in hand.tiles:
            if tile.is_honor():
                # 字牌看数量
                if hand.count_tile(tile) == 1:
                    isolated.append(tile)
            else:
                # 数牌看是否有相邻牌
                values = groups[tile.tile_type]
                has_neighbor = False

                for v in values:
                    if abs(v - tile.value) == 1 or abs(v - tile.value) == 2:
                        has_neighbor = True
                        break

                if not has_neighbor and hand.count_tile(tile) == 1:
                    isolated.append(tile)

        return isolated

    def decide_chi(self, hand: Hand, tile: Tile, available_combinations: List[Tuple[Tile, Tile]]) -> Optional[Tuple[Tile, Tile]]:
        """
        决定是否吃牌

        简单策略：不吃牌，保持门前清

        Args:
            hand: 当前手牌
            tile: 可以吃的牌
            available_combinations: 可用的组合

        Returns:
            None: 不吃牌
        """
        logger.debug("SimpleAI 不吃牌（保持门前清）")
        return None

    def decide_pon(self, hand: Hand, tile: Tile) -> bool:
        """
        决定是否碰牌

        策略：
        1. 如果已立直，不碰
        2. 如果有 3 张同样的牌，考虑碰（保留刻子）
        3. 字牌优先碰

        Args:
            hand: 当前手牌
            tile: 可以碰的牌

        Returns:
            bool: 是否碰牌
        """
        if self.is_riichi:
            return False

        # 检查是否有 3 张
        if hand.count_tile(tile) >= 3:
            logger.info(f"碰牌（有刻子）: {tile}")
            return True

        # 字牌且有 2 张时考虑碰
        if tile.is_honor() and hand.count_tile(tile) >= 2:
            logger.info(f"碰字牌: {tile}")
            return True

        return False

    def decide_kan(self, hand: Hand, tile: Tile, is_ankan: bool = False) -> bool:
        """
        决定是否杠牌

        策略：简单起见，不杠牌（避免风险）

        Args:
            hand: 当前手牌
            tile: 可以杠的牌
            is_ankan: 是否为暗杠

        Returns:
            bool: 是否杠牌
        """
        logger.debug("SimpleAI 不杠牌")
        return False

    def decide_riichi(self, hand: Hand) -> bool:
        """
        决定是否立直

        策略：
        1. 门前清
        2. 听牌
        3. 有一定的和牌机会（简化：总是立直）

        Args:
            hand: 当前手牌

        Returns:
            bool: 是否立直
        """
        if self.is_riichi:
            return False

        # 检查是否可以立直
        if MahjongRules.can_riichi(hand):
            waiting_tiles = MahjongRules.get_waiting_tiles(hand)
            if len(waiting_tiles) > 0:
                logger.info(f"决定立直，等待: {[str(t) for t in waiting_tiles]}")
                self.is_riichi = True
                return True

        return False

    def decide_ron(self, hand: Hand, tile: Tile) -> bool:
        """
        决定是否荣和

        策略：只要能和就和

        Args:
            hand: 当前手牌
            tile: 可以荣和的牌

        Returns:
            bool: 是否荣和
        """
        # 添加牌并检查是否和牌
        hand.add_tile(tile)
        is_winning = hand.is_winning()
        hand.remove_tile(tile)

        if is_winning:
            logger.info(f"荣和: {tile}")
            return True

        return False

    def decide_tsumo(self, hand: Hand) -> bool:
        """
        决定是否自摸和

        策略：只要能和就和

        Args:
            hand: 当前手牌

        Returns:
            bool: 是否自摸和
        """
        if hand.is_winning():
            logger.info("自摸和")
            return True

        return False

    def on_game_start(self):
        """游戏开始时重置状态"""
        self.is_riichi = False
        self.discarded_tiles.clear()
        logger.info(f"{self.name} 游戏开始")

    def on_game_end(self):
        """游戏结束"""
        logger.info(f"{self.name} 游戏结束")

    def on_round_start(self):
        """新局开始时重置立直状态"""
        self.is_riichi = False
        logger.info(f"{self.name} 新局开始")

    def on_round_end(self):
        """局结束"""
        logger.info(f"{self.name} 局结束")

    def update_game_state(self, **kwargs):
        """
        更新游戏状态

        Args:
            **kwargs: 游戏状态参数
        """
        # 更新已打出的牌
        if "discarded_tile" in kwargs:
            tile = kwargs["discarded_tile"]
            if tile:
                self.discarded_tiles.append(tile)
