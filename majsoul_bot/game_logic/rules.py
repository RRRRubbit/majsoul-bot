"""
规则判断模块
实现麻将规则的判断，包括听牌、和牌、番数计算等
"""
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter
from .tile import Tile, TileType
from .hand import Hand


class MahjongRules:
    """麻将规则判断类"""

    @staticmethod
    def is_winning_hand(hand: Hand) -> bool:
        """
        判断是否和牌

        Args:
            hand: 手牌对象

        Returns:
            bool: 是否和牌
        """
        return hand.is_winning()

    @staticmethod
    def is_tenpai(hand: Hand) -> bool:
        """
        判断是否听牌

        Args:
            hand: 手牌对象

        Returns:
            bool: 是否听牌
        """
        return hand.calculate_shanten() == 0

    @staticmethod
    def get_waiting_tiles(hand: Hand) -> List[Tile]:
        """
        获取听牌时等待的牌

        Args:
            hand: 手牌对象

        Returns:
            List[Tile]: 等待的牌列表
        """
        if not MahjongRules.is_tenpai(hand):
            return []

        waiting_tiles = []

        # 遍历所有可能的牌
        for tile_type in [TileType.MAN, TileType.PIN, TileType.SOU, TileType.HONOR]:
            max_value = 7 if tile_type == TileType.HONOR else 9
            for value in range(1, max_value + 1):
                test_tile = Tile(tile_type, value)

                # 临时添加这张牌
                hand.add_tile(test_tile)

                # 检查是否和牌
                if hand.is_winning():
                    waiting_tiles.append(test_tile)

                # 移除测试牌
                hand.remove_tile(test_tile)

        return waiting_tiles

    @staticmethod
    def calculate_han(hand: Hand, winning_tile: Tile,
                     is_tsumo: bool = False,
                     is_riichi: bool = False,
                     is_ippatsu: bool = False,
                     dora_count: int = 0) -> Tuple[int, List[str]]:
        """
        计算番数

        Args:
            hand: 手牌对象
            winning_tile: 和牌的牌
            is_tsumo: 是否自摸
            is_riichi: 是否立直
            is_ippatsu: 是否一发
            dora_count: 宝牌数量

        Returns:
            Tuple[int, List[str]]: (总番数, 役种列表)
        """
        han = 0
        yaku_list = []

        # 立直
        if is_riichi:
            han += 1
            yaku_list.append("立直")

        # 一发
        if is_ippatsu:
            han += 1
            yaku_list.append("一发")

        # 门前清自摸和
        if is_tsumo and len(hand.melds) == 0:
            han += 1
            yaku_list.append("门前清自摸和")

        # 断么九
        if MahjongRules._is_tanyao(hand):
            han += 1
            yaku_list.append("断么九")

        # 平和
        if MahjongRules._is_pinfu(hand):
            han += 1
            yaku_list.append("平和")

        # 一杯口
        if MahjongRules._is_iipeikou(hand):
            han += 1
            yaku_list.append("一杯口")

        # 三色同顺
        if MahjongRules._is_sanshoku(hand):
            han += 2 if len(hand.melds) == 0 else 1
            yaku_list.append("三色同顺")

        # 一气通贯
        if MahjongRules._is_ittsu(hand):
            han += 2 if len(hand.melds) == 0 else 1
            yaku_list.append("一气通贯")

        # 对对和
        if MahjongRules._is_toitoi(hand):
            han += 2
            yaku_list.append("对对和")

        # 三暗刻
        if MahjongRules._is_sanankou(hand):
            han += 2
            yaku_list.append("三暗刻")

        # 混全带么九
        if MahjongRules._is_chanta(hand):
            han += 2 if len(hand.melds) == 0 else 1
            yaku_list.append("混全带么九")

        # 七对子
        if hand.is_seven_pairs():
            han += 2
            yaku_list.append("七对子")

        # 混一色
        if MahjongRules._is_honitsu(hand):
            han += 3 if len(hand.melds) == 0 else 2
            yaku_list.append("混一色")

        # 纯全带么九
        if MahjongRules._is_junchan(hand):
            han += 3 if len(hand.melds) == 0 else 2
            yaku_list.append("纯全带么九")

        # 二杯口
        if MahjongRules._is_ryanpeikou(hand):
            han += 3
            yaku_list.append("二杯口")

        # 清一色
        if MahjongRules._is_chinitsu(hand):
            han += 6 if len(hand.melds) == 0 else 5
            yaku_list.append("清一色")

        # 国士无双
        if hand.is_kokushi():
            han += 13
            yaku_list.append("国士无双")

        # 宝牌
        if dora_count > 0:
            han += dora_count
            yaku_list.append(f"宝牌 x{dora_count}")

        return han, yaku_list

    @staticmethod
    def _is_tanyao(hand: Hand) -> bool:
        """断么九：所有牌都是中张牌（2-8）"""
        all_tiles = hand.tiles + [tile for meld in hand.melds for tile in meld]
        return all(tile.is_simple() for tile in all_tiles)

    @staticmethod
    def _is_pinfu(hand: Hand) -> bool:
        """平和：门前清，四组顺子，不是两面听"""
        if len(hand.melds) > 0:
            return False
        # 简化判断
        return True

    @staticmethod
    def _is_iipeikou(hand: Hand) -> bool:
        """一杯口：门前清，有一组相同的顺子"""
        if len(hand.melds) > 0:
            return False
        # 简化判断
        return False

    @staticmethod
    def _is_sanshoku(hand: Hand) -> bool:
        """三色同顺：三种花色有相同数字的顺子"""
        # 简化判断
        return False

    @staticmethod
    def _is_ittsu(hand: Hand) -> bool:
        """一气通贯：同一花色有 123、456、789"""
        # 简化判断
        return False

    @staticmethod
    def _is_toitoi(hand: Hand) -> bool:
        """对对和：四组刻子（或杠）"""
        all_tiles = hand.tiles + [tile for meld in hand.melds for tile in meld]
        counter = Counter((t.tile_type, t.value) for t in all_tiles)
        triplets = sum(1 for count in counter.values() if count >= 3)
        return triplets >= 4

    @staticmethod
    def _is_sanankou(hand: Hand) -> bool:
        """三暗刻：三组暗刻"""
        # 简化判断
        return False

    @staticmethod
    def _is_chanta(hand: Hand) -> bool:
        """混全带么九：每组面子都有么九牌，有字牌"""
        # 简化判断
        return False

    @staticmethod
    def _is_honitsu(hand: Hand) -> bool:
        """混一色：只有一种花色的数牌和字牌"""
        all_tiles = hand.tiles + [tile for meld in hand.melds for tile in meld]
        suit_types = set(t.tile_type for t in all_tiles if t.tile_type != TileType.HONOR)
        has_honor = any(t.is_honor() for t in all_tiles)
        return len(suit_types) == 1 and has_honor

    @staticmethod
    def _is_junchan(hand: Hand) -> bool:
        """纯全带么九：每组面子都有么九牌，无字牌"""
        # 简化判断
        return False

    @staticmethod
    def _is_ryanpeikou(hand: Hand) -> bool:
        """二杯口：门前清，有两组相同的顺子"""
        if len(hand.melds) > 0:
            return False
        # 简化判断
        return False

    @staticmethod
    def _is_chinitsu(hand: Hand) -> bool:
        """清一色：只有一种花色的数牌"""
        all_tiles = hand.tiles + [tile for meld in hand.melds for tile in meld]
        if any(t.is_honor() for t in all_tiles):
            return False
        suit_types = set(t.tile_type for t in all_tiles)
        return len(suit_types) == 1

    @staticmethod
    def can_chi(hand: Hand, tile: Tile, tiles_from_hand: List[Tile]) -> bool:
        """
        判断是否可以吃牌

        Args:
            hand: 手牌对象
            tile: 打出的牌
            tiles_from_hand: 用于吃牌的手牌

        Returns:
            bool: 是否可以吃
        """
        if tile.is_honor():
            return False

        if len(tiles_from_hand) != 2:
            return False

        # 检查是否能组成顺子
        all_tiles = [tile] + tiles_from_hand
        if len(set((t.tile_type, t.value) for t in all_tiles)) != 3:
            return False

        values = sorted([t.value for t in all_tiles])
        return values == [values[0], values[0] + 1, values[0] + 2]

    @staticmethod
    def can_pon(hand: Hand, tile: Tile) -> bool:
        """
        判断是否可以碰牌

        Args:
            hand: 手牌对象
            tile: 打出的牌

        Returns:
            bool: 是否可以碰
        """
        return hand.count_tile(tile) >= 2

    @staticmethod
    def can_kan(hand: Hand, tile: Tile) -> bool:
        """
        判断是否可以杠牌

        Args:
            hand: 手牌对象
            tile: 打出的牌或自己的牌

        Returns:
            bool: 是否可以杠
        """
        return hand.count_tile(tile) >= 3

    @staticmethod
    def can_riichi(hand: Hand) -> bool:
        """
        判断是否可以立直

        Args:
            hand: 手牌对象

        Returns:
            bool: 是否可以立直
        """
        # 门前清且听牌
        return len(hand.melds) == 0 and MahjongRules.is_tenpai(hand)

    @staticmethod
    def get_safe_tiles(hand: Hand, discarded_tiles: List[Tile]) -> List[Tile]:
        """
        获取相对安全的牌（简化版本）

        Args:
            hand: 手牌对象
            discarded_tiles: 已打出的牌

        Returns:
            List[Tile]: 安全的牌列表
        """
        # 简化：返回字牌和已经被打出很多的牌
        safe_tiles = []
        discarded_counter = Counter((t.tile_type, t.value) for t in discarded_tiles)

        for tile in hand.tiles:
            # 字牌相对安全
            if tile.is_honor():
                safe_tiles.append(tile)
            # 已经被打出 2 张以上的牌相对安全
            elif discarded_counter.get((tile.tile_type, tile.value), 0) >= 2:
                safe_tiles.append(tile)

        return safe_tiles
