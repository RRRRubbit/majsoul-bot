"""
手牌管理模块
管理玩家的手牌，提供添加、删除、排序等功能
"""
from typing import List, Dict, Optional, Tuple
from collections import Counter
from .tile import Tile, TileType


class Hand:
    """
    手牌类

    Attributes:
        tiles: 手牌列表
        melds: 副露列表（吃、碰、杠）
    """

    def __init__(self, tiles: Optional[List[Tile]] = None):
        """
        初始化手牌

        Args:
            tiles: 初始牌列表
        """
        self.tiles: List[Tile] = tiles if tiles else []
        self.melds: List[List[Tile]] = []  # 副露的牌组

    def add_tile(self, tile: Tile):
        """
        添加一张牌到手牌

        Args:
            tile: 要添加的牌
        """
        self.tiles.append(tile)
        self.sort()

    def remove_tile(self, tile: Tile) -> bool:
        """
        从手牌中移除一张牌

        Args:
            tile: 要移除的牌

        Returns:
            bool: 是否成功移除
        """
        for i, t in enumerate(self.tiles):
            if t == tile:
                self.tiles.pop(i)
                return True
        return False

    def sort(self):
        """按类型和数值排序手牌"""
        self.tiles.sort()

    def get_tile_count(self) -> int:
        """
        获取手牌数量

        Returns:
            int: 手牌总数（不包括副露）
        """
        return len(self.tiles)

    def get_total_count(self) -> int:
        """
        获取所有牌的数量

        Returns:
            int: 手牌 + 副露的总数
        """
        meld_count = sum(len(meld) for meld in self.melds)
        return len(self.tiles) + meld_count

    def get_tile_counter(self) -> Counter:
        """
        获取手牌的计数器

        Returns:
            Counter: 每种牌的数量
        """
        return Counter((tile.tile_type, tile.value) for tile in self.tiles)

    def has_tile(self, tile: Tile) -> bool:
        """
        检查是否拥有某张牌

        Args:
            tile: 要检查的牌

        Returns:
            bool: 是否拥有
        """
        return tile in self.tiles

    def count_tile(self, tile: Tile) -> int:
        """
        统计某张牌的数量

        Args:
            tile: 要统计的牌

        Returns:
            int: 牌的数量
        """
        return sum(1 for t in self.tiles if t == tile)

    def add_meld(self, meld: List[Tile]):
        """
        添加一个副露

        Args:
            meld: 副露的牌组（顺子、刻子或杠）
        """
        self.melds.append(meld)

    def clear(self):
        """清空手牌和副露"""
        self.tiles.clear()
        self.melds.clear()

    def get_groups(self) -> Dict[TileType, List[int]]:
        """
        按类型分组手牌

        Returns:
            Dict: 每种类型的牌值列表
        """
        groups = {
            TileType.MAN: [],
            TileType.PIN: [],
            TileType.SOU: [],
            TileType.HONOR: []
        }

        for tile in self.tiles:
            groups[tile.tile_type].append(tile.value)

        # 对每组进行排序
        for tile_type in groups:
            groups[tile_type].sort()

        return groups

    def calculate_shanten(self) -> int:
        """
        计算向听数（简化版本）

        Returns:
            int: 向听数（-1 表示和牌，0 表示听牌，>0 表示距离听牌的步数）

        Note:
            这是一个简化的向听数计算，完整实现需要考虑所有牌型
        """
        # 检查是否和牌
        if self.is_winning():
            return -1

        # 简化计算：基于对子、顺子、刻子的数量
        tile_counter = self.get_tile_counter()
        pairs = 0
        melds = len(self.melds)
        potential_melds = 0

        # 统计对子和刻子
        for tile_key, count in tile_counter.items():
            if count >= 2:
                pairs += 1
            if count >= 3:
                potential_melds += 1

        # 统计顺子（简化：只检查连续数字）
        groups = self.get_groups()
        for tile_type, values in groups.items():
            if tile_type != TileType.HONOR:
                value_set = set(values)
                for v in value_set:
                    if v + 1 in value_set and v + 2 in value_set:
                        potential_melds += 1

        # 简化的向听数估算
        total_melds = melds + potential_melds
        has_pair = pairs > 0

        if total_melds >= 4 and has_pair:
            return 0  # 听牌
        elif total_melds >= 3 and has_pair:
            return 1
        elif total_melds >= 2:
            return 2
        else:
            return 3

    def is_winning(self) -> bool:
        """
        检查是否和牌（简化版本）

        Returns:
            bool: 是否和牌

        Note:
            完整实现需要在 rules.py 中进行详细判断
        """
        # 基本检查：牌数是否正确
        if self.get_tile_count() % 3 != 2:
            return False

        # 检查特殊和牌型（七对子、国士无双）
        if self.is_seven_pairs():
            return True
        if self.is_kokushi():
            return True

        # 检查标准和牌（4 面子 + 1 雀头）
        return self._check_standard_winning()

    def is_seven_pairs(self) -> bool:
        """
        检查是否为七对子

        Returns:
            bool: 是否为七对子
        """
        if len(self.tiles) != 14 or len(self.melds) > 0:
            return False

        tile_counter = self.get_tile_counter()
        pairs = sum(1 for count in tile_counter.values() if count == 2)
        return pairs == 7

    def is_kokushi(self) -> bool:
        """
        检查是否为国士无双

        Returns:
            bool: 是否为国士无双
        """
        if len(self.tiles) != 14 or len(self.melds) > 0:
            return False

        # 国士无双需要的牌：1m 9m 1p 9p 1s 9s + 7种字牌
        required_tiles = [
            (TileType.MAN, 1), (TileType.MAN, 9),
            (TileType.PIN, 1), (TileType.PIN, 9),
            (TileType.SOU, 1), (TileType.SOU, 9),
            (TileType.HONOR, 1), (TileType.HONOR, 2), (TileType.HONOR, 3),
            (TileType.HONOR, 4), (TileType.HONOR, 5), (TileType.HONOR, 6),
            (TileType.HONOR, 7)
        ]

        tile_counter = self.get_tile_counter()

        # 检查是否有所有 13 种牌，且其中一种有 2 张
        for tile_key in required_tiles:
            if tile_key not in tile_counter:
                return False

        # 应该正好有一种牌是 2 张，其余都是 1 张
        counts = [tile_counter[tile_key] for tile_key in required_tiles]
        return counts.count(2) == 1 and counts.count(1) == 12

    def _check_standard_winning(self) -> bool:
        """
        检查标准和牌（4 面子 + 1 雀头）

        Returns:
            bool: 是否为标准和牌
        """
        # 这是一个简化实现，完整的和牌判断算法较复杂
        # 需要递归检查所有可能的面子组合

        tile_counter = self.get_tile_counter()

        # 尝试每种牌作为雀头
        for pair_tile in tile_counter:
            if tile_counter[pair_tile] >= 2:
                # 移除雀头
                temp_counter = tile_counter.copy()
                temp_counter[pair_tile] -= 2

                # 检查剩余牌是否能组成 4 个面子
                if self._check_melds(temp_counter, 4 - len(self.melds)):
                    return True

        return False

    def _check_melds(self, counter: Counter, needed: int) -> bool:
        """
        递归检查是否能组成指定数量的面子

        Args:
            counter: 牌的计数器
            needed: 需要的面子数量

        Returns:
            bool: 是否可以组成
        """
        if needed == 0:
            return all(count == 0 for count in counter.values())

        # 简化实现：只检查刻子
        for tile_key in list(counter.keys()):
            if counter[tile_key] >= 3:
                temp_counter = counter.copy()
                temp_counter[tile_key] -= 3
                if self._check_melds(temp_counter, needed - 1):
                    return True

        return False

    def __str__(self) -> str:
        """字符串表示"""
        hand_str = ' '.join(str(tile) for tile in sorted(self.tiles))
        if self.melds:
            meld_strs = [' '.join(str(tile) for tile in meld) for meld in self.melds]
            return f"Hand: {hand_str} | Melds: {', '.join(meld_strs)}"
        return f"Hand: {hand_str}"

    def __repr__(self) -> str:
        """详细表示"""
        return f"Hand(tiles={len(self.tiles)}, melds={len(self.melds)})"
