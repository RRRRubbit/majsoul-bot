"""
麻将牌定义模块
定义麻将牌的类型、表示和转换方法
"""
from enum import Enum
from typing import List, Optional, Tuple


class TileType(Enum):
    """牌的类型"""
    MAN = "m"  # 万子 (characters)
    PIN = "p"  # 筒子 (circles)
    SOU = "s"  # 索子 (bamboos)
    HONOR = "z"  # 字牌 (honors: winds + dragons)


class Tile:
    """
    麻将牌类

    Attributes:
        tile_type: 牌的类型
        value: 牌的数值 (1-9 for suits, 1-7 for honors)
        aka: 是否为红宝牌 (赤ドラ)
    """

    def __init__(self, tile_type: TileType, value: int, aka: bool = False):
        """
        初始化麻将牌

        Args:
            tile_type: 牌的类型
            value: 牌的数值
            aka: 是否为红宝牌

        Raises:
            ValueError: 牌的数值不合法
        """
        self.tile_type = tile_type
        self.aka = aka

        # 验证数值范围
        if tile_type == TileType.HONOR:
            if not 1 <= value <= 7:
                raise ValueError(f"字牌数值必须在 1-7 之间: {value}")
        else:
            if not 1 <= value <= 9:
                raise ValueError(f"数牌数值必须在 1-9 之间: {value}")

        self.value = value

    def __str__(self) -> str:
        """字符串表示"""
        aka_mark = "r" if self.aka else ""
        return f"{self.value}{aka_mark}{self.tile_type.value}"

    def __repr__(self) -> str:
        """详细表示"""
        return f"Tile({self.tile_type.name}, {self.value}, aka={self.aka})"

    def __eq__(self, other) -> bool:
        """相等比较（忽略赤宝牌标记）"""
        if not isinstance(other, Tile):
            return False
        return self.tile_type == other.tile_type and self.value == other.value

    def __hash__(self) -> int:
        """哈希值（用于集合和字典）"""
        return hash((self.tile_type, self.value))

    def __lt__(self, other) -> bool:
        """小于比较（用于排序）"""
        if not isinstance(other, Tile):
            return NotImplemented
        # 按类型排序: 万 < 筒 < 索 < 字
        type_order = {"m": 0, "p": 1, "s": 2, "z": 3}
        if self.tile_type.value != other.tile_type.value:
            return type_order[self.tile_type.value] < type_order[other.tile_type.value]
        return self.value < other.value

    def is_terminal(self) -> bool:
        """是否为么九牌（1 或 9）"""
        return self.tile_type != TileType.HONOR and self.value in [1, 9]

    def is_honor(self) -> bool:
        """是否为字牌"""
        return self.tile_type == TileType.HONOR

    def is_wind(self) -> bool:
        """是否为风牌（东南西北）"""
        return self.tile_type == TileType.HONOR and 1 <= self.value <= 4

    def is_dragon(self) -> bool:
        """是否为三元牌（白发中）"""
        return self.tile_type == TileType.HONOR and 5 <= self.value <= 7

    def is_yaochuhai(self) -> bool:
        """是否为么九牌（包括字牌）"""
        return self.is_terminal() or self.is_honor()

    def is_simple(self) -> bool:
        """是否为中张牌（2-8 的数牌）"""
        return self.tile_type != TileType.HONOR and 2 <= self.value <= 8

    def get_display_name(self) -> str:
        """
        获取牌的显示名称

        Returns:
            str: 牌的中文名称
        """
        if self.tile_type == TileType.MAN:
            return f"{self.value}万"
        elif self.tile_type == TileType.PIN:
            return f"{self.value}筒"
        elif self.tile_type == TileType.SOU:
            return f"{self.value}索"
        elif self.tile_type == TileType.HONOR:
            honor_names = {
                1: "东", 2: "南", 3: "西", 4: "北",
                5: "白", 6: "发", 7: "中"
            }
            return honor_names[self.value]
        return str(self)

    @classmethod
    def from_string(cls, tile_str: str) -> "Tile":
        """
        从字符串创建牌

        Args:
            tile_str: 牌的字符串表示，如 "1m", "5rp", "7z"

        Returns:
            Tile: 牌对象

        Raises:
            ValueError: 字符串格式错误
        """
        tile_str = tile_str.strip().lower()

        if len(tile_str) < 2:
            raise ValueError(f"无效的牌字符串: {tile_str}")

        # 检查是否为赤宝牌
        aka = 'r' in tile_str
        tile_str = tile_str.replace('r', '')

        # 兼容赤宝简写：0m/0p/0s -> 5m/5p/5s 且 aka=True
        if len(tile_str) == 2 and tile_str[0] == '0' and tile_str[1] in {'m', 'p', 's'}:
            tile_str = f"5{tile_str[1]}"
            aka = True

        # 解析数值和类型
        try:
            value = int(tile_str[0])
            tile_type_str = tile_str[1]
            tile_type = TileType(tile_type_str)
        except (ValueError, IndexError):
            raise ValueError(f"无效的牌字符串: {tile_str}")

        return cls(tile_type, value, aka)


def parse_tiles(tiles_str: str) -> List[Tile]:
    """
    从字符串解析多张牌

    支持格式：
    - "123m406p789s1234567z" (紧凑格式，0 表示赤五)
    - "1m 2m 3m 4p 5p 6p" (空格分隔)

    Args:
        tiles_str: 牌的字符串表示

    Returns:
        List[Tile]: 牌列表
    """
    tiles = []
    tiles_str = tiles_str.strip()

    # 空格分隔格式
    if ' ' in tiles_str:
        for tile_str in tiles_str.split():
            tiles.append(Tile.from_string(tile_str))
        return tiles

    # 紧凑格式
    current_values: List[Tuple[int, bool]] = []
    for char in tiles_str:
        if char.isdigit():
            val = int(char)
            if val == 0:
                # 紧凑格式中的 0 表示赤宝 5（仅 m/p/s 合法）
                current_values.append((5, True))
            else:
                current_values.append((val, False))
        elif char in ['m', 'p', 's', 'z']:
            tile_type = TileType(char)
            for value, aka in current_values:
                # 字牌不支持 0z（即 aka=True 的 z）
                tiles.append(Tile(tile_type, value, aka=(aka and tile_type != TileType.HONOR)))
            current_values = []
        elif char == 'r':
            # 赤宝牌标记，下一个数字是赤宝
            continue
        else:
            raise ValueError(f"无效的字符: {char}")

    return tiles


def tiles_to_string(tiles: List[Tile], compact: bool = True) -> str:
    """
    将牌列表转换为字符串

    Args:
        tiles: 牌列表
        compact: 是否使用紧凑格式

    Returns:
        str: 牌的字符串表示
    """
    if not compact:
        return ' '.join(str(tile) for tile in tiles)

    # 紧凑格式：按类型分组
    sorted_tiles = sorted(tiles)
    result = []
    current_type = None
    current_values = []

    for tile in sorted_tiles:
        if tile.tile_type != current_type:
            if current_values:
                result.append(''.join(map(str, current_values)) + current_type.value)
            current_type = tile.tile_type
            current_values = []
        current_values.append(tile.value)

    if current_values:
        result.append(''.join(map(str, current_values)) + current_type.value)

    return ''.join(result)


def create_all_tiles() -> List[Tile]:
    """
    创建所有的麻将牌（136 张）

    Returns:
        List[Tile]: 包含所有牌的列表
    """
    tiles = []

    # 数牌：万、筒、索各 36 张（每种 4 张）
    for tile_type in [TileType.MAN, TileType.PIN, TileType.SOU]:
        for value in range(1, 10):
            for _ in range(4):
                tiles.append(Tile(tile_type, value))

    # 字牌：7 种各 4 张
    for value in range(1, 8):
        for _ in range(4):
            tiles.append(Tile(TileType.HONOR, value))

    return tiles
