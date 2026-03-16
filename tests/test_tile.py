"""
测试麻将牌类
"""
import pytest
from majsoul_bot.game_logic import Tile, TileType, parse_tiles, tiles_to_string, create_all_tiles


class TestTile:
    """测试 Tile 类"""

    def test_create_tile(self):
        """测试创建麻将牌"""
        tile = Tile(TileType.MAN, 1)
        assert tile.tile_type == TileType.MAN
        assert tile.value == 1
        assert not tile.aka

    def test_create_aka_tile(self):
        """测试创建赤宝牌"""
        tile = Tile(TileType.PIN, 5, aka=True)
        assert tile.tile_type == TileType.PIN
        assert tile.value == 5
        assert tile.aka

    def test_tile_invalid_value(self):
        """测试无效的牌值"""
        with pytest.raises(ValueError):
            Tile(TileType.MAN, 0)

        with pytest.raises(ValueError):
            Tile(TileType.MAN, 10)

        with pytest.raises(ValueError):
            Tile(TileType.HONOR, 8)

    def test_tile_str(self):
        """测试牌的字符串表示"""
        tile = Tile(TileType.MAN, 1)
        assert str(tile) == "1m"

        tile_aka = Tile(TileType.PIN, 5, aka=True)
        assert str(tile_aka) == "5rp"

    def test_tile_equality(self):
        """测试牌的相等比较"""
        tile1 = Tile(TileType.MAN, 1)
        tile2 = Tile(TileType.MAN, 1)
        tile3 = Tile(TileType.MAN, 2)

        assert tile1 == tile2
        assert tile1 != tile3

    def test_tile_comparison(self):
        """测试牌的大小比较"""
        tile1 = Tile(TileType.MAN, 1)
        tile2 = Tile(TileType.MAN, 2)
        tile3 = Tile(TileType.PIN, 1)

        assert tile1 < tile2
        assert tile1 < tile3

    def test_is_terminal(self):
        """测试是否为边张"""
        assert Tile(TileType.MAN, 1).is_terminal()
        assert Tile(TileType.MAN, 9).is_terminal()
        assert not Tile(TileType.MAN, 5).is_terminal()
        assert not Tile(TileType.HONOR, 1).is_terminal()

    def test_is_honor(self):
        """测试是否为字牌"""
        assert Tile(TileType.HONOR, 1).is_honor()
        assert not Tile(TileType.MAN, 1).is_honor()

    def test_is_wind(self):
        """测试是否为风牌"""
        assert Tile(TileType.HONOR, 1).is_wind()  # 东
        assert Tile(TileType.HONOR, 4).is_wind()  # 北
        assert not Tile(TileType.HONOR, 5).is_wind()  # 白

    def test_is_dragon(self):
        """测试是否为三元牌"""
        assert Tile(TileType.HONOR, 5).is_dragon()  # 白
        assert Tile(TileType.HONOR, 7).is_dragon()  # 中
        assert not Tile(TileType.HONOR, 1).is_dragon()  # 东

    def test_is_yaochuhai(self):
        """测试是否为么九牌"""
        assert Tile(TileType.MAN, 1).is_yaochuhai()
        assert Tile(TileType.MAN, 9).is_yaochuhai()
        assert Tile(TileType.HONOR, 1).is_yaochuhai()
        assert not Tile(TileType.MAN, 5).is_yaochuhai()

    def test_is_simple(self):
        """测试是否为中张牌"""
        assert Tile(TileType.MAN, 2).is_simple()
        assert Tile(TileType.MAN, 8).is_simple()
        assert not Tile(TileType.MAN, 1).is_simple()
        assert not Tile(TileType.HONOR, 1).is_simple()

    def test_get_display_name(self):
        """测试获取显示名称"""
        assert Tile(TileType.MAN, 1).get_display_name() == "1万"
        assert Tile(TileType.PIN, 5).get_display_name() == "5筒"
        assert Tile(TileType.SOU, 9).get_display_name() == "9索"
        assert Tile(TileType.HONOR, 1).get_display_name() == "东"
        assert Tile(TileType.HONOR, 5).get_display_name() == "白"
        assert Tile(TileType.HONOR, 7).get_display_name() == "中"

    def test_from_string(self):
        """测试从字符串创建牌"""
        tile = Tile.from_string("1m")
        assert tile.tile_type == TileType.MAN
        assert tile.value == 1

        tile_aka = Tile.from_string("5rp")
        assert tile_aka.tile_type == TileType.PIN
        assert tile_aka.value == 5
        assert tile_aka.aka

        tile_red_short = Tile.from_string("0m")
        assert tile_red_short.tile_type == TileType.MAN
        assert tile_red_short.value == 5
        assert tile_red_short.aka

    def test_from_string_invalid(self):
        """测试无效字符串"""
        with pytest.raises(ValueError):
            Tile.from_string("10m")

        with pytest.raises(ValueError):
            Tile.from_string("x")


class TestParseTiles:
    """测试牌的解析功能"""

    def test_parse_compact_format(self):
        """测试紧凑格式解析"""
        tiles = parse_tiles("123m")
        assert len(tiles) == 3
        assert tiles[0].value == 1
        assert tiles[1].value == 2
        assert tiles[2].value == 3
        assert all(t.tile_type == TileType.MAN for t in tiles)

    def test_parse_multiple_types(self):
        """测试多种类型解析"""
        tiles = parse_tiles("123m456p789s")
        assert len(tiles) == 9
        assert sum(1 for t in tiles if t.tile_type == TileType.MAN) == 3
        assert sum(1 for t in tiles if t.tile_type == TileType.PIN) == 3
        assert sum(1 for t in tiles if t.tile_type == TileType.SOU) == 3

    def test_parse_compact_with_red_dora_zero(self):
        """测试紧凑格式中的 0 表示赤五。"""
        tiles = parse_tiles("406m")
        assert len(tiles) == 3
        assert tiles[0].value == 4 and not tiles[0].aka
        assert tiles[1].value == 5 and tiles[1].aka
        assert tiles[2].value == 6 and not tiles[2].aka

    def test_parse_space_separated(self):
        """测试空格分隔格式解析"""
        tiles = parse_tiles("1m 2m 3m")
        assert len(tiles) == 3
        assert all(t.tile_type == TileType.MAN for t in tiles)

    def test_tiles_to_string_compact(self):
        """测试转换为紧凑格式字符串"""
        tiles = [
            Tile(TileType.MAN, 1),
            Tile(TileType.MAN, 2),
            Tile(TileType.MAN, 3)
        ]
        result = tiles_to_string(tiles, compact=True)
        assert result == "123m"

    def test_tiles_to_string_spaced(self):
        """测试转换为空格分隔字符串"""
        tiles = [
            Tile(TileType.MAN, 1),
            Tile(TileType.MAN, 2)
        ]
        result = tiles_to_string(tiles, compact=False)
        assert "1m" in result and "2m" in result


class TestCreateAllTiles:
    """测试创建所有牌"""

    def test_create_all_tiles(self):
        """测试创建所有 136 张牌"""
        tiles = create_all_tiles()
        assert len(tiles) == 136

        # 检查数牌：万、筒、索各 36 张
        man_tiles = [t for t in tiles if t.tile_type == TileType.MAN]
        assert len(man_tiles) == 36

        pin_tiles = [t for t in tiles if t.tile_type == TileType.PIN]
        assert len(pin_tiles) == 36

        sou_tiles = [t for t in tiles if t.tile_type == TileType.SOU]
        assert len(sou_tiles) == 36

        # 检查字牌：28 张
        honor_tiles = [t for t in tiles if t.tile_type == TileType.HONOR]
        assert len(honor_tiles) == 28
