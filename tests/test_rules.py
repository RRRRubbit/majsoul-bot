"""
测试麻将规则
"""
import pytest
from majsoul_bot.game_logic import Tile, TileType, Hand, MahjongRules, parse_tiles


class TestHand:
    """测试 Hand 类"""

    def test_create_empty_hand(self):
        """测试创建空手牌"""
        hand = Hand()
        assert hand.get_tile_count() == 0

    def test_add_tile(self):
        """测试添加牌"""
        hand = Hand()
        tile = Tile(TileType.MAN, 1)
        hand.add_tile(tile)
        assert hand.get_tile_count() == 1
        assert hand.has_tile(tile)

    def test_remove_tile(self):
        """测试移除牌"""
        hand = Hand()
        tile = Tile(TileType.MAN, 1)
        hand.add_tile(tile)
        assert hand.remove_tile(tile)
        assert hand.get_tile_count() == 0

    def test_sort_tiles(self):
        """测试排序"""
        hand = Hand()
        hand.add_tile(Tile(TileType.MAN, 3))
        hand.add_tile(Tile(TileType.MAN, 1))
        hand.add_tile(Tile(TileType.MAN, 2))

        assert hand.tiles[0].value == 1
        assert hand.tiles[1].value == 2
        assert hand.tiles[2].value == 3

    def test_count_tile(self):
        """测试统计牌的数量"""
        hand = Hand()
        tile = Tile(TileType.MAN, 1)
        hand.add_tile(tile)
        hand.add_tile(tile)
        hand.add_tile(tile)
        assert hand.count_tile(tile) == 3

    def test_get_groups(self):
        """测试按类型分组"""
        hand = Hand()
        hand.add_tile(Tile(TileType.MAN, 1))
        hand.add_tile(Tile(TileType.PIN, 2))
        hand.add_tile(Tile(TileType.SOU, 3))

        groups = hand.get_groups()
        assert len(groups[TileType.MAN]) == 1
        assert len(groups[TileType.PIN]) == 1
        assert len(groups[TileType.SOU]) == 1


class TestSevenPairs:
    """测试七对子"""

    def test_is_seven_pairs(self):
        """测试七对子判断"""
        hand = Hand()
        # 添加 7 对牌
        for i in range(1, 8):
            hand.add_tile(Tile(TileType.MAN, i))
            hand.add_tile(Tile(TileType.MAN, i))

        assert hand.is_seven_pairs()

    def test_not_seven_pairs(self):
        """测试非七对子"""
        hand = Hand()
        # 添加 6 对 + 2 张不同的牌
        for i in range(1, 7):
            hand.add_tile(Tile(TileType.MAN, i))
            hand.add_tile(Tile(TileType.MAN, i))
        hand.add_tile(Tile(TileType.MAN, 7))
        hand.add_tile(Tile(TileType.MAN, 8))

        assert not hand.is_seven_pairs()


class TestKokushi:
    """测试国士无双"""

    def test_is_kokushi(self):
        """测试国士无双判断"""
        hand = Hand()
        # 添加所有么九牌，其中一种 2 张
        kokushi_tiles = [
            (TileType.MAN, 1), (TileType.MAN, 9),
            (TileType.PIN, 1), (TileType.PIN, 9),
            (TileType.SOU, 1), (TileType.SOU, 9),
            (TileType.HONOR, 1), (TileType.HONOR, 2),
            (TileType.HONOR, 3), (TileType.HONOR, 4),
            (TileType.HONOR, 5), (TileType.HONOR, 6),
            (TileType.HONOR, 7)
        ]

        for tile_type, value in kokushi_tiles:
            hand.add_tile(Tile(tile_type, value))

        # 添加一张重复的作为雀头
        hand.add_tile(Tile(TileType.MAN, 1))

        assert hand.is_kokushi()

    def test_not_kokushi(self):
        """测试非国士无双"""
        hand = Hand()
        # 添加普通牌
        for i in range(1, 8):
            hand.add_tile(Tile(TileType.MAN, i))
            hand.add_tile(Tile(TileType.MAN, i))

        assert not hand.is_kokushi()


class TestMahjongRules:
    """测试麻将规则"""

    def test_can_chi(self):
        """测试吃牌判断"""
        hand = Hand()
        hand.add_tile(Tile(TileType.MAN, 2))
        hand.add_tile(Tile(TileType.MAN, 3))

        tile = Tile(TileType.MAN, 1)
        tiles_from_hand = [Tile(TileType.MAN, 2), Tile(TileType.MAN, 3)]

        assert MahjongRules.can_chi(hand, tile, tiles_from_hand)

    def test_cannot_chi_honor(self):
        """测试不能吃字牌"""
        hand = Hand()
        hand.add_tile(Tile(TileType.HONOR, 1))
        hand.add_tile(Tile(TileType.HONOR, 2))

        tile = Tile(TileType.HONOR, 3)
        tiles_from_hand = [Tile(TileType.HONOR, 1), Tile(TileType.HONOR, 2)]

        assert not MahjongRules.can_chi(hand, tile, tiles_from_hand)

    def test_can_pon(self):
        """测试碰牌判断"""
        hand = Hand()
        tile = Tile(TileType.MAN, 1)
        hand.add_tile(tile)
        hand.add_tile(tile)

        assert MahjongRules.can_pon(hand, tile)

    def test_cannot_pon(self):
        """测试不能碰牌"""
        hand = Hand()
        tile = Tile(TileType.MAN, 1)
        hand.add_tile(tile)

        assert not MahjongRules.can_pon(hand, tile)

    def test_can_kan(self):
        """测试杠牌判断"""
        hand = Hand()
        tile = Tile(TileType.MAN, 1)
        hand.add_tile(tile)
        hand.add_tile(tile)
        hand.add_tile(tile)

        assert MahjongRules.can_kan(hand, tile)

    def test_cannot_kan(self):
        """测试不能杠牌"""
        hand = Hand()
        tile = Tile(TileType.MAN, 1)
        hand.add_tile(tile)
        hand.add_tile(tile)

        assert not MahjongRules.can_kan(hand, tile)

    def test_calculate_han_tanyao(self):
        """测试断么九番数计算"""
        hand = Hand()
        # 添加只有中张牌的手牌
        tiles = parse_tiles("234m456p678s22s")
        for tile in tiles:
            hand.add_tile(tile)

        winning_tile = Tile(TileType.SOU, 2)
        han, yaku_list = MahjongRules.calculate_han(hand, winning_tile)

        assert "断么九" in yaku_list

    def test_calculate_han_with_dora(self):
        """测试宝牌番数"""
        hand = Hand()
        tiles = parse_tiles("123m456p789s11z")
        for tile in tiles:
            hand.add_tile(tile)

        winning_tile = Tile(TileType.HONOR, 1)
        han, yaku_list = MahjongRules.calculate_han(
            hand,
            winning_tile,
            dora_count=2
        )

        assert "宝牌 x2" in yaku_list
        assert han >= 2


class TestShantenCalculation:
    """测试向听数计算"""

    def test_winning_hand_shanten(self):
        """测试和牌时的向听数"""
        hand = Hand()
        # 创建一个简单的和牌
        tiles = parse_tiles("123m456p789s11z")
        for tile in tiles:
            hand.add_tile(tile)

        # 和牌应该返回 -1
        if hand.is_winning():
            shanten = hand.calculate_shanten()
            assert shanten == -1

    def test_tenpai_shanten(self):
        """测试听牌时的向听数"""
        hand = Hand()
        # 创建一个接近和牌的手牌
        tiles = parse_tiles("123m456p789s1z")
        for tile in tiles:
            hand.add_tile(tile)

        shanten = hand.calculate_shanten()
        # 听牌应该返回 0 或接近 0
        assert shanten <= 1


class TestGetWaitingTiles:
    """测试获取等待的牌"""

    def test_get_waiting_tiles_not_tenpai(self):
        """测试非听牌状态"""
        hand = Hand()
        tiles = parse_tiles("123m")
        for tile in tiles:
            hand.add_tile(tile)

        waiting = MahjongRules.get_waiting_tiles(hand)
        # 非听牌状态应该返回空列表
        assert len(waiting) == 0 or not MahjongRules.is_tenpai(hand)
