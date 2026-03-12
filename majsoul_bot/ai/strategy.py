"""
策略基类模块
定义 AI 策略的接口
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from ..game_logic import Tile, Hand, MahjongRules


class Strategy(ABC):
    """
    AI 策略基类

    定义所有策略必须实现的接口
    """

    def __init__(self, name: str = "BaseStrategy"):
        """
        初始化策略

        Args:
            name: 策略名称
        """
        self.name = name

    @abstractmethod
    def decide_discard(self, hand: Hand, drawn_tile: Optional[Tile] = None) -> Tile:
        """
        决定打出哪张牌

        Args:
            hand: 当前手牌
            drawn_tile: 刚摸到的牌（如果有）

        Returns:
            Tile: 决定打出的牌
        """
        pass

    @abstractmethod
    def decide_chi(self, hand: Hand, tile: Tile, available_combinations: List[Tuple[Tile, Tile]]) -> Optional[Tuple[Tile, Tile]]:
        """
        决定是否吃牌以及用哪两张牌吃

        Args:
            hand: 当前手牌
            tile: 可以吃的牌
            available_combinations: 可用的组合列表

        Returns:
            Optional[Tuple[Tile, Tile]]: 用于吃牌的两张牌，None 表示不吃
        """
        pass

    @abstractmethod
    def decide_pon(self, hand: Hand, tile: Tile) -> bool:
        """
        决定是否碰牌

        Args:
            hand: 当前手牌
            tile: 可以碰的牌

        Returns:
            bool: 是否碰牌
        """
        pass

    @abstractmethod
    def decide_kan(self, hand: Hand, tile: Tile, is_ankan: bool = False) -> bool:
        """
        决定是否杠牌

        Args:
            hand: 当前手牌
            tile: 可以杠的牌
            is_ankan: 是否为暗杠

        Returns:
            bool: 是否杠牌
        """
        pass

    @abstractmethod
    def decide_riichi(self, hand: Hand) -> bool:
        """
        决定是否立直

        Args:
            hand: 当前手牌

        Returns:
            bool: 是否立直
        """
        pass

    @abstractmethod
    def decide_ron(self, hand: Hand, tile: Tile) -> bool:
        """
        决定是否荣和

        Args:
            hand: 当前手牌
            tile: 可以荣和的牌

        Returns:
            bool: 是否荣和
        """
        pass

    @abstractmethod
    def decide_tsumo(self, hand: Hand) -> bool:
        """
        决定是否自摸和

        Args:
            hand: 当前手牌

        Returns:
            bool: 是否自摸和
        """
        pass

    def on_game_start(self):
        """游戏开始时的回调"""
        pass

    def on_game_end(self):
        """游戏结束时的回调"""
        pass

    def on_round_start(self):
        """新局开始时的回调"""
        pass

    def on_round_end(self):
        """局结束时的回调"""
        pass

    def update_game_state(self, **kwargs):
        """
        更新游戏状态信息

        Args:
            **kwargs: 游戏状态参数
        """
        pass

    def __str__(self) -> str:
        """字符串表示"""
        return f"Strategy: {self.name}"

    def __repr__(self) -> str:
        """详细表示"""
        return f"{self.__class__.__name__}(name='{self.name}')"
