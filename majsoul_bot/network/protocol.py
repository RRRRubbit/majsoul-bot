"""
协议解析模块
处理雀魂游戏协议的解析和序列化
"""
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from loguru import logger


class MessageType(Enum):
    """消息类型枚举"""
    # 连接相关
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    HEARTBEAT = "heartbeat"

    # 认证相关
    LOGIN = "login"
    LOGOUT = "logout"

    # 游戏大厅
    LOBBY = "lobby"
    CREATE_ROOM = "create_room"
    JOIN_ROOM = "join_room"
    LEAVE_ROOM = "leave_room"

    # 游戏流程
    GAME_START = "game_start"
    GAME_END = "game_end"
    NEW_ROUND = "new_round"
    DEAL_TILE = "deal_tile"
    DISCARD_TILE = "discard_tile"

    # 操作
    DRAW = "draw"
    CHI = "chi"
    PON = "pon"
    KAN = "kan"
    RIICHI = "riichi"
    RON = "ron"
    TSUMO = "tsumo"

    # 其他
    DORA_INDICATOR = "dora_indicator"
    OPERATION = "operation"
    UNKNOWN = "unknown"


@dataclass
class GameMessage:
    """
    游戏消息数据类

    Attributes:
        msg_type: 消息类型
        data: 消息数据
        raw: 原始消息
    """
    msg_type: MessageType
    data: Dict[str, Any]
    raw: Any


class ProtocolHandler:
    """协议处理器"""

    def __init__(self):
        """初始化协议处理器"""
        self.message_handlers: Dict[MessageType, callable] = {}

    def register_handler(self, msg_type: MessageType, handler: callable):
        """
        注册消息处理器

        Args:
            msg_type: 消息类型
            handler: 处理函数
        """
        self.message_handlers[msg_type] = handler
        logger.debug(f"Registered handler for {msg_type.value}")

    def parse_message(self, raw_message: Any) -> Optional[GameMessage]:
        """
        解析消息

        Args:
            raw_message: 原始消息（JSON 对象或二进制数据）

        Returns:
            GameMessage: 解析后的消息对象，如果解析失败则返回 None
        """
        try:
            # 处理 JSON 消息
            if isinstance(raw_message, dict):
                return self._parse_json_message(raw_message)

            # 处理二进制消息
            elif isinstance(raw_message, bytes):
                return self._parse_binary_message(raw_message)

            else:
                logger.warning(f"Unknown message format: {type(raw_message)}")
                return None

        except Exception as e:
            logger.error(f"Failed to parse message: {e}")
            return None

    def _parse_json_message(self, message: Dict[str, Any]) -> GameMessage:
        """
        解析 JSON 消息

        Args:
            message: JSON 消息对象

        Returns:
            GameMessage: 解析后的消息对象
        """
        # 尝试从消息中提取类型
        msg_type_str = message.get("type", "unknown")

        # 将字符串转换为 MessageType
        try:
            msg_type = MessageType(msg_type_str)
        except ValueError:
            msg_type = MessageType.UNKNOWN

        return GameMessage(
            msg_type=msg_type,
            data=message.get("data", {}),
            raw=message
        )

    def _parse_binary_message(self, data: bytes) -> GameMessage:
        """
        解析二进制消息（用于 Protocol Buffers 等）

        Args:
            data: 二进制数据

        Returns:
            GameMessage: 解析后的消息对象

        Note:
            这是一个占位实现，实际需要根据雀魂的协议格式进行解析
        """
        logger.debug(f"Parsing binary message: {len(data)} bytes")

        # 占位实现：返回一个未知类型的消息
        return GameMessage(
            msg_type=MessageType.UNKNOWN,
            data={"binary_data": data},
            raw=data
        )

    async def handle_message(self, message: GameMessage):
        """
        处理消息

        Args:
            message: 游戏消息对象
        """
        handler = self.message_handlers.get(message.msg_type)

        if handler:
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Error handling message {message.msg_type.value}: {e}")
        else:
            logger.debug(f"No handler for message type: {message.msg_type.value}")

    def create_message(self, msg_type: MessageType, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建消息

        Args:
            msg_type: 消息类型
            data: 消息数据

        Returns:
            Dict: 消息对象
        """
        return {
            "type": msg_type.value,
            "data": data
        }

    def create_login_message(self, username: str, password: str) -> Dict[str, Any]:
        """
        创建登录消息

        Args:
            username: 用户名
            password: 密码

        Returns:
            Dict: 登录消息
        """
        return self.create_message(
            MessageType.LOGIN,
            {
                "username": username,
                "password": password
            }
        )

    def create_discard_message(self, tile: str) -> Dict[str, Any]:
        """
        创建打牌消息

        Args:
            tile: 打出的牌

        Returns:
            Dict: 打牌消息
        """
        return self.create_message(
            MessageType.DISCARD_TILE,
            {"tile": tile}
        )

    def create_operation_message(self, operation: str, tiles: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        创建操作消息（吃、碰、杠等）

        Args:
            operation: 操作类型（chi, pon, kan, riichi, etc.）
            tiles: 相关的牌（可选）

        Returns:
            Dict: 操作消息
        """
        data = {"operation": operation}
        if tiles:
            data["tiles"] = tiles

        return self.create_message(MessageType.OPERATION, data)


class GameState:
    """
    游戏状态类

    用于跟踪当前游戏状态
    """

    def __init__(self):
        """初始化游戏状态"""
        self.is_in_game = False
        self.seat = -1  # 座位号 (0-3)
        self.round = 0  # 局数
        self.dealer = -1  # 庄家座位
        self.dora_indicators: List[str] = []  # 宝牌指示牌
        self.discarded_tiles: List[str] = []  # 已打出的牌
        self.player_hands: Dict[int, List[str]] = {}  # 各玩家手牌（只知道自己的）
        self.last_operation: Optional[Dict[str, Any]] = None  # 最后一次操作

    def reset(self):
        """重置游戏状态"""
        self.is_in_game = False
        self.seat = -1
        self.round = 0
        self.dealer = -1
        self.dora_indicators.clear()
        self.discarded_tiles.clear()
        self.player_hands.clear()
        self.last_operation = None

    def update_from_message(self, message: GameMessage):
        """
        从消息更新游戏状态

        Args:
            message: 游戏消息
        """
        if message.msg_type == MessageType.GAME_START:
            self.is_in_game = True
            self.seat = message.data.get("seat", -1)
            logger.info(f"Game started, seat: {self.seat}")

        elif message.msg_type == MessageType.GAME_END:
            self.is_in_game = False
            logger.info("Game ended")

        elif message.msg_type == MessageType.NEW_ROUND:
            self.round = message.data.get("round", 0)
            self.dealer = message.data.get("dealer", -1)
            logger.info(f"New round: {self.round}, dealer: {self.dealer}")

        elif message.msg_type == MessageType.DEAL_TILE:
            tile = message.data.get("tile")
            if tile:
                hand = self.player_hands.get(self.seat, [])
                hand.append(tile)
                self.player_hands[self.seat] = hand

        elif message.msg_type == MessageType.DISCARD_TILE:
            tile = message.data.get("tile")
            if tile:
                self.discarded_tiles.append(tile)

        elif message.msg_type == MessageType.DORA_INDICATOR:
            indicators = message.data.get("indicators", [])
            self.dora_indicators.extend(indicators)

    def __str__(self) -> str:
        """字符串表示"""
        return (
            f"GameState(in_game={self.is_in_game}, seat={self.seat}, "
            f"round={self.round}, dealer={self.dealer})"
        )
