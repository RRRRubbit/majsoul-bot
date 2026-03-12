"""
雀魂机器人主程序
整合所有模块，提供完整的自动打牌功能
"""
import asyncio
import random
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from majsoul_bot.utils import setup_logger, get_logger
from majsoul_bot.config import get_settings
from majsoul_bot.network import WebSocketClient, ProtocolHandler, MessageType, GameState, GameMessage
from majsoul_bot.game_logic import Tile, Hand, parse_tiles, MahjongRules
from majsoul_bot.ai import SimpleAI


class MajsoulBot:
    """
    雀魂机器人主类

    整合网络通信、游戏逻辑和 AI 决策
    """

    def __init__(self):
        """初始化机器人"""
        # 加载配置
        try:
            self.settings = get_settings()
        except FileNotFoundError as e:
            print(f"错误: {e}")
            sys.exit(1)

        # 设置日志
        setup_logger(
            log_level=self.settings.logging.level,
            log_file=self.settings.logging.file
        )
        self.logger = get_logger()

        # 初始化模块
        self.ws_client = WebSocketClient(
            host=self.settings.server.host,
            port=self.settings.server.port,
            use_ssl=self.settings.server.use_ssl,
            path=self.settings.server.path
        )
        self.protocol_handler = ProtocolHandler()
        self.game_state = GameState()
        self.ai = SimpleAI()
        self.hand = Hand()

        # 注册消息处理器
        self._register_handlers()

        # 运行状态
        self.is_running = False

    def _register_handlers(self):
        """注册消息处理器"""
        self.protocol_handler.register_handler(
            MessageType.GAME_START,
            self._handle_game_start
        )
        self.protocol_handler.register_handler(
            MessageType.GAME_END,
            self._handle_game_end
        )
        self.protocol_handler.register_handler(
            MessageType.NEW_ROUND,
            self._handle_new_round
        )
        self.protocol_handler.register_handler(
            MessageType.DEAL_TILE,
            self._handle_deal_tile
        )
        self.protocol_handler.register_handler(
            MessageType.DISCARD_TILE,
            self._handle_discard_tile
        )
        self.protocol_handler.register_handler(
            MessageType.OPERATION,
            self._handle_operation
        )

        self.logger.info("消息处理器注册完成")

    async def start(self):
        """启动机器人"""
        self.logger.info("=" * 50)
        self.logger.info("雀魂机器人启动中...")
        self.logger.info("=" * 50)

        # 连接到服务器
        if not await self.ws_client.connect():
            self.logger.error("无法连接到服务器")
            return

        # 设置消息处理器
        self.ws_client.set_message_handler(self._on_message)

        # 登录
        await self._login()

        # 设置运行状态
        self.is_running = True

        # 主循环
        try:
            await self._main_loop()
        except KeyboardInterrupt:
            self.logger.info("收到中断信号，正在退出...")
        finally:
            await self.stop()

    async def stop(self):
        """停止机器人"""
        self.logger.info("机器人正在停止...")
        self.is_running = False
        await self.ws_client.disconnect()
        self.logger.info("机器人已停止")

    async def _login(self):
        """登录到雀魂"""
        username = self.settings.account.username
        password = self.settings.account.password

        if not username or not password:
            self.logger.warning("未配置账号信息，跳过登录")
            return

        self.logger.info(f"正在登录账号: {username}")

        # 创建登录消息
        login_msg = self.protocol_handler.create_login_message(username, password)

        # 发送登录消息
        await self.ws_client.send_message(login_msg)

        self.logger.info("登录请求已发送")

    async def _main_loop(self):
        """主循环"""
        self.logger.info("进入主循环")

        while self.is_running:
            # 等待一段时间
            await asyncio.sleep(1)

            # 检查连接状态
            if not self.ws_client.is_connected:
                self.logger.warning("连接已断开，尝试重连...")
                if await self.ws_client.reconnect():
                    await self._login()
                else:
                    self.logger.error("重连失败，退出")
                    break

        self.logger.info("退出主循环")

    async def _on_message(self, raw_message):
        """
        处理接收到的消息

        Args:
            raw_message: 原始消息
        """
        # 解析消息
        message = self.protocol_handler.parse_message(raw_message)

        if message:
            # 更新游戏状态
            self.game_state.update_from_message(message)

            # 处理消息
            await self.protocol_handler.handle_message(message)

    async def _handle_game_start(self, message: GameMessage):
        """处理游戏开始"""
        self.logger.info("=" * 50)
        self.logger.info("游戏开始！")
        self.logger.info("=" * 50)

        self.hand.clear()
        self.ai.on_game_start()

    async def _handle_game_end(self, message: GameMessage):
        """处理游戏结束"""
        self.logger.info("=" * 50)
        self.logger.info("游戏结束！")
        self.logger.info("=" * 50)

        self.ai.on_game_end()

    async def _handle_new_round(self, message: GameMessage):
        """处理新局开始"""
        round_num = message.data.get("round", 0)
        dealer = message.data.get("dealer", -1)

        self.logger.info(f"新局开始 - 局数: {round_num}, 庄家座位: {dealer}")

        self.hand.clear()
        self.ai.on_round_start()

    async def _handle_deal_tile(self, message: GameMessage):
        """处理摸牌"""
        tile_str = message.data.get("tile")

        if not tile_str:
            return

        try:
            tile = Tile.from_string(tile_str)
            self.hand.add_tile(tile)

            self.logger.info(f"摸牌: {tile.get_display_name()} ({tile})")
            self.logger.info(f"当前手牌: {self.hand}")

            # 计算向听数
            shanten = self.hand.calculate_shanten()
            self.logger.info(f"向听数: {shanten}")

            # 决定打牌
            await self._auto_discard(tile)

        except ValueError as e:
            self.logger.error(f"无效的牌: {tile_str}, 错误: {e}")

    async def _handle_discard_tile(self, message: GameMessage):
        """处理打牌"""
        tile_str = message.data.get("tile")
        seat = message.data.get("seat", -1)

        if tile_str:
            self.logger.info(f"玩家 {seat} 打出: {tile_str}")

    async def _handle_operation(self, message: GameMessage):
        """处理操作请求（吃、碰、杠、和等）"""
        operations = message.data.get("operations", [])

        if not operations:
            return

        self.logger.info(f"收到操作请求: {operations}")

        # 检查是否可以和牌
        if "ron" in operations or "tsumo" in operations:
            if "tsumo" in operations and self.ai.decide_tsumo(self.hand):
                await self._do_tsumo()
                return

        # 检查是否可以立直
        if "riichi" in operations and self.ai.decide_riichi(self.hand):
            await self._do_riichi()
            return

        # 检查其他操作（碰、杠等）
        # 简化实现：跳过其他操作

        self.logger.info("跳过操作")

    async def _auto_discard(self, drawn_tile: Tile):
        """
        自动打牌

        Args:
            drawn_tile: 刚摸到的牌
        """
        # 添加随机延迟（模拟人类思考时间）
        delay = random.uniform(*self.settings.game.delay_range)
        await asyncio.sleep(delay)

        # 使用 AI 决定打哪张牌
        tile_to_discard = self.ai.decide_discard(self.hand, drawn_tile)

        # 从手牌中移除
        if self.hand.remove_tile(tile_to_discard):
            self.logger.info(f"打出: {tile_to_discard.get_display_name()} ({tile_to_discard})")

            # 发送打牌消息
            discard_msg = self.protocol_handler.create_discard_message(str(tile_to_discard))
            await self.ws_client.send_message(discard_msg)
        else:
            self.logger.error(f"无法打出 {tile_to_discard}，牌不在手中")

    async def _do_riichi(self):
        """执行立直"""
        self.logger.info("执行立直")

        # 发送立直消息
        riichi_msg = self.protocol_handler.create_operation_message("riichi")
        await self.ws_client.send_message(riichi_msg)

    async def _do_tsumo(self):
        """执行自摸和"""
        self.logger.info("执行自摸和")

        # 发送自摸消息
        tsumo_msg = self.protocol_handler.create_operation_message("tsumo")
        await self.ws_client.send_message(tsumo_msg)


async def main():
    """主函数"""
    bot = MajsoulBot()
    await bot.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序已终止")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
