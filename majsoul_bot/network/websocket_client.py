"""
WebSocket 客户端模块
处理与雀魂服务器的 WebSocket 连接
"""
import asyncio
import json
from typing import Optional, Callable, Dict, Any
from loguru import logger
import websockets
from websockets.client import WebSocketClientProtocol


class WebSocketClient:
    """
    WebSocket 客户端类

    Attributes:
        url: WebSocket 服务器地址
        websocket: WebSocket 连接对象
        is_connected: 是否已连接
        message_handler: 消息处理回调函数
    """

    def __init__(self, host: str, port: int, use_ssl: bool = True, path: str = ""):
        """
        初始化 WebSocket 客户端

        Args:
            host: 服务器地址
            port: 服务器端口
            use_ssl: 是否使用 SSL
            path: WebSocket 路径
        """
        protocol = "wss" if use_ssl else "ws"
        # 构建完整的 WebSocket URL
        if path:
            self.url = f"{protocol}://{host}:{port}{path}"
        else:
            self.url = f"{protocol}://{host}:{port}"

        self.host = host
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.is_connected = False
        self.message_handler: Optional[Callable] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """
        连接到 WebSocket 服务器

        Returns:
            bool: 是否连接成功
        """
        try:
            logger.info(f"Connecting to {self.url}...")

            # 添加必要的请求头
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Origin": f"https://{self.host}",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            }

            self.websocket = await websockets.connect(
                self.url,
                extra_headers=headers,
                ping_interval=30,
                ping_timeout=10
            )
            self.is_connected = True
            logger.info("WebSocket connected successfully")

            # 启动心跳和接收消息任务
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._receive_task = asyncio.create_task(self._receive_loop())

            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.is_connected = False
            return False

    async def disconnect(self):
        """断开 WebSocket 连接"""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._receive_task:
            self._receive_task.cancel()

        if self.websocket:
            await self.websocket.close()
            logger.info("WebSocket disconnected")

        self.is_connected = False

    async def send_message(self, message: Dict[str, Any]) -> bool:
        """
        发送消息到服务器

        Args:
            message: 要发送的消息（字典格式）

        Returns:
            bool: 是否发送成功
        """
        if not self.is_connected or not self.websocket:
            logger.error("WebSocket is not connected")
            return False

        try:
            message_json = json.dumps(message)
            await self.websocket.send(message_json)
            logger.debug(f"Sent message: {message_json}")
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False

    async def send_binary(self, data: bytes) -> bool:
        """
        发送二进制数据到服务器

        Args:
            data: 要发送的二进制数据

        Returns:
            bool: 是否发送成功
        """
        if not self.is_connected or not self.websocket:
            logger.error("WebSocket is not connected")
            return False

        try:
            await self.websocket.send(data)
            logger.debug(f"Sent binary data: {len(data)} bytes")
            return True
        except Exception as e:
            logger.error(f"Failed to send binary data: {e}")
            return False

    def set_message_handler(self, handler: Callable):
        """
        设置消息处理回调函数

        Args:
            handler: 消息处理函数，接收消息数据作为参数
        """
        self.message_handler = handler

    async def _receive_loop(self):
        """接收消息循环"""
        try:
            while self.is_connected and self.websocket:
                try:
                    message = await self.websocket.recv()

                    # 处理文本消息
                    if isinstance(message, str):
                        try:
                            data = json.loads(message)
                            logger.debug(f"Received message: {message[:200]}")
                            if self.message_handler:
                                await self.message_handler(data)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to decode JSON: {message[:100]}")

                    # 处理二进制消息
                    elif isinstance(message, bytes):
                        logger.debug(f"Received binary data: {len(message)} bytes")
                        if self.message_handler:
                            await self.message_handler(message)

                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WebSocket connection closed")
                    self.is_connected = False
                    break
                except Exception as e:
                    logger.error(f"Error in receive loop: {e}")

        except asyncio.CancelledError:
            logger.info("Receive loop cancelled")

    async def _heartbeat_loop(self):
        """心跳循环"""
        try:
            while self.is_connected and self.websocket:
                await asyncio.sleep(30)  # 每 30 秒发送一次心跳

                if self.is_connected:
                    try:
                        # 发送 ping
                        pong_waiter = await self.websocket.ping()
                        await asyncio.wait_for(pong_waiter, timeout=10)
                        logger.debug("Heartbeat sent")
                    except asyncio.TimeoutError:
                        logger.error("Heartbeat timeout")
                        self.is_connected = False
                        break
                    except Exception as e:
                        logger.error(f"Heartbeat error: {e}")
                        self.is_connected = False
                        break

        except asyncio.CancelledError:
            logger.info("Heartbeat loop cancelled")

    async def reconnect(self, max_retries: int = 5, retry_delay: int = 5) -> bool:
        """
        重新连接

        Args:
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）

        Returns:
            bool: 是否重连成功
        """
        for i in range(max_retries):
            logger.info(f"Reconnecting... (attempt {i + 1}/{max_retries})")
            await self.disconnect()
            await asyncio.sleep(retry_delay)

            if await self.connect():
                logger.info("Reconnected successfully")
                return True

        logger.error("Failed to reconnect after maximum retries")
        return False

    async def wait_until_disconnected(self):
        """等待直到连接断开"""
        if self._receive_task:
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
