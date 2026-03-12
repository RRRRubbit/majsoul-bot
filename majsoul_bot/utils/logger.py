"""
日志工具模块
使用 loguru 提供统一的日志记录功能
"""
from loguru import logger
import sys
from pathlib import Path


def setup_logger(log_level: str = "INFO", log_file: str = "logs/bot.log"):
    """
    配置日志记录器

    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径
    """
    # 移除默认的 handler
    logger.remove()

    # 添加控制台输出
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )

    # 确保日志目录存在
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 添加文件输出
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level,
        rotation="10 MB",  # 文件大小超过 10MB 时轮转
        retention="7 days",  # 保留 7 天的日志
        compression="zip",  # 压缩旧日志
        encoding="utf-8"
    )

    logger.info(f"Logger initialized with level: {log_level}")
    return logger


def get_logger():
    """
    获取日志记录器实例

    Returns:
        logger: loguru logger 实例
    """
    return logger
