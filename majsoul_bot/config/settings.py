"""
配置管理模块
使用 pydantic 进行配置验证和管理
"""
from pathlib import Path
from typing import List, Tuple
import yaml
from pydantic import BaseModel, Field


class AccountConfig(BaseModel):
    """账号配置"""
    username: str = Field(default="", description="雀魂账号用户名")
    password: str = Field(default="", description="雀魂账号密码")


class ServerConfig(BaseModel):
    """服务器配置"""
    host: str = Field(default="game.maj-soul.com", description="服务器地址")
    port: int = Field(default=443, description="服务器端口")
    use_ssl: bool = Field(default=True, description="是否使用 SSL")
    path: str = Field(default="", description="WebSocket 路径")


class GameConfig(BaseModel):
    """游戏配置"""
    auto_ready: bool = Field(default=True, description="是否自动准备")
    delay_range: Tuple[float, float] = Field(
        default=(1.0, 3.0),
        description="操作延迟范围（秒）"
    )
    match_mode: int = Field(default=1, description="匹配模式：1-铜之间，2-银之间，3-金之间")


class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = Field(default="INFO", description="日志级别")
    file: str = Field(default="logs/bot.log", description="日志文件路径")


class Settings(BaseModel):
    """主配置类"""
    account: AccountConfig = Field(default_factory=AccountConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    game: GameConfig = Field(default_factory=GameConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def load_from_yaml(cls, config_path: str = "config/config.yaml") -> "Settings":
        """
        从 YAML 文件加载配置

        Args:
            config_path: 配置文件路径

        Returns:
            Settings: 配置实例

        Raises:
            FileNotFoundError: 配置文件不存在
            ValueError: 配置文件格式错误
        """
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(
                f"配置文件不存在: {config_path}\n"
                f"请复制 config/config.example.yaml 到 config/config.yaml 并填写配置"
            )

        try:
            with open(path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            if config_data is None:
                config_data = {}

            return cls(**config_data)
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件格式错误: {e}")
        except Exception as e:
            raise ValueError(f"加载配置文件失败: {e}")

    def save_to_yaml(self, config_path: str = "config/config.yaml"):
        """
        保存配置到 YAML 文件

        Args:
            config_path: 配置文件路径
        """
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.model_dump(),
                f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False
            )


# 全局配置实例
_settings: Settings = None


def get_settings() -> Settings:
    """
    获取全局配置实例

    Returns:
        Settings: 配置实例
    """
    global _settings
    if _settings is None:
        _settings = Settings.load_from_yaml()
    return _settings


def reload_settings(config_path: str = "config/config.yaml"):
    """
    重新加载配置

    Args:
        config_path: 配置文件路径
    """
    global _settings
    _settings = Settings.load_from_yaml(config_path)
    return _settings
