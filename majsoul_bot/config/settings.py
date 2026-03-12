"""
配置管理模块
使用 pydantic 进行配置验证和管理
"""
from pathlib import Path
from typing import Tuple

import yaml
from pydantic import BaseModel, Field, model_validator


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
    prefer_riichi: bool = Field(default=True, description="是否优先保持门前清/立直")


class VisionConfig(BaseModel):
    """机器视觉模式配置"""

    templates_dir: str = Field(default="templates", description="模板根目录")
    capture_interval: float = Field(default=0.5, ge=0.05, le=5.0, description="截图间隔（秒）")
    template_threshold: float = Field(default=0.75, ge=0.0, le=1.0, description="牌面模板匹配阈值")
    button_threshold: float = Field(default=0.72, ge=0.0, le=1.0, description="按钮模板匹配阈值")
    debug_mode: bool = Field(default=False, description="是否开启调试模式")
    action_cooldown: float = Field(default=4.0, ge=0.0, le=10.0, description="操作冷却时间（秒）")
    discard_lock_timeout: float = Field(
        default=3.0,
        ge=0.0,
        le=10.0,
        description="出牌锁超时时间（秒）",
    )

    # 神经网络识别配置（OpenCV ANN_MLP）
    nn_enabled: bool = Field(default=True, description="是否启用神经网络辅助识别")
    nn_model_path: str = Field(default="models/tile_ann.xml", description="NN 模型文件路径")
    nn_labels_path: str = Field(
        default="",
        description="NN 标签文件路径（为空时使用模型同名 .labels.json）",
    )
    nn_fusion_weight: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="模板分与 NN 概率融合权重（越大越偏向 NN）",
    )
    nn_min_confidence: float = Field(
        default=0.58,
        ge=0.0,
        le=1.0,
        description="NN 兜底识别最小置信度（融合分不足时可单独通过）",
    )
    nn_top_k: int = Field(
        default=5,
        ge=1,
        le=34,
        description="NN 候选数量（用于融合与日志）",
    )


class ControllerConfig(BaseModel):
    """输入控制配置（鼠标/键盘）"""

    min_delay: float = Field(default=1.0, ge=0.0, le=10.0, description="最小操作延迟（秒）")
    max_delay: float = Field(default=2.5, ge=0.0, le=10.0, description="最大操作延迟（秒）")
    click_variance: int = Field(default=6, ge=0, le=50, description="点击像素随机偏移")

    @model_validator(mode="after")
    def _validate_delay_range(self) -> "ControllerConfig":
        if self.min_delay > self.max_delay:
            raise ValueError("controller.min_delay 不能大于 controller.max_delay")
        return self


class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = Field(default="INFO", description="日志级别")
    file: str = Field(default="logs/vision_bot.log", description="日志文件路径")


class Settings(BaseModel):
    """主配置类"""
    account: AccountConfig = Field(default_factory=AccountConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    controller: ControllerConfig = Field(default_factory=ControllerConfig)
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
                "请创建 config/config.yaml（可参考 majsoul_bot/config/config.example.yaml）"
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
