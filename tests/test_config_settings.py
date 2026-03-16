"""
测试配置模型（机器视觉版）
"""

import pytest
from pydantic import ValidationError

from majsoul_bot.config.settings import ControllerConfig, Settings


class TestVisionSettings:
    """测试视觉配置相关字段与校验逻辑"""

    def test_default_values(self):
        """默认配置应包含视觉与控制器参数"""
        settings = Settings()

        assert settings.vision.templates_dir == "templates"
        assert settings.vision.capture_interval == 0.5
        assert settings.vision.template_threshold == 0.75
        assert settings.vision.button_threshold == 0.72
        assert settings.vision.action_cooldown == 4.0
        assert settings.vision.discard_lock_timeout == 3.0
        assert settings.vision.nn_enabled is True
        assert settings.vision.nn_model_path == "models/tile_ann.xml"
        assert settings.vision.nn_labels_path == ""
        assert settings.vision.nn_fusion_weight == 0.90
        assert settings.vision.nn_min_confidence == 0.58
        assert settings.vision.nn_top_k == 5
        assert settings.vision.nn_priority is True
        assert settings.vision.browser_auto_open is True
        assert settings.vision.browser_url == "https://game.maj-soul.com/1/"
        assert settings.vision.browser_executable == ""
        assert settings.vision.browser_wait_seconds == 2.0
        assert settings.vision.login_auto_fill is False
        assert settings.vision.auto_collect_dataset is False
        assert settings.vision.auto_collect_dir == "datasets/auto"
        assert settings.vision.auto_collect_min_score == 0.93
        assert settings.vision.auto_collect_include_unknown is False
        assert settings.vision.auto_collect_max_per_label == 2000
        assert settings.vision.lock_width == 800
        assert settings.vision.lock_height == 600

        assert settings.controller.min_delay == 1.0
        assert settings.controller.max_delay == 2.5
        assert settings.controller.click_variance == 6

    def test_controller_delay_range_validation(self):
        """控制器延迟范围应满足 min_delay <= max_delay"""
        with pytest.raises(ValidationError):
            ControllerConfig(min_delay=2.0, max_delay=1.0)

    def test_load_from_yaml_with_vision_fields(self, tmp_path):
        """应能从 YAML 加载视觉与控制器配置"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
vision:
  templates_dir: "custom_templates"
  capture_interval: 0.4
  template_threshold: 0.70
  button_threshold: 0.68
  debug_mode: true
  action_cooldown: 3.2
  discard_lock_timeout: 2.5
  nn_enabled: true
  nn_model_path: "models/custom_ann.xml"
  nn_labels_path: "models/custom_ann.labels.json"
  nn_fusion_weight: 0.55
  nn_min_confidence: 0.66
  nn_top_k: 7
  nn_priority: false
  browser_auto_open: true
  browser_url: "https://game.maj-soul.com/1/"
  browser_executable: "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
  browser_wait_seconds: 1.8
  login_auto_fill: true
  auto_collect_dataset: true
  auto_collect_dir: "datasets/runtime_auto"
  auto_collect_min_score: 0.91
  auto_collect_include_unknown: true
  auto_collect_max_per_label: 345
account:
  username: "demo_user"
  password: "demo_password"
controller:
  min_delay: 0.8
  max_delay: 1.6
  click_variance: 9
logging:
  level: "DEBUG"
  file: "logs/custom_vision.log"
            """.strip(),
            encoding="utf-8",
        )

        settings = Settings.load_from_yaml(str(config_file))

        assert settings.vision.templates_dir == "custom_templates"
        assert settings.vision.capture_interval == 0.4
        assert settings.vision.template_threshold == 0.70
        assert settings.vision.button_threshold == 0.68
        assert settings.vision.debug_mode is True
        assert settings.vision.action_cooldown == 3.2
        assert settings.vision.discard_lock_timeout == 2.5
        assert settings.vision.nn_enabled is True
        assert settings.vision.nn_model_path == "models/custom_ann.xml"
        assert settings.vision.nn_labels_path == "models/custom_ann.labels.json"
        assert settings.vision.nn_fusion_weight == 0.55
        assert settings.vision.nn_min_confidence == 0.66
        assert settings.vision.nn_top_k == 7
        assert settings.vision.nn_priority is False
        assert settings.vision.browser_auto_open is True
        assert settings.vision.browser_url == "https://game.maj-soul.com/1/"
        assert settings.vision.browser_executable.endswith("msedge.exe")
        assert settings.vision.browser_wait_seconds == 1.8
        assert settings.vision.login_auto_fill is True
        assert settings.vision.auto_collect_dataset is True
        assert settings.vision.auto_collect_dir == "datasets/runtime_auto"
        assert settings.vision.auto_collect_min_score == 0.91
        assert settings.vision.auto_collect_include_unknown is True
        assert settings.vision.auto_collect_max_per_label == 345

        assert settings.account.username == "demo_user"
        assert settings.account.password == "demo_password"

        assert settings.controller.min_delay == 0.8
        assert settings.controller.max_delay == 1.6
        assert settings.controller.click_variance == 9

        assert settings.logging.level == "DEBUG"
        assert settings.logging.file == "logs/custom_vision.log"

    def test_load_from_yaml_missing_file(self, tmp_path):
        """缺失配置文件时应给出清晰报错"""
        missing = tmp_path / "missing_config.yaml"
        with pytest.raises(FileNotFoundError) as exc_info:
            Settings.load_from_yaml(str(missing))

        msg = str(exc_info.value)
        assert "配置文件不存在" in msg
        assert "config.example.yaml" in msg
