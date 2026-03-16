# 雀魂机器人 (Majsoul Bot) — 机器视觉版

一个基于 **机器视觉** 的雀魂（Majsoul）自动打牌机器人。
不依赖网络协议逆向工程，直接通过截图识别游戏画面、模拟鼠标点击操作。

> ⚠️ **免责声明**：本项目仅供学习和研究目的，使用本项目造成的任何后果由使用者自行承担。

---

## 核心原理

```
截图 → 模板匹配 + ANN_MLP 神经网络融合识别 → AI决策 → pyautogui模拟点击
```

与旧版 WebSocket 方案的对比：

| 特性 | 旧版（WebSocket） | 新版（机器视觉） |
|------|-----------------|----------------|
| 连接方式 | 需逆向协议/protobuf | 直接截图，无需协议 |
| 环境依赖 | 服务器不封锁 | 游戏窗口可见即可 |
| 识别准确率 | 100%（协议数据） | 取决于模板/训练数据质量 |
| 适应性 | 游戏更新后失效 | 重新捕获模板即可 |

---

## 项目结构

```
majsoul-bot/
├── majsoul_bot/
│   ├── vision/                    # 🆕 机器视觉模块
│   │   ├── screen_capture.py      # 屏幕截图 & 窗口检测
│   │   ├── tile_recognizer.py     # 麻将牌识别（模板 + NN 融合）
│   │   ├── tile_nn_classifier.py  # ANN_MLP 神经网络分类器
│   │   ├── game_state_detector.py # 游戏状态检测（按钮/轮次）
│   │   └── regions.py             # 屏幕区域坐标定义
│   ├── controller/                # 🆕 操作控制模块
│   │   └── mouse_controller.py    # 人性化鼠标控制
│   ├── vision_main.py             # 🆕 视觉机器人主入口
│   ├── ai/
│   │   ├── strategy.py            # AI 策略基类
│   │   └── simple_ai.py           # 简单打牌策略
│   ├── game_logic/
│   │   ├── tile.py                # 麻将牌定义
│   │   ├── hand.py                # 手牌管理
│   │   └── rules.py               # 规则判断
│   ├── main.py                    # 旧版 WebSocket 入口（保留）
│   └── config/
│       └── settings.py
├── templates/                     # 🆕 识别模板（需自行捕获）
│   ├── tiles/                     # 麻将牌模板图片
│   └── buttons/                   # 操作按钮模板图片
├── tools/
│   ├── capture_templates.py       # 🆕 模板捕获工具
│   ├── train_tile_ann.py          # 🆕 训练麻将牌 ANN 模型
│   └── calibrate_regions.py       # 🆕 区域校准工具
├── models/                        # 🆕 训练得到的 NN 模型（可选）
│   ├── tile_ann.xml
│   └── tile_ann.labels.json
├── config/
│   └── config.yaml                # 运行时配置（可选）
├── majsoul_bot/config/
│   └── config.example.yaml        # 配置模板（复制到 config/config.yaml）
├── logs/                          # 日志和调试截图
└── requirements.txt
```

---

## 快速开始

> 提示：视觉机器人配置文件位于 `config/config.yaml`（项目根目录）。
> 可从 `majsoul_bot/config/config.example.yaml` 复制并按需修改。

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要新增依赖：
- `opencv-python` — 图像处理与模板匹配
- `mss` — 高性能屏幕截图
- `pyautogui` — 鼠标控制
- `pygetwindow` — 窗口检测

### 2. 校准手牌区域（首次运行必须）

不同分辨率/窗口大小需要校准手牌坐标：

```bash
python tools/calibrate_regions.py
```

操作步骤：
1. 打开雀魂游戏，进入一局对局（确保手牌可见）
2. 在校准工具窗口中按**空格键**截图
3. 用鼠标**拖选整个手牌区域**（从第1张到第13张）
4. 按 `s` 保存校准结果
5. 按 `q` 退出

校准数据保存至 `config/vision_calibration.json`。

### 3. 捕获牌型模板

```bash
python tools/capture_templates.py
```

操作步骤：
1. 确保游戏对局中手牌可见
2. 按 **Enter** 截图
3. 根据提示，逐张输入手牌名称（如 `1m`、`5p`、`7z`）
4. 输入 `skip` 跳过当前位置，`quit` 结束

捕获技巧：
- 每种牌只需捕获 1 张样本
- 多局对局后可以捕获更多样本覆盖全部 34 种牌
- 也可以捕获按钮模板：`python tools/capture_templates.py --mode buttons`

牌名对照表：

| 类型 | 命名格式 | 说明 |
|------|---------|------|
| 万子 | `1m`~`9m` | 一万到九万 |
| 筒子 | `1p`~`9p` | 一筒到九筒 |
| 索子 | `1s`~`9s` | 一索到九索 |
| 字牌 | `1z`~`7z` | 东南西北白发中 |

### 4. （可选）训练神经网络模型（提升识别率）

当模板匹配识别率低（大量 unknown）时，建议训练 ANN 模型：

```bash
# 使用已标注模板训练（默认含轻量数据增强）
python tools/train_tile_ann.py --data-dir templates/tiles --output-model models/tile_ann.xml

# 默认会输出训练细节（网络结构、训练参数、进行中耗时）
# 可调节训练中日志频率
python tools/train_tile_ann.py --data-dir templates/tiles --train-log-interval 1.0

# 若只想保留关键结果输出
python tools/train_tile_ann.py --data-dir templates/tiles --quiet
```

训练完成后会生成：
- `models/tile_ann.xml`
- `models/tile_ann.labels.json`

机器人会按配置自动加载该模型并与模板分做融合。

### 5. 启动机器人

```bash
python majsoul_bot/vision_main.py
```

或使用命令行参数：

```bash
# 开启调试模式（保存带标注的截图到 logs/）
python majsoul_bot/vision_main.py --debug

# 调整操作延迟
python majsoul_bot/vision_main.py --min-delay 1.5 --max-delay 3.0

# 指定模板目录
python majsoul_bot/vision_main.py --templates my_templates/

# 使用指定配置文件
python majsoul_bot/vision_main.py --config config/config.yaml

# 调整识别与节奏参数
python majsoul_bot/vision_main.py --capture-interval 0.4 --tile-threshold 0.72 --button-threshold 0.70

# 显式禁用 NN，仅使用模板
python majsoul_bot/vision_main.py --no-nn

# 指定自定义 NN 模型并调整融合参数
python majsoul_bot/vision_main.py --nn-model-path models/tile_ann.xml --nn-fusion-weight 0.90 --nn-min-confidence 0.62
```

参数优先级：`命令行参数 > config/config.yaml > 程序内默认值`。

启动自动化说明（浏览器 + 登录填充）：
- 启动机器人时会按配置自动打开 `vision.browser_url`（默认雀魂地址）。
- 如需指定浏览器程序，可配置 `vision.browser_executable`（留空则使用系统默认浏览器）。
- 若设置 `vision.login_auto_fill: true`，并在 `account.username/password` 提供账号密码，
  机器人检测到登录页后会尝试自动填充并回车提交。

NN 优先识别说明：
- 默认启用 `vision.nn_priority: true`，手牌识别会优先采用 NN 结果。
- 同时默认 `nn_fusion_weight: 0.90`，进一步提高 NN 在融合排序中的权重。

---

## 工作流程

```
┌────────────────────────────────────────────────┐
│               主循环（每 0.5 秒）                │
│                                                 │
│  截图 → 游戏状态检测                             │
│         │                                       │
│         ├─ WIN_AVAILABLE    → 自动和牌           │
│         ├─ RIICHI_AVAILABLE → AI决策立直         │
│         ├─ OPERATION_AVAILABLE → 跳过/碰/吃/杠  │
│         ├─ MY_TURN_DISCARD  → 识别手牌+打牌      │
│         └─ WAITING          → 等待              │
└────────────────────────────────────────────────┘
```

### 状态检测机制

**操作按钮检测**（优先级最高）：
- 若 `templates/buttons/` 有模板 → 模板匹配
- 否则 → HSV 颜色检测（检测鲜艳的矩形色块）

**打牌回合检测**：
- 检测手牌区域的亮度（我的回合时手牌更亮）
- 检测第 14 张位置是否有牌（摸牌判断）

**牌面识别**：
- 从截图中按坐标切割每张牌的图像
- 与 `templates/tiles/` 中的模板逐一比对，得到模板分
- 可选加载 `models/tile_ann.xml` 做 NN 分类，得到概率分
- 对模板分与 NN 概率做融合排序；必要时使用 NN 高置信兜底

---

## 调试技巧

### 开启调试模式

```bash
python majsoul_bot/vision_main.py --debug
```

调试模式会定期保存 `logs/debug_latest.png`，包含：
- 绿框：手牌扫描区域
- 蓝框：摸牌位置
- 橙框：按钮扫描区域
- 红圈：检测到的操作按钮
- 顶部文字：当前游戏阶段

### 查看日志

```bash
# 实时查看日志
type logs\vision_bot.log
```

---

## 常见问题

### Q: 机器人没有反应，一直显示 WAITING？

原因：`MY_TURN_BRIGHTNESS` 阈值不适合当前游戏主题。

解决方案：
1. 开启调试模式查看截图
2. 检查手牌区域亮度是否达到阈值（默认 148）
3. 在 `game_state_detector.py` 中调整 `MY_TURN_BRIGHTNESS`

### Q: 牌面识别准确率低？

解决方案：
1. 确保模板图片清晰（不模糊、不遮挡）
2. 适当降低匹配阈值（`template_threshold: 0.70`）
3. 重新运行校准工具校准坐标
4. 多捕获几张样本（特别是赤宝牌）
5. 训练并启用 NN 模型（`python tools/train_tile_ann.py`）
6. 在 `config/config.yaml` 中调整融合参数（`nn_fusion_weight` / `nn_min_confidence`）

### Q: 点击位置偏移？

解决方案：
1. 重新运行 `python tools/calibrate_regions.py`
2. 确保游戏窗口没有最小化或被遮挡

### Q: 没有按钮模板，能用吗？

可以。无模板时自动使用颜色检测（检测画面中鲜艳的矩形色块）。
颜色检测准确率略低，建议在有按钮出现时手动捕获 1~2 张模板。

---

## 技术栈

| 组件 | 库 | 用途 |
|------|---|------|
| 屏幕截图 | `mss` | 高性能截图 |
| 图像处理 | `opencv-python` | 模板匹配、颜色检测 |
| 鼠标控制 | `pyautogui` | 模拟鼠标点击 |
| 窗口检测 | `pygetwindow` | 定位游戏窗口 |
| 数组处理 | `numpy` | 图像数组操作 |
| AI 策略 | 内置 | 简单打牌策略 |
| 异步框架 | `asyncio` | 非阻塞主循环 |

---

## 注意事项

1. **合规性**：使用游戏机器人可能违反雀魂服务条款，**本项目仅供学习研究**
2. **账号风险**：建议使用小号测试，避免主账号被封
3. **操作延迟**：默认设置了 1~2.5 秒随机延迟，避免被检测
4. **游戏更新**：游戏 UI 更新后可能需要重新捕获模板和校准

---

## 许可证

MIT License
