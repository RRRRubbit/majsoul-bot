# 🔧 牌型识别问题快速修复指南

## 问题症状

日志显示：
```
识别到手牌: ['unknown_0', 'unknown_1', ..., 'unknown_13']
⚠  无模板：使用视觉聚类/位置模式打牌
```

## 根本原因

常见并发根因：

1. **没有捕获牌型模板** → 无法进行图像匹配 → 所有牌识别为 `unknown`
2. **模板尺寸与实时切图尺寸偏差较大**（如窗口缩放后）→ 匹配分普遍过低
3. **仅靠模板匹配鲁棒性不足**（光照/特效/边缘遮挡）→ 识别波动大

---

## 🚀 快速修复步骤

### 步骤 1: 运行调试工具

```bash
python tools/debug_recognition.py
```

这个工具会自动：
- ✅ 检查模板文件是否存在
- ✅ 检查校准数据是否正确
- ✅ 截图并提取所有牌的图片
- ✅ 保存到 `logs/debug_recognition/tiles/`

### 步骤 2: 手动创建模板

打开提取的牌图片目录：
```
logs/debug_recognition/tiles/
```

您会看到类似这样的文件：
```
tile_00_pos_0.png
tile_01_pos_1.png
...
tile_13_pos_13.png
```

操作步骤：
1. **识别每张牌**：打开图片，看清楚是什么牌
2. **重命名**：将文件重命名为标准格式：
   ```
   tile_00_pos_0.png → 1m.png  (一万)
   tile_01_pos_1.png → 5p.png  (五筒)
   tile_02_pos_2.png → 9s.png  (九索)
   tile_03_pos_3.png → 1z.png  (东风)
   ```
3. **复制到模板目录**：
   ```bash
   # Windows
   copy logs\debug_recognition\tiles\*.png templates\tiles\

   # Linux/Mac
   cp logs/debug_recognition/tiles/*.png templates/tiles/
   ```

### 步骤 3: 验证修复

重新运行机器人：
```bash
python majsoul_bot/vision_main.py --debug
```

如果仍有较多 `unknown`，建议继续执行「高级方案：训练 NN 模型」。

查看日志，应该看到：
```
✅ 识别到手牌: ['1m', '2m', '3m', '4m', '5m', ...]
```

---

## 📖 牌名对照表

| 牌类型 | 编号 | 牌名 | 说明 |
|--------|------|------|------|
| 万子 | 1m-9m | `1m`, `2m`, ..., `9m` | 一万到九万 |
| 筒子 | 1p-9p | `1p`, `2p`, ..., `9p` | 一筒到九筒 |
| 索子 | 1s-9s | `1s`, `2s`, ..., `9s` | 一索到九索 |
| 字牌 | 1z-7z | `1z`, `2z`, `3z`, `4z`, `5z`, `6z`, `7z` | 东、南、西、北、中、发、白 |

**重要**：字牌顺序是固定的！
- `1z` = 东
- `2z` = 南
- `3z` = 西
- `4z` = 北
- `5z` = 中（红中）
- `6z` = 发（绿发）
- `7z` = 白（白板）

---

## 🔍 高级：使用自动捕获工具

如果您想要更高质量的模板：

```bash
python tools/capture_templates.py
```

交互式操作：
1. 确保游戏在对局中（手牌可见）
2. 按 **Enter** 截图
3. 工具会依次询问每张牌的名称
4. 输入牌名（如 `1m`）或 `skip` 跳过
5. 输入 `quit` 结束

---

## 🧠 高级方案：训练并启用神经网络识别（推荐）

当模板匹配分长期偏低（例如候选最高常在 0.3~0.6）时，建议训练 ANN 模型并与模板融合：

```bash
# 1) 使用已标注样本训练（默认含轻量增强）
python tools/train_tile_ann.py --data-dir templates/tiles --output-model models/tile_ann.xml

# 2) 启动机器人（默认会自动加载 models/tile_ann.xml）
python majsoul_bot/vision_main.py --debug
```

可选：如果要指定自定义模型与融合参数：

```bash
python majsoul_bot/vision_main.py \
  --nn-model-path models/tile_ann.xml \
  --nn-fusion-weight 0.70 \
  --nn-min-confidence 0.62
```

---

## ⚙️ 调整识别参数

如果模板已存在但识别率低，编辑配置文件：

```yaml
# majsoul_bot/config/config.yaml
vision:
  template_threshold: 0.70  # 降低模板阈值（默认 0.75）
  nn_enabled: true          # 启用 NN 融合识别
  nn_fusion_weight: 0.65    # 融合权重（越大越偏向 NN）
  nn_min_confidence: 0.58   # NN 兜底置信度
  debug_mode: true          # 开启调试模式
```

**阈值说明**：
- `0.90` - 非常严格，几乎要完全一致
- `0.75` - 推荐值（默认）
- `0.60` - 宽松，可能有误识别
- `0.50` - 太宽松，不推荐

---

## 🐛 调试技巧

### 查看识别过程的可视化

开启调试模式后，会定期保存标注图片：
```
logs/debug_latest.png
```

图中会显示：
- 🟢 绿框 = 成功识别的牌
- 🔴 红框 = 未识别的牌 (unknown)
- 🔵 蓝框 = 摸牌位置

### 实时查看日志

```bash
# Windows PowerShell
Get-Content logs\vision_bot.log -Wait -Tail 20

# Linux/Mac
tail -f logs/vision_bot.log
```

### 检查模板匹配得分

调试工具的 `test_template_matching` 会显示每个模板的匹配分数：

```bash
python tools/debug_recognition.py
```

输出示例：
```
📊 匹配结果（阈值 = 0.75）:
   ✅ 0.923 ███████████████████████████     1m
   ❌ 0.652 ███████████████████            2m
   ❌ 0.543 ████████████████               3m
```

如果最高分也低于阈值 → 降低阈值或重新捕获模板

---

## 🎯 常见问题

### Q: 只有部分牌识别失败？

**原因**：缺少这些牌的模板

**解决**：
1. 运行 `python tools/debug_recognition.py` 提取所有牌
2. 只复制未识别的牌（如 `unknown_5`）到模板目录
3. 重命名为正确的牌名

### Q: 识别完全不准确？

**可能原因**：
1. 校准数据不正确 → 重新运行 `python tools/calibrate_regions.py`
2. 游戏分辨率/窗口大小改变 → 重新校准
3. 模板质量差（模糊、遮挡）→ 重新捕获

### Q: 有时准确有时不准？

**原因**：游戏画面变化（光照、特效、动画）

**解决**：
1. 多捕获几张同一牌的样本
2. 降低匹配阈值
3. 在静止画面时截图

---

## 📚 相关工具

| 工具 | 用途 | 命令 |
|------|------|------|
| 校准工具 | 设置手牌区域坐标 | `python tools/calibrate_regions.py` |
| 模板捕获 | 交互式捕获牌型 | `python tools/capture_templates.py` |
| 调试工具 | 诊断识别问题 | `python tools/debug_recognition.py` |

---

## 💡 最佳实践

1. **首次使用**：
   - 先校准 → 再捕获模板 → 最后运行机器人

2. **模板质量**：
   - 在对局静止时截图（避免动画）
   - 每种牌至少1张样本
   - 清晰、无遮挡、无模糊

3. **定期维护**：
   - 游戏更新后重新校准
   - 发现识别错误时补充模板
   - 定期查看调试日志

---

如果以上步骤都无法解决，请提供：
- `logs/debug_recognition/` 中的截图
- `config/vision_calibration.json` 内容
- 完整的错误日志

祝使用愉快！ 🎮
