# 雀魂机器人连接问题排查指南

## 🔴 常见错误：HTTP 403 Forbidden

如果您看到以下错误：
```
ERROR | Failed to connect: server rejected WebSocket connection: HTTP 403
```

这是**预期的行为**，因为雀魂服务器有反爬虫和安全防护措施。

## ❓ 为什么会出现 403 错误？

1. **协议未实现**: 本项目目前只实现了基础框架，雀魂的完整通信协议需要逆向工程
2. **认证缺失**: 雀魂使用 Protocol Buffers 进行消息序列化，需要特定的认证流程
3. **WebSocket 路径错误**: 服务器的真实 WebSocket 端点可能不是根路径
4. **请求头验证**: 服务器可能验证特定的请求头或 Cookie

## 🔧 解决方案

### 方案 1: 抓包分析（推荐）

使用工具分析雀魂客户端的真实通信协议：

1. **安装工具**:
   - Windows: Fiddler, Wireshark
   - Chrome DevTools (F12 → Network → WS)

2. **抓包步骤**:
   ```bash
   # 1. 打开雀魂网页版
   https://game.maj-soul.com/1/

   # 2. 打开 Chrome DevTools (F12)
   # 3. 切换到 Network 标签
   # 4. 过滤 WS (WebSocket)
   # 5. 登录游戏
   # 6. 观察 WebSocket 连接
   ```

3. **查找信息**:
   - WebSocket URL（包含路径和参数）
   - 请求头（Headers）
   - 认证方式
   - 消息格式（Protocol Buffers）

### 方案 2: 使用现有项目

参考已有的雀魂机器人项目：

- [mjai](https://github.com/gimite/mjai) - 麻将 AI 框架
- [mahjong](https://github.com/MahjongRepository) - 雀魂相关工具集
- [majsoul-api](https://github.com/yuanfengyun/majsoul_api) - 雀魂 API 封装

### 方案 3: 修改配置文件

如果您已经知道正确的 WebSocket 端点，修改配置：

```yaml
# majsoul_bot/config/config.yaml
server:
  host: "game.maj-soul.com"
  port: 443
  use_ssl: true
  path: "/gateway"  # 添加正确的路径
```

## 📝 需要实现的功能

要让机器人真正工作，需要实现：

### 1. Protocol Buffers 解析

雀魂使用 protobuf 进行消息序列化，需要：

```bash
# 安装 protobuf
pip install protobuf

# 需要 .proto 文件定义
# 通过逆向工程或官方文档获取
```

### 2. 认证流程

```python
# 示例：可能的认证流程
async def authenticate(self):
    # 1. 获取认证 token
    # 2. 建立 WebSocket 连接
    # 3. 发送认证消息
    # 4. 等待认证响应
    pass
```

### 3. 消息序列化

```python
# 示例：protobuf 消息
import majsoul_pb2  # 需要生成的 protobuf 文件

# 创建登录消息
login_req = majsoul_pb2.ReqLogin()
login_req.account = "username"
login_req.password = "password"

# 序列化
data = login_req.SerializeToString()

# 发送
await ws_client.send_binary(data)
```

## 🧪 测试连接

### 基本连接测试

```python
# test_connection.py
import asyncio
import websockets

async def test_connection():
    url = "wss://game.maj-soul.com:443"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Origin": "https://game.maj-soul.com"
    }

    try:
        async with websockets.connect(url, extra_headers=headers) as ws:
            print("✓ 连接成功")
            # 等待消息
            message = await asyncio.wait_for(ws.recv(), timeout=5)
            print(f"收到消息: {message}")
    except Exception as e:
        print(f"✗ 连接失败: {e}")

asyncio.run(test_connection())
```

### 使用 Chrome DevTools 查看

1. 访问雀魂网页版: https://game.maj-soul.com/1/
2. 按 F12 打开开发者工具
3. 切换到 Network 标签
4. 点击 WS 过滤器
5. 登录游戏
6. 查看 WebSocket 连接详情：
   - Request URL
   - Request Headers
   - Messages（消息格式）

## 🎯 下一步

1. **获取协议定义**: 通过逆向工程或社区资源获取 .proto 文件
2. **实现认证**: 根据抓包结果实现登录流程
3. **消息解析**: 实现 protobuf 消息的序列化和反序列化
4. **游戏逻辑**: 完善游戏状态管理和 AI 决策

## ⚠️ 重要提醒

- 使用机器人可能违反雀魂服务条款
- 仅用于学习和研究目的
- 使用测试账号，避免封号风险
- 遵守游戏规则和服务条款

## 📚 参考资源

- [WebSocket 协议](https://datatracker.ietf.org/doc/html/rfc6455)
- [Protocol Buffers](https://developers.google.com/protocol-buffers)
- [雀魂官网](https://mahjongsoul.com/)
- [Python websockets 文档](https://websockets.readthedocs.io/)

## 💡 提示

如果您只是想测试项目的其他功能（游戏逻辑、AI 决策等），可以：

1. **单元测试**: 运行 `pytest tests/` 测试游戏逻辑
2. **模拟数据**: 修改代码使用模拟的游戏数据进行测试
3. **离线模式**: 实现一个离线的游戏模拟器

## 🤝 贡献

如果您成功实现了雀魂协议的解析，欢迎：

1. Fork 本项目
2. 添加协议实现
3. 提交 Pull Request
4. 分享您的经验

---

**最后更新**: 2026-03-12
