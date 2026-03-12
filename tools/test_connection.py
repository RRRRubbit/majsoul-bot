"""
WebSocket 连接测试脚本
用于测试与雀魂服务器的连接
"""
import asyncio
import sys
import websockets


async def test_basic_connection(url: str):
    """测试基本 WebSocket 连接"""
    print(f"\n{'='*60}")
    print(f"测试连接: {url}")
    print(f"{'='*60}\n")

    try:
        print("🔄 正在连接...")
        async with websockets.connect(url, timeout=10) as ws:
            print("✅ 连接成功！")
            print(f"   - 协议: {ws.subprotocol or 'None'}")
            print(f"   - 状态: {ws.state.name}")

            # 尝试接收消息
            print("\n🔄 等待服务器消息...")
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=5)
                print(f"✅ 收到消息:")
                if isinstance(message, bytes):
                    print(f"   - 类型: 二进制")
                    print(f"   - 长度: {len(message)} bytes")
                    print(f"   - 前 50 字节: {message[:50]}")
                else:
                    print(f"   - 类型: 文本")
                    print(f"   - 内容: {message[:200]}")
            except asyncio.TimeoutError:
                print("⏱️  超时：5秒内未收到消息")

    except websockets.exceptions.InvalidStatusCode as e:
        print(f"❌ 连接被拒绝: HTTP {e.status_code}")
        print(f"   错误信息: {e}")
        return False

    except asyncio.TimeoutError:
        print("❌ 连接超时")
        return False

    except Exception as e:
        print(f"❌ 连接失败: {type(e).__name__}")
        print(f"   错误信息: {e}")
        return False

    return True


async def test_with_headers(url: str, headers: dict):
    """测试带请求头的连接"""
    print(f"\n{'='*60}")
    print(f"测试连接（带请求头）: {url}")
    print(f"{'='*60}\n")
    print("📋 请求头:")
    for key, value in headers.items():
        print(f"   - {key}: {value[:50]}...")

    try:
        print("\n🔄 正在连接...")
        async with websockets.connect(url, extra_headers=headers, timeout=10) as ws:
            print("✅ 连接成功！")

            # 尝试接收消息
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=5)
                print(f"✅ 收到消息:")
                if isinstance(message, bytes):
                    print(f"   - 类型: 二进制 ({len(message)} bytes)")
                else:
                    print(f"   - 类型: 文本 ({len(message)} chars)")
            except asyncio.TimeoutError:
                print("⏱️  超时：5秒内未收到消息")

    except websockets.exceptions.InvalidStatusCode as e:
        print(f"❌ 连接被拒绝: HTTP {e.status_code}")
        return False

    except Exception as e:
        print(f"❌ 连接失败: {type(e).__name__}: {e}")
        return False

    return True


async def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("雀魂 WebSocket 连接测试工具")
    print("="*60)

    # 测试的 URL 列表
    test_urls = [
        "wss://game.maj-soul.com:443",
        "wss://game.maj-soul.com:443/",
        "wss://game.mahjongsoul.com:443",
        "wss://gateway-v2.maj-soul.com:443",
    ]

    # 测试请求头
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Origin": "https://game.maj-soul.com",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    results = []

    # 测试基本连接
    print("\n\n" + "🧪 " * 20)
    print("第一阶段：基本连接测试")
    print("🧪 " * 20)

    for url in test_urls:
        success = await test_basic_connection(url)
        results.append((url, "基本连接", success))
        await asyncio.sleep(1)

    # 测试带请求头的连接
    print("\n\n" + "🧪 " * 20)
    print("第二阶段：带请求头的连接测试")
    print("🧪 " * 20)

    for url in test_urls:
        success = await test_with_headers(url, headers)
        results.append((url, "带请求头", success))
        await asyncio.sleep(1)

    # 显示测试结果汇总
    print("\n\n" + "="*60)
    print("测试结果汇总")
    print("="*60)

    successful = 0
    failed = 0

    for url, test_type, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{status} | {test_type:12} | {url}")
        if success:
            successful += 1
        else:
            failed += 1

    print(f"\n总计: {successful} 成功, {failed} 失败")

    # 给出建议
    print("\n" + "="*60)
    print("建议与说明")
    print("="*60)

    if successful == 0:
        print("""
❌ 所有连接测试均失败

可能的原因：
1. 网络连接问题
2. 防火墙阻止
3. 服务器地址不正确
4. 需要特殊的认证流程

建议：
1. 检查网络连接
2. 使用浏览器打开 https://game.maj-soul.com/ 验证可访问性
3. 查看 Chrome DevTools (F12) → Network → WS 标签
4. 参考 docs/CONNECTION_TROUBLESHOOTING.md 文档
        """)
    else:
        print(f"""
✅ 部分连接成功 ({successful}/{len(results)})

说明：
- 基本连接成功说明网络可达
- 但可能仍需要特定的认证流程
- 建议抓包分析完整的通信流程

下一步：
1. 使用 Chrome DevTools 抓取真实的 WebSocket 通信
2. 分析消息格式（可能是 Protocol Buffers）
3. 实现认证和协议解析
        """)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  测试已中断")
        sys.exit(0)
