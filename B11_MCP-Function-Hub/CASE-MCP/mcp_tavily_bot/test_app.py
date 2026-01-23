"""测试脚本 v2：验证基于 langchain-mcp-adapters 的重构代码是否正常工作"""

import asyncio
import logging
import os
import subprocess

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_imports():
    """测试所有必要的导入"""
    print("测试 1: 检查依赖包导入...")
    try:
        import streamlit
        import matplotlib
        import seaborn
        import pandas
        from langchain_openai import ChatOpenAI
        from langchain.agents import create_agent
        from langchain_mcp_adapters.client import MultiServerMCPClient
        print("  ✅ 所有核心依赖包导入成功！\n")
        return True
    except ImportError as e:
        print(f"  ❌ 导入失败: {e}\n")
        return False

def test_environment():
    """测试运行环境（Node.js, npx, API Keys）"""
    print("测试 2: 检查运行环境...")
    success = True
    try:
        # 检查 Node.js
        node_version = subprocess.check_output(['node', '--version'], text=True).strip()
        print(f"  ✅ Node.js: {node_version}")
        
        # 检查 npx
        npx_version = subprocess.check_output(['npx', '--version'], text=True).strip()
        print(f"  ✅ npx: {npx_version}")
        
        # 检查 API Keys
        openai_key = os.getenv('OPENAI_API_KEY')
        tavily_key = os.getenv('TAVILY_API_KEY')
        
        if openai_key:
            print(f"  ✅ OPENAI_API_KEY 已配置")
        else:
            print(f"  ⚠️  OPENAI_API_KEY 未设置")
            success = False
            
        if tavily_key:
            print(f"  ✅ TAVILY_API_KEY 已配置")
        else:
            print(f"  ⚠️  TAVILY_API_KEY 未设置")
            success = False
            
        print()
        return success
    except Exception as e:
        print(f"  ❌ 环境检查失败: {e}\n")
        return False

async def test_agent_logic():
    """测试 Agent 核心逻辑（连接 MCP 并调用 LLM）"""
    print("测试 3: 验证 Agent 核心逻辑 (MCP + LLM)...")
    
    server_config = {
        "tavily_search": {
            "command": "npx",
            "args": ["-y", "tavily-mcp@0.1.4"],
            "transport": "stdio",
            "env": {**os.environ, "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", "")}
        }
    }
    
    try:
        # 1. 初始化 MCP 客户端
        client = MultiServerMCPClient(server_config)
        print("  🔄 正在连接 MCP Server 并获取工具...")
        tools = await client.get_tools()
        print(f"  ✅ 成功获取 {len(tools)} 个工具")

        # 2. 初始化 LLM
        llm = ChatOpenAI(model="gpt-4o", temperature=0, timeout=30)
        
        # 3. 创建 Agent
        agent = create_agent(llm, tools)
        print("  ✅ Agent 创建成功")

        # 4. 简单调用测试
        print("  🔄 正在执行简单搜索测试...")
        test_input = "Say 'Hello' and tell me the current weather in London."
        result = await agent.ainvoke({"messages": [HumanMessage(content=test_input)]})
        
        final_msg = result["messages"][-1].content
        print(f"  ✅ Agent 响应成功")
        print(f"  响应预览: {final_msg[:100]}...")
        print()
        return True
    except Exception as e:
        print(f"  ❌ Agent 逻辑测试失败: {e}\n")
        return False

async def run_all_tests():
    """运行所有测试项"""
    print("=" * 60)
    print("  Tavily MCP 助手重构版 - 测试套件 v2")
    print("=" * 60)
    print()
    
    results = []
    
    # 1. 导入测试
    results.append(("依赖包导入", test_imports()))
    
    # 2. 环境测试
    results.append(("运行环境检查", test_environment()))
    
    # 3. Agent 逻辑测试 (仅在环境检查通过时运行)
    if results[-1][1] and results[-2][1]:
        results.append(("Agent 核心逻辑", await test_agent_logic()))
    else:
        print("⚠️  由于环境或导入测试未通过，跳过 Agent 逻辑测试。\n")
        results.append(("Agent 核心逻辑", False))
    
    # 汇总结果
    print("=" * 60)
    print("  测试结果汇总")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:25s} {status}")
    
    print()
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！您可以放心运行应用：")
        print("  streamlit run app_mcp_tavily_bot.py")
    else:
        print("\n⚠️  部分测试未通过，请根据上方错误信息进行排查。")
    print()

if __name__ == '__main__':
    asyncio.run(run_all_tests())
