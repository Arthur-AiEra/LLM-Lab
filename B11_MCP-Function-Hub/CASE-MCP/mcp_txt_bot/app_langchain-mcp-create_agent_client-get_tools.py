import asyncio
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

import logging

# 简单配置：设置根记录器的级别为 DEBUG
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')



# --- 1. 环境配置与中文字体修复 ---
def setup_chinese_font():
    """设置 Matplotlib 中文字体"""
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(font='Noto Sans CJK SC')


setup_chinese_font()

# --- 2. Streamlit UI 配置 ---
st.set_page_config(page_title="MCP Agent Web UI", layout="wide")

st.title("🤖 Langchain agent + langchain_mcp_adapters.client + Streamlit 助手")
st.markdown("当前架构：**OpenAI SDK + Langchain + MCP + Streamlit**")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_value" not in st.session_state:
    st.session_state.input_value = ""

# --- 3. 侧边栏配置 ---
with st.sidebar:
    st.header("🔧 服务配置")

    # MCP Server 配置方式选择
    transport_type = st.selectbox(
        "传输协议",
        ["sse", "stdio"],
        index=0
    )

    if transport_type == "sse":
        sse_url = st.text_input("MCP SSE URL", value="http://127.0.0.1:8001/sse")
    else:
        stdio_command = st.text_input("命令", value="python3")
        stdio_args = st.text_input("参数 (逗号分隔)", value="txt_counter.py")

    st.divider()

    st.header("💡 试试这些问题")
    suggestions = [
        "桌面上有什么 txt 文件？",
        "统计桌面上所有 txt 文件的字数",
        "读取桌面上的第一个 txt 文件内容",
        "分析桌面上的文本文件并生成统计图表"
    ]
    for suggestion in suggestions:
        if st.button(suggestion):
            st.session_state.input_value = suggestion

    st.divider()

    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.input_value = ""
        st.rerun()

# --- 4. 显示历史消息 ---
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)


# --- 5. 核心 Agent 运行逻辑 ---
async def run_agent(user_input: str, server_config: dict):
    """
    使用 langchain-mcp-adapters 连接 MCP Server 并运行 Agent
    """
    # 修正：直接初始化客户端，而不是使用 'async with'
    client = MultiServerMCPClient(server_config)

    try:
        # 1. 获取所有 MCP 工具（已自动转换为 LangChain Tool）
        # 修正：调用 await client.get_tools() 来异步获取工具
        tools = await client.get_tools()

        # 2. 创建 LLM
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

        # 3. 使用 LangGraph 预构建的 ReAct Agent
        agent = create_agent(llm, tools)

        # 4. 准备输入消息
        input_messages = st.session_state.messages

        # 5. 运行 Agent
        result = await agent.ainvoke({"messages": input_messages})

        # 6. 提取最终回复
        final_message = result["messages"][-1]
        return final_message
    except Exception as e:
        logging.error(f"run_agent error, {str(e)}")
        return f"run_agent error, {str(e)}"


def extract_and_visualize(response_text: str):
    """尝试从响应中提取 JSON 数据并可视化"""
    if "{" not in response_text or "}" not in response_text:
        return

    try:
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            return

        data = json.loads(json_match.group())
        if not isinstance(data, dict):
            return
        if not any(isinstance(v, (int, float)) for v in data.values()):
            return

        st.subheader("📊 数据可视化分析")
        df = pd.DataFrame(list(data.items()), columns=['指标', '数值'])
        df = df[pd.to_numeric(df['数值'], errors='coerce').notnull()]

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x='指标', y='数值', data=df, ax=ax, palette='viridis')
        plt.xticks(rotation=45)
        plt.title("文本统计维度分析")
        st.pyplot(fig)

    except (json.JSONDecodeError, Exception):
        pass


# --- 6. 处理用户输入 ---
prompt = st.chat_input("帮我看看桌面上有什么 txt 文件？")

if st.session_state.input_value:
    prompt = st.session_state.input_value
    st.session_state.input_value = ""

if prompt:
    # 显示用户消息
    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()

        try:
            # 构建 MCP Server 配置
            if transport_type == "sse":
                server_config = {
                    "desktop_tools": {
                        "url": sse_url,
                        "transport": "sse",
                    }
                }
            else:
                server_config = {
                    "desktop_tools": {
                        "command": stdio_command,
                        "args": [arg.strip() for arg in stdio_args.split(",")],
                        "transport": "stdio",
                    }
                }

            status_placeholder.status("🔄 正在连接 MCP Server 并执行任务...")

            # 运行 Agent
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                ai_response = loop.run_until_complete(run_agent(prompt, server_config))
            finally:
                # 不要关闭 loop，让 Streamlit 管理
                pass


            # --- 核心部分 ---
            # 检查 run_agent 的返回类型
            if isinstance(ai_response, AIMessage):
                # 成功情况：返回的是 AIMessage 对象
                response_content = ai_response.content
                response_placeholder.markdown(response_content)
                st.session_state.messages.append(ai_response)
                extract_and_visualize(response_content)
            elif isinstance(ai_response, str):
                # 失败情况：返回的是错误字符串
                response_content = ai_response
                st.error(f"❌ Agent 执行失败: {response_content}")
            else:
                # 其他意外情况
                st.error(f"❌ 收到未知的响应类型: {type(ai_response)}")

        except Exception as e:
            st.error(f"❌ 连接或调用失败: {str(e)}")
            import traceback

            st.code(traceback.format_exc())
        finally:
            status_placeholder.empty()

    # --- '上一个assistant 消息会显示2次'关键修复点: 强制触发 Rerun ---
    # 这一步非常重要。它会结束当前的“临时渲染”周期，
    # 重新运行脚本。在重新运行周期中，prompt 为空，
    # 所有的消息（包括刚刚存入 state 的 assistant 消息）
    # 都会由顶部的“历史消息渲染”循环统一绘制，从而避免重复。
    st.rerun()