import asyncio
import json
import logging
from typing import List, Annotated, TypedDict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import nest_asyncio
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
# LangChain MCP 适配器
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# 允许在现有事件循环中嵌套运行（解决 Streamlit 中的异步冲突）
nest_asyncio.apply()

# 简单配置：设置根记录器的级别为 DEBUG
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# --- 1. 环境配置与中文字体修复 ---
def setup_chinese_font():
    """设置 Matplotlib 中文字体"""
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    try:
        sns.set_theme(font='Noto Sans CJK SC')
    except:
        pass


setup_chinese_font()


# --- 2. MCP SSE 客户端管理器 ---
class MCPClientManager:
    def __init__(self, sse_url: str):
        self.sse_url = sse_url
        self.client: Optional[MultiServerMCPClient] = None
        self.tools: List[BaseTool] = []

    async def connect(self):
        """通过 SSE 连接到正在运行的 MCP Server，并使用 MultiServerMCPClient 加载工具"""
        # ✅ 使用 MultiServerMCPClient 替换原有的手动 Session 管理和 load_mcp_tools
        # 在 langchain-mcp-adapters 0.1.0+ 中，推荐使用字典配置并包含 'transport' 键
        connections = {
            "sse_server": {
                "transport": "sse",
                "url": self.sse_url
            }
        }

        # 1. 初始化 MultiServerMCPClient
        self.client = MultiServerMCPClient(connections=connections)

        # 2. 获取所有已连接服务器的工具
        # 该方法会自动处理连接建立和工具加载
        self.tools = await self.client.get_tools()
        logging.info(f"成功加载了 {len(self.tools)} 个 MCP 工具")

    async def close(self):
        """
        关闭连接。
        注意：MultiServerMCPClient 目前没有公开的 aclose() 方法。
        """
        pass


# --- 3. LangGraph 定义 ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


async def chatbot_node(state: AgentState, config: RunnableConfig):
    # 使用支持 Tool Calling 的模型
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    mcp_manager = config["configurable"].get("mcp_manager")

    # 绑定加载的 MCP 工具
    llm_with_tools = llm.bind_tools(mcp_manager.tools)

    messages = state["messages"]
    system_msg = SystemMessage(content="你是一个桌面文件管理助手。你通过 MCP 协议与本地文件系统交互。")

    # 确保 SystemMessage 在首位
    filtered_messages = [m for m in messages if not isinstance(m, SystemMessage)]
    full_messages = [system_msg] + filtered_messages

    response = await llm_with_tools.ainvoke(full_messages)
    return {"messages": [response]}


def create_app(mcp_manager):
    workflow = StateGraph(AgentState)
    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("tools", ToolNode(mcp_manager.tools))
    workflow.set_entry_point("chatbot")
    workflow.add_conditional_edges("chatbot", tools_condition)
    workflow.add_edge("tools", "chatbot")
    return workflow.compile()


# --- 4. Streamlit UI 逻辑 ---
st.set_page_config(page_title="MCP SSE LangGraph 助手", layout="wide")

st.title("🤖 MCP (SSE) + LangGraph 智能助手")
st.markdown("当前架构：**OpenAI SDK + LangGraph + MultiServerMCPClient + Streamlit**")

# 初始化 Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_value" not in st.session_state:
    st.session_state.input_value = ""

# 侧边栏功能说明
with st.sidebar:
    st.header("服务配置")
    sse_url = st.text_input("MCP SSE URL", value="http://localhost:8001/sse")

    st.header("功能说明")
    st.info(
        "1. **SSE 协议**：连接到独立运行的 MCP 服务进程。\n2. **MultiServerMCPClient**：统一管理多个 MCP 服务器及其工具。\n3. **LangGraph**：管理 Agent 决策流。")

    st.header("试试这些问题")
    suggestions = [
        "桌面上有什么 txt 文件？",
        "统计桌面上所有 txt 文件的字数",
        "读取桌面上的第一个 txt 文件内容",
        "分析桌面上的文本文件并生成统计图表"
    ]
    for suggestion in suggestions:
        if st.button(suggestion):
            st.session_state.input_value = suggestion

    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.input_value = ""
        st.rerun()

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 处理用户输入
prompt = st.chat_input("帮我看看桌面上有什么 txt 文件？")

if st.session_state.input_value:
    prompt = st.session_state.input_value
    st.session_state.input_value = ""

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()


        async def run_agent():
            mcp_manager = MCPClientManager(sse_url)
            try:
                status_placeholder.status(f"正在连接到 MCP SSE 服务: {sse_url}...")
                await mcp_manager.connect()

                app = create_app(mcp_manager)
                # 传入历史消息
                initial_messages = []
                for m in st.session_state.messages[:-1]:
                    if m["role"] == "user":
                        initial_messages.append(HumanMessage(content=m["content"]))
                    else:
                        initial_messages.append(AIMessage(content=m["content"]))
                initial_messages.append(HumanMessage(content=prompt))

                inputs = {"messages": initial_messages}
                config = {"configurable": {"mcp_manager": mcp_manager}}

                full_response = ""
                async for output in app.astream(inputs, config=config):
                    for key, value in output.items():
                        if key == "chatbot":
                            last_msg = value["messages"][-1]
                            if last_msg.content:
                                full_response = last_msg.content
                                response_placeholder.markdown(full_response)
                        elif key == "tools":
                            status_placeholder.status(f"正在通过 SSE 调用远程工具...")

                st.session_state.messages.append({"role": "assistant", "content": full_response})

                # 统计绘图逻辑
                if "{" in full_response and "}" in full_response:
                    try:
                        import re
                        json_match = re.search(r'\{.*\}', full_response, re.DOTALL)
                        if json_match:
                            data = json.loads(json_match.group())
                            if isinstance(data, dict) and any(isinstance(v, (int, float)) for v in data.values()):
                                st.subheader("📊 数据可视化分析")
                                df = pd.DataFrame(list(data.items()), columns=['指标', '数值'])
                                df = df[pd.to_numeric(df['数值'], errors='coerce').notnull()]
                                fig, ax = plt.subplots(figsize=(10, 5))
                                sns.barplot(x='指标', y='数值', data=df, ax=ax, palette='viridis')
                                plt.xticks(rotation=45)
                                plt.title("文本统计维度分析")
                                st.pyplot(fig)
                    except:
                        pass

            except Exception as e:
                st.error(f"连接或调用失败: {str(e)}")
                st.info("请确保 MCP 服务已经以 SSE 模式启动并在运行中。")
                logging.exception("Agent execution failed")
            finally:
                await mcp_manager.close()
                status_placeholder.empty()


        # 修复：使用 get_event_loop() 配合 nest_asyncio 处理 Streamlit 的异步环境
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(run_agent())
        except Exception as e:
            st.error(f"运行出错: {e}")
            logging.exception("Agent execution failed")

        st.rerun()

