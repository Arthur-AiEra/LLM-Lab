import asyncio
import os
from typing import Annotated, List, TypedDict, Optional

import nest_asyncio
import streamlit as st
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# 允许在现有事件循环中嵌套运行（解决 Streamlit 中的异步冲突）
nest_asyncio.apply()

import logging

# 简单配置：设置根记录器的级别为 DEBUG
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Configuration ---
st.set_page_config(
    page_title="高德地图智能助手 (LangGraph)",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- MCP Tools Loading (Using MultiServerMCPClient) ---

class MCPManager:
    def __init__(self):
        self.client: Optional[MultiServerMCPClient] = None
        self.tools = []

    async def load_tools(self):
        """使用 MultiServerMCPClient 加载工具。"""
        server_config = {
            "amap_server": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@amap/amap-maps-mcp-server"],
                "env": {
                    **os.environ,
                    "AMAP_MAPS_API_KEY": os.getenv('AMAP_MAPS_API_KEY', '')
                }
            }
        }

        # 1. 初始化 MultiServerMCPClient
        self.client = MultiServerMCPClient(connections=server_config)

        # 2. ✅ 使用 get_tools() 替换 load_mcp_tools
        # 该方法会自动处理连接建立和工具加载
        self.tools = await self.client.get_tools()
        return self.tools


@st.cache_resource
def get_cached_mcp_manager():
    """缓存 MCP 管理器以避免重复加载工具。"""
    manager = MCPManager()
    try:
        # 使用当前事件循环运行异步加载
        loop = asyncio.get_event_loop()
        loop.run_until_complete(manager.load_tools())
        return manager
    except Exception as e:
        st.error(f"MCP 工具加载失败: {e}")
        return manager


# --- LangGraph Agent Definition ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def call_model(state: AgentState):
    """Call the LLM with the current messages."""
    manager = get_cached_mcp_manager()
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    ).bind_tools(manager.tools)

    system_message = SystemMessage(content=(
        "你扮演一个地图助手，你具有查询地图、规划路线、推荐景点等能力。"
        "你可以帮助用户规划旅游行程，查找地点，导航等。"
        "你应该充分利用高德地图的各种功能来提供专业的建议。"
    ))

    messages = state['messages']
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [system_message] + messages

    response = llm.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState):
    """Determine if the agent should continue calling tools or end."""
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END


def get_agent_app():
    manager = get_cached_mcp_manager()
    tool_node = ToolNode(manager.tools)

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# --- Streamlit UI ---

st.title("📍 高德地图智能助手")
st.markdown("基于 OpenAI & LangGraph & MCP 构建")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

suggestions = [
    '帮我规划上海一日游行程，主要想去外滩和迪士尼',
    '我在南京路步行街，帮我找一家评分高的本帮菜餐厅',
    '从浦东机场到外滩怎么走最方便？',
    '推荐上海三个适合拍照的网红景点',
    '帮我查找上海科技馆的具体地址和营业时间'
]

st.sidebar.title("快速查询")
for suggestion in suggestions:
    if st.sidebar.button(suggestion):
        st.session_state.user_input = suggestion

if prompt := st.chat_input("请输入您的问题..."):
    st.session_state.user_input = prompt

if "user_input" in st.session_state and st.session_state.user_input:
    user_query = st.session_state.user_input
    del st.session_state.user_input

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        try:
            app = get_agent_app()

            initial_messages = []
            for m in st.session_state.messages[:-1]:
                if m["role"] == "user":
                    initial_messages.append(HumanMessage(content=m["content"]))
                else:
                    initial_messages.append(AIMessage(content=m["content"]))
            initial_messages.append(HumanMessage(content=user_query))

            with st.status("正在思考并调用地图服务...", expanded=False) as status:
                # 使用 get_event_loop 配合 nest_asyncio
                loop = asyncio.get_event_loop()
                final_state = loop.run_until_complete(app.ainvoke({"messages": initial_messages}))
                status.update(label="处理完成！", state="complete", expanded=False)

            full_response = final_state["messages"][-1].content
            st.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"发生错误: {str(e)}")
            st.markdown("抱歉，处理您的请求时出现了错误。")
