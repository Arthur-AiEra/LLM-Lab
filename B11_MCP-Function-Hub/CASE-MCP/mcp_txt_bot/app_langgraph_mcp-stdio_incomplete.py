import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack
from typing import List, Annotated, TypedDict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
# MCP 官方 SDK
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field, create_model
from typing_extensions import deprecated

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


# --- 2. MCP 协议客户端管理器 ---
class MCPClientManager:
    def __init__(self, server_script_path: str):
        self.server_script_path = server_script_path
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self.tools: List[StructuredTool] = []

    async def connect(self):
        server_params = StdioServerParameters(
            command="python3",
            args=[self.server_script_path],
            env=os.environ.copy()
        )
        read_stream, write_stream = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await self.session.initialize()
        mcp_tools = await self.session.list_tools()
        self.tools = [self._make_langchain_tool(t) for t in mcp_tools.tools]

    # 优化方案（参考 app_langgraph_mcp-stdio_load_mcp_tools.py）
    # 在 MCPClientManager.connect 方法中，引入了 langchain_mcp_adapters.tools.load_mcp_tools
    # 利用适配器自动处理 MCP 工具到 LangChain StructuredTool 的转换，包括参数校验和异步调用封装，极大地提高了代码的可维护性和鲁棒性
    # 移除手动包装逻辑：完全删除了原有的 _make_langchain_tool 函数及其复杂的动态 Schema 构建逻辑
    @deprecated("Use app_langgraph_mcp-stdio_load_mcp_tools.py load_mcp_tools(self.session) instead")
    def _make_langchain_tool(self, mcp_tool) -> StructuredTool:
        """
        ✅ 关键修复：优化动态 Schema 构建，解决 API 500 兼容性问题。
        """
        schema_properties = mcp_tool.inputSchema.get("properties", {})
        required_fields = mcp_tool.inputSchema.get("required", [])

        # 如果没有参数，使用一个简单的空模型，避免生成复杂的空 properties 结构
        if not schema_properties:
            class EmptySchema(BaseModel):
                pass

            args_schema = EmptySchema
        else:
            fields = {}
            for name, prop in schema_properties.items():
                field_type = str
                if prop.get("type") == "integer":
                    field_type = int
                elif prop.get("type") == "boolean":
                    field_type = bool

                default_value = ... if name in required_fields else None
                fields[name] = (field_type, Field(default=default_value, description=prop.get("description", "")))

            args_schema = create_model(f"{mcp_tool.name}Schema", **fields)

        async def call_tool(**kwargs):
            result = await self.session.call_tool(mcp_tool.name, kwargs)
            return "\n".join([c.text for c in result.content if hasattr(c, 'text')])

        return StructuredTool.from_function(
            coroutine=call_tool,
            name=mcp_tool.name,
            description=mcp_tool.description,
            args_schema=args_schema
        )

    async def close(self):
        await self.exit_stack.aclose()


# --- 3. LangGraph 定义 ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


async def chatbot_node(state: AgentState, config: RunnableConfig):
    # 这里如果采用 gpt-4.1-mini llm, 会报 ’500 Internal Server Error‘
    # gpt-4.1-mini 作为一个较新的模型名称，中转 API 网关（api.fe8.cn）虽然声称支持 OpenAI 格式，但在处理 tools（Function Calling）这种复杂协议时，往往因为后端逻辑编写不严谨而导致 500 错误
    # 可以换用其他模型，或不再发送 tools 参数，采用ReAct 模式：将工具说明直接写在 SystemMessage 里
    llm = ChatOpenAI(model="deepseek-v3", temperature=0) # gpt-4.1-mini, "deepseek-v3
    mcp_manager = config["configurable"].get("mcp_manager")
    llm_with_tools = llm.bind_tools(mcp_manager.tools)

    messages = state["messages"]
    system_msg = SystemMessage(content="你是一个桌面文件管理助手。你通过 MCP 协议与本地文件系统交互")

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
st.set_page_config(page_title="MCP LangGraph 助手", layout="wide")

st.title("🤖 MCP + LangGraph 智能助手")
st.markdown("当前架构：**OpenAI SDK + LangGraph + MCP 协议 (stdio) + Streamlit**")

if "langchain_messages" not in st.session_state:
    st.session_state.langchain_messages = []
if "input_value" not in st.session_state:
    st.session_state.input_value = ""

# ========== 侧边栏（集成快捷消息功能）==========
with st.sidebar:
    st.header("功能说明")
    st.info(
        "1. **MCP 协议**：通过标准协议调用外部工具。\n2. **LangGraph**：管理复杂的 Agent 决策流。\n3. **可视化**：自动解析统计数据并绘图。")

    # ✅ 新增：快捷消息按钮
    st.header("💡 试试这些问题")
    suggestions = [
        "桌面上有什么 txt 文件？",
        "统计桌面上所有 txt 文件的字数",
        "读取桌面上的第一个 txt 文件内容",
        "分析桌面上的文本文件并生成统计图表"
    ]
    for suggestion in suggestions:
        if st.button(suggestion, key=f"suggestion_{suggestion}"):
            st.session_state.input_value = suggestion
            st.rerun()  # 立即触发重新运行以处理输入

    st.divider()  # 分隔线

    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.langchain_messages = []
        st.session_state.input_value = ""
        st.rerun()

# 显示历史消息
for message in st.session_state.langchain_messages:
    if isinstance(message, (HumanMessage, AIMessage)) and message.content:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

# 处理用户输入
prompt = st.chat_input("帮我看看桌面上有什么 txt 文件？")

# ✅ 如果点击了快捷按钮，覆盖 prompt
if st.session_state.input_value:
    prompt = st.session_state.input_value
    st.session_state.input_value = ""  # 清空以便下次输入

if prompt:
    user_msg = HumanMessage(content=prompt)
    st.session_state.langchain_messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()
        plot_placeholder = st.container()


        async def run_agent():
            server_path = os.path.join(os.getcwd(), "txt_counter.py")
            mcp_manager = MCPClientManager(server_path)
            try:
                status_placeholder.status("正在连接 MCP Server...")
                await mcp_manager.connect()

                app = create_app(mcp_manager)
                inputs = {"messages": st.session_state.langchain_messages}
                config = {"configurable": {"mcp_manager": mcp_manager}}

                full_response = ""
                new_messages = []

                async for output in app.astream(inputs, config=config):
                    for key, value in output.items():
                        if "messages" in value:
                            new_msg = value["messages"][-1]
                            new_messages.append(new_msg)

                            if key == "chatbot":
                                if new_msg.content:
                                    full_response = new_msg.content
                                    response_placeholder.markdown(full_response)
                            elif key == "tools":
                                status_placeholder.status(f"正在执行 MCP 工具调用...")

                st.session_state.langchain_messages.extend(new_messages)

                if "{" in full_response and "}" in full_response:
                    try:
                        import re
                        json_match = re.search(r'\{.*\}', full_response, re.DOTALL)
                        if json_match:
                            data = json.loads(json_match.group())
                            if isinstance(data, dict) and any(isinstance(v, (int, float)) for v in data.values()):
                                with plot_placeholder:
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

            finally:
                await mcp_manager.close()
                status_placeholder.empty()


        # 运行 Agent
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            ai_response = loop.run_until_complete(run_agent())
        finally:
            pass

        st.rerun()
