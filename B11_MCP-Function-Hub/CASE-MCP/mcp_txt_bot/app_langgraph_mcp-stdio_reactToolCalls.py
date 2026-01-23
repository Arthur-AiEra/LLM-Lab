import asyncio
import json
import logging
import os
import re
from contextlib import AsyncExitStack
from typing import List, Annotated, TypedDict, Optional, Type, Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

# MCP 官方 SDK
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 简单配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- 1. 环境配置与中文字体修复 ---
def setup_chinese_font():
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
        self.tools_info: List[Dict] = []

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
        self.tools_info = [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.inputSchema
            } for t in mcp_tools.tools
        ]

    async def call_tool(self, name: str, arguments: Dict) -> str:
        try:
            result = await self.session.call_tool(name, arguments)
            return "\n".join([c.text for c in result.content if hasattr(c, 'text')])
        except Exception as e:
            return f"Error calling tool {name}: {str(e)}"

    async def close(self):
        await self.exit_stack.aclose()


# --- 3. ReAct 逻辑定义 ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


async def chatbot_node(state: AgentState, config: RunnableConfig):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_retries=3)
    mcp_manager = config["configurable"].get("mcp_manager")

    # 构建 ReAct 提示词
    tools_desc = "\n".join([f"- {t['name']}: {t['description']}. Parameters: {json.dumps(t['parameters'])}" for t in
                            mcp_manager.tools_info])

    system_prompt = f"""你是一个桌面文件管理助手。你通过 MCP 协议与本地文件系统交互。
由于 API 限制，你不能直接使用 tool_calls。请遵循以下 ReAct 模式进行思考和行动：

1. **Thought**: 思考下一步该做什么。
2. **Action**: 如果需要调用工具，请严格按照以下 JSON 格式输出（不要包含其他文字）：
   {{"action": "工具名称", "action_input": {{"参数名": "参数值"}}}}
3. **Observation**: 你会收到工具执行的结果。
4. **Final Answer**: 当你获得足够信息后，直接回答用户。

可用工具列表：
{tools_desc}

请记住之前的对话上下文。如果用户说"好"或"继续"，请执行之前讨论的操作。"""

    messages = state["messages"]
    filtered_messages = [m for m in messages if not isinstance(m, SystemMessage)]
    full_messages = [SystemMessage(content=system_prompt)] + filtered_messages

    try:
        response = await llm.ainvoke(full_messages)
        return {"messages": [response]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"API 错误: {str(e)}")]}


async def tools_node(state: AgentState, config: RunnableConfig):
    mcp_manager = config["configurable"].get("mcp_manager")
    last_message = state["messages"][-1]
    content = last_message.content.strip()

    # 尝试解析 JSON Action
    try:
        # 提取 JSON 部分
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            action_data = json.loads(json_match.group())
            action = action_data.get("action")
            action_input = action_data.get("action_input", {})

            if action:
                result = await mcp_manager.call_tool(action, action_input)
                return {"messages": [HumanMessage(content=f"Observation: {result}")]}
    except Exception as e:
        return {"messages": [HumanMessage(content=f"Observation: Error parsing action: {str(e)}")]}

    return {"messages": []}


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and (
            "\"action\":" in last_message.content or "action" in last_message.content.lower()):
        return "tools"
    return "__end__"


def create_app():
    workflow = StateGraph(AgentState)
    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("tools", tools_node)
    workflow.set_entry_point("chatbot")
    workflow.add_conditional_edges("chatbot", should_continue)
    workflow.add_edge("tools", "chatbot")
    return workflow.compile()


# --- 4. Streamlit UI ---
st.set_page_config(page_title="MCP LangGraph 助手", layout="wide")

st.title("🤖 MCP + LangGraph 智能助手")
st.info("💡 此版本通过 ReAct 提示词模式运行，彻底绕过了 API 网关对 tools 参数的 500 校验错误。")
st.markdown("当前架构：**OpenAI SDK(提示词 ReAct 模式解析 tool_calls) + LangGraph + MCP 协议 (stdio) + Streamlit**")

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
        if "Observation:" in message.content: continue  # 隐藏中间过程
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
    st.session_state.langchain_messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()


        async def run_agent():
            server_path = os.path.join(os.getcwd(), "txt_counter.py")
            mcp_manager = MCPClientManager(server_path)
            try:
                status_placeholder.status("正在连接 MCP Server...")
                await mcp_manager.connect()

                app = create_app()
                inputs = {"messages": st.session_state.langchain_messages}
                config = {"configurable": {"mcp_manager": mcp_manager}}

                full_response = ""
                new_messages = []

                async for output in app.astream(inputs, config=config):
                    for key, value in output.items():
                        if "messages" in value:
                            new_msg = value["messages"][-1]
                            new_messages.append(new_msg)
                            if key == "chatbot" and not ("\"action\":" in new_msg.content):
                                full_response = new_msg.content
                                response_placeholder.markdown(full_response)
                            elif key == "tools":
                                status_placeholder.status("正在执行工具调用...")

                st.session_state.langchain_messages.extend(new_messages)
            finally:
                await mcp_manager.close()
                status_placeholder.empty()


        asyncio.run(run_agent())
        st.rerun()
