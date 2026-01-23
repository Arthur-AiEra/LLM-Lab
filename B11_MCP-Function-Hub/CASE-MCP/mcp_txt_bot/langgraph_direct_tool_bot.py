import json
import json
import logging
import re
from typing import List, Annotated, TypedDict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# 1. 导入工具
try:
    from txt_counter import (
        count_desktop_txt_files,
        list_desktop_txt_files,
        read_txt_file
    )

    tools = [count_desktop_txt_files, list_desktop_txt_files, read_txt_file]
except ImportError:
    st.error("无法导入 txt_counter 工具，请确保 txt_counter.py 在当前目录下。")
    tools = []


# --- 2. 环境配置与中文字体修复 ---
def setup_chinese_font():
    """设置 Matplotlib 中文字体，防止乱码"""
    # 尝试常见的中文字体名称
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    # 如果系统安装了 seaborn，也进行配置
    try:
        sns.set_theme(font='Noto Sans CJK SC')
    except:
        pass


setup_chinese_font()


# --- 3. LangGraph 定义 ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def chatbot(state: AgentState):
    """调用 LLM 的节点"""
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    system_msg = SystemMessage(
        content="你是一个桌面文件管理助手。你可以统计、列出并读取桌面上的 .txt 文件。如果你分析了数据并想展示图表，请在回复中包含一个 JSON 格式的统计数据，例如：{\"文件A\": 10, \"文件B\": 20}。")

    messages = state["messages"]
    # 确保系统消息在最前面
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [system_msg] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def create_mcp_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("chatbot", chatbot)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("chatbot")
    workflow.add_conditional_edges("chatbot", tools_condition)
    workflow.add_edge("tools", "chatbot")
    return workflow.compile()


# --- 4. Streamlit UI 逻辑 ---
st.set_page_config(page_title="LangGraph MCP 助手", layout="wide")

st.title("🤖 LangGraph + MCP 智能文件助手")
st.markdown("重构版本：**LangGraph + Direct Tool Import + Streamlit**")

# 初始化 Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 侧边栏
with st.sidebar:
    st.header("功能说明")
    st.info("该助手直接导入本地工具并使用 LangGraph 管理对话流。支持自动解析 JSON 数据并生成可视化图表。")

    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 1. 统一渲染历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. 处理用户输入
prompt = st.chat_input("桌面上有什么 txt 文件？")

if prompt:
    # 立即将用户消息存入状态并显示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 开启 Assistant 响应区域（临时显示）
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()
        plot_placeholder = st.container()

        # 运行 LangGraph
        app = create_mcp_graph()
        inputs = {"messages": [HumanMessage(content=prompt)]}

        full_response = ""
        # 使用同步 stream 模拟异步体验，或者直接 invoke
        with status_placeholder.status("正在处理请求...") as status:
            for output in app.stream(inputs):
                for key, value in output.items():
                    if key == "chatbot":
                        last_msg = value["messages"][-1]
                        if last_msg.content:
                            full_response = last_msg.content
                            response_placeholder.markdown(full_response)
                    elif key == "tools":
                        status.update(label="正在执行本地工具调用...")
            status.update(label="处理完成", state="complete")

        # 尝试解析统计数据并绘图
        if "{" in full_response and "}" in full_response:
            try:
                json_match = re.search(r'\{.*\}', full_response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    # 检查是否为适合绘图的字典格式
                    if isinstance(data, dict) and any(isinstance(v, (int, float)) for v in data.values()):
                        with plot_placeholder:
                            st.subheader("📊 数据可视化分析")
                            df = pd.DataFrame(list(data.items()), columns=['指标', '数值'])
                            df = df[pd.to_numeric(df['数值'], errors='coerce').notnull()]

                            fig, ax = plt.subplots(figsize=(10, 5))
                            sns.barplot(x='指标', y='数值', data=df, ax=ax, palette='viridis')
                            plt.xticks(rotation=45)
                            plt.title("分析结果统计图")
                            st.pyplot(fig)
            except Exception as e:
                logging.error(f"绘图失败: {e}")

        # 3. 处理完成后，将结果存入 session_state
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # 4. 关键修复：强制触发 Rerun，确保 UI 状态同步，防止消息重复显示
        st.rerun()
