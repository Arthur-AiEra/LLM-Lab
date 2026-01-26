#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 自动集成（LangChain 模式）LangFuse - CallbackHandler
"""
混合智能体（Hybrid Agent）- 财富管理投顾AI助手（LangGraph + Streamlit + Langfuse 版本）

基于 LangGraph 和 OpenAI 实现的混合型智能体，集成 Langfuse 监测，
通过 Streamlit 提供 Web 交互界面，并支持中文显示。
包含"插件"和"推荐对话"功能。
"""

import json
import logging
import operator
import os
from typing import Dict, Any, List, TypedDict, Annotated

import matplotlib.pyplot as plt
import streamlit as st
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langgraph.graph import StateGraph, END

# 简单配置：设置根记录器的级别为 DEBUG
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. 环境与配置 ---

# 设置API密钥
if "OPENAI_API_KEY" not in os.environ:
    st.error("请设置 OPENAI_API_KEY 环境变量")
    st.stop()

# Langfuse 配置
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

# 初始化 Langfuse Callback Handler
langfuse_handler = None
if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    try:
        # 首先初始化 Langfuse 客户端
        Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST
        )
        # 然后初始化针对 LangChain 的 CallbackHandler
        langfuse_handler = CallbackHandler(public_key=LANGFUSE_PUBLIC_KEY)
        st.sidebar.success("Langfuse 监测已启用")
    except Exception as e:
        st.sidebar.error(f"Langfuse 初始化失败: {e}")
else:
    st.sidebar.warning("Langfuse 密钥未配置，监测已禁用")

# 设置matplotlib支持中文（为后续可能的图表扩展做准备）
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 模拟数据与工具定义 ---

SAMPLE_CUSTOMER_PROFILES = {
    "customer1": {
        "customer_id": "C10012345",
        "risk_tolerance": "平衡型",
        "investment_horizon": "中期",
        "financial_goals": ["退休规划", "子女教育金"],
        "investment_preferences": ["ESG投资", "科技行业"],
        "portfolio_value": 1500000.0,
        "current_allocations": {
            "股票": 0.40,
            "债券": 0.30,
            "现金": 0.10,
            "另类投资": 0.20
        }
    },
    "customer2": {
        "customer_id": "C10067890",
        "risk_tolerance": "进取型",
        "investment_horizon": "长期",
        "financial_goals": ["财富增长", "资产配置多元化"],
        "investment_preferences": ["新兴市场", "高成长行业"],
        "portfolio_value": 3000000.0,
        "current_allocations": {
            "股票": 0.65,
            "债券": 0.15,
            "现金": 0.05,
            "另类投资": 0.15
        }
    }
}

# 推荐对话列表
RECOMMENDED_CONVERSATIONS = [
    "今天上证指数的表现如何？",
    "我的投资组合中科技股占比是多少？",
    "根据当前市场情况，我应该如何调整投资组合以应对可能的经济衰退？",
    "考虑到我的退休目标，请评估我当前的投资策略并提供优化建议。",
    "我想为子女准备教育金，请帮我设计一个10年期的投资计划。",
]

# 可用插件列表
AVAILABLE_PLUGINS = {
    "query_shanghai_index": {
        "name": "上证指数查询",
        "description": "查询上证指数的最新行情数据",
        "enabled": True
    }
}


@tool
def query_shanghai_index() -> str:
    """查询上证指数的最新行情数据，包括当前点位、涨跌和涨跌幅。"""
    name = "上证指数"
    price = "3125.62"
    change = "6.32"
    pct = "0.20"
    return f'{name} 当前点位: {price}，涨跌: {change}，涨跌幅: {pct}%'


# --- 3. LangGraph 智能体状态与节点定义 ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    customer_id: str
    customer_profile: Dict[str, Any]


# 定义工具和模型
tools = [query_shanghai_index]
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)


def get_system_prompt(customer_profile: Dict[str, Any]) -> str:
    customer_info = json.dumps(customer_profile, ensure_ascii=False, indent=2)
    return f"""你是一个专业的财富管理投顾AI助手。

## 客户信息
{customer_info}

## 你的能力
- `query_shanghai_index`: 查询实时上证指数行情。

## 工作流程
1. 评估用户查询，判断是否需要调用工具。
2. 生成专业、详细且个性化的投顾建议。
"""


def call_model(state: AgentState):
    messages = state['messages']
    # 在调用时传入 Langfuse handler
    config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


def call_tool(state: AgentState):
    last_message = state['messages'][-1]
    tool_calls = last_message.tool_calls
    tool_messages = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_to_call = {t.name: t for t in tools}[tool_name]
        # 工具调用也可以加入 callback
        config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
        tool_output = tool_to_call.invoke(tool_call["args"], config=config)
        tool_messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
    return {"messages": tool_messages}


def should_continue(state: AgentState) -> str:
    last_message = state['messages'][-1]
    return "continue" if last_message.tool_calls else "end"


# --- 4. 构建 LangGraph ---

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
workflow.add_edge('action', 'agent')
app = workflow.compile()


# --- 5. 辅助函数：处理用户查询 ---

def process_user_query(prompt: str, customer_id: str, customer_profile: Dict[str, Any]):
    """处理用户查询并生成响应"""
    system_prompt = get_system_prompt(customer_profile)
    initial_messages = [
        HumanMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]

    inputs = {
        "messages": initial_messages,
        "customer_id": customer_id,
        "customer_profile": customer_profile
    }

    full_response = ""
    message_placeholder = st.empty()

    # 运行 Graph
    # 设置 Langfuse 配置：包括 Trace 名称 (run_name) 和 元数据 (metadata)
    config = {
        "callbacks": [langfuse_handler],
        "run_name": f"1-hybrid_wealth_advisor_langgraph_langfuse - {customer_id}",
        "metadata": {
            "langfuse_user_id": customer_id,
            "langfuse_customer_type": customer_profile.get("risk_tolerance", "未知")
        }
    } if langfuse_handler else {}

    for output in app.stream(inputs, config=config):
        if "agent" in output:
            agent_output = output['agent']['messages'][-1]
            if agent_output.content:
                full_response += agent_output.content
                message_placeholder.markdown(full_response + "▌")

    message_placeholder.markdown(full_response)
    return full_response


# --- 6. Streamlit Web 界面 ---

st.set_page_config(page_title="财富管理投顾AI助手", layout="wide")

st.title("混合智能体 - 财富管理投顾AI助手")
st.markdown("---")

# 初始化 session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'customer_id' not in st.session_state:
    st.session_state.customer_id = "customer1"
if 'enabled_plugins' not in st.session_state:
    st.session_state.enabled_plugins = {"query_shanghai_index": True}

# 侧边栏：客户选择、插件管理
with st.sidebar:
    st.header("⚙️ 配置")

    # 客户选择
    st.subheader("客户选择")
    customer_id = st.radio(
        "选择一个客户画像:",
        options=list(SAMPLE_CUSTOMER_PROFILES.keys()),
        format_func=lambda x: f'{x} ({SAMPLE_CUSTOMER_PROFILES[x]["risk_tolerance"]})',
        key='customer_id'
    )
    customer_profile = SAMPLE_CUSTOMER_PROFILES[customer_id]

    with st.expander("📋 客户详情"):
        st.json(customer_profile)

    # 插件管理
    st.subheader("🔌 插件")
    for plugin_id, plugin_info in AVAILABLE_PLUGINS.items():
        col1, col2 = st.columns([0.2, 0.8])
        with col1:
            enabled = st.checkbox(
                "启用",
                value=st.session_state.enabled_plugins.get(plugin_id, True),
                key=f"plugin_{plugin_id}",
                label_visibility="collapsed"
            )
            st.session_state.enabled_plugins[plugin_id] = enabled
        with col2:
            st.write(f"**{plugin_info['name']}**")
            st.caption(plugin_info['description'])

# 主聊天界面
st.subheader("💬 对话")

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 推荐对话部分（仅在没有消息时显示）
if not st.session_state.messages:
    st.subheader("📌 推荐对话")
    st.markdown("点击下方任一问题快速开始对话：")

    # 创建列布局以展示建议
    for suggestion in RECOMMENDED_CONVERSATIONS:
        if st.button(suggestion, key=f"suggestion_{suggestion}", use_container_width=True):
            # 添加用户消息到历史
            st.session_state.messages.append({"role": "user", "content": suggestion})
            # 显示用户消息
            with st.chat_message("user"):
                st.markdown(suggestion)

            # 处理查询并显示助手响应
            with st.chat_message("assistant"):
                with st.spinner("正在思考..."):
                    response = process_user_query(suggestion, customer_id, customer_profile)
                    st.session_state.messages.append({"role": "assistant", "content": response})

            # 重新运行应用以刷新界面
            st.rerun()

# 输入框
if prompt := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("正在思考..."):
            response = process_user_query(prompt, customer_id, customer_profile)
            st.session_state.messages.append({"role": "assistant", "content": response})
