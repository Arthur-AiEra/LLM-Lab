import asyncio
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# --- 1. 环境配置与日志设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def setup_chinese_font():
    """设置 Matplotlib 中文字体，确保 macOS 环境下中文正常显示"""
    # Arial Unicode MS 是 macOS 自带的支持中文的字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'SimSun', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(font='Arial Unicode MS')


setup_chinese_font()


# --- 2. 业务逻辑工具定义 (模拟原代码中的五个阶段) ---
# 这里我们将原代码中的逻辑封装为 LangChain 工具，以便 Agent 调用

@tool
def market_perception(research_topic: str, industry_focus: str, time_horizon: str) -> str:
    """感知阶段：收集市场数据和信息，包括市场概况、关键指标、重要新闻、行业趋势等"""
    perception_data = {
        "market_overview": f"基于{research_topic}的市场概况分析，重点关注{industry_focus}领域在{time_horizon}时间框架下的发展态势。",
        "key_indicators": {
            "GDP增长率": "5.2%",
            "CPI指数": "2.1%",
            "PMI指数": "51.2",
            "市场情绪指数": "中性偏乐观"
        },
        "recent_news": [
            f"政策支持{industry_focus}发展",
            f"{industry_focus}技术突破",
            f"国际竞争加剧"
        ]
    }
    return json.dumps(perception_data, ensure_ascii=False)


@tool
def market_modeling(perception_data_json: str) -> str:
    """建模阶段：构建市场内部模型，分析市场状态、经济周期、风险因素等"""
    data = json.loads(perception_data_json)
    world_model = {
        "market_state": "成长期向成熟期过渡阶段",
        "economic_cycle": "复苏期",
        "risk_factors": ["技术路线风险", "政策不确定性"],
        "opportunity_areas": ["成本下降", "需求增长"]
    }
    return json.dumps(world_model, ensure_ascii=False)


@tool
def investment_reasoning(world_model_json: str) -> str:
    """推理阶段：生成多个候选投资分析方案，评估不同策略的可行性"""
    reasoning_plans = [
        {"plan_id": "growth", "hypothesis": "重点投资成长性企业", "confidence": 0.75},
        {"plan_id": "value", "hypothesis": "关注被低估的优质企业", "confidence": 0.65}
    ]
    return json.dumps(reasoning_plans, ensure_ascii=False)


@tool
def investment_decision(reasoning_plans_json: str) -> str:
    """决策阶段：评估候选方案，选择最优投资观点"""
    selected_plan = {
        "selected_id": "growth",
        "thesis": "建议采用成长型投资策略",
        "recommendation": "配置60-80%资金于成长型标的"
    }
    return json.dumps(selected_plan, ensure_ascii=False)


@tool
def generate_report(research_topic: str, selected_plan_json: str) -> str:
    """报告阶段：生成完整的投资研究报告"""
    plan = json.loads(selected_plan_json)
    report = f"""# {research_topic} - 投资研究报告

## 核心观点
{plan['thesis']}

## 投资建议
{plan['recommendation']}

---
*报告由智能投研助手生成*
"""
    return report


# 组合本地工具
LOCAL_TOOLS = [market_perception, market_modeling, investment_reasoning, investment_decision, generate_report]

# --- 3. Streamlit UI 配置 ---
st.set_page_config(page_title="智能投研助手 (LangChain+MCP)", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. 侧边栏配置 ---
with st.sidebar:
    st.title("📈 投研助手配置")
    st.info("基于 OpenAI + LangChain + MCP 架构重构")

    st.header("🔧 MCP 服务配置 (可选)")
    use_mcp = st.checkbox("启用外部 MCP 工具", value=False)
    if use_mcp:
        transport_type = st.selectbox("传输协议", ["stdio", "sse"])
        if transport_type == "stdio":
            stdio_cmd = st.text_input("命令", value="python3")
            stdio_args = st.text_input("参数 (逗号分隔)", value="server.py")
        else:
            sse_url = st.text_input("SSE URL", value="http://127.0.0.1:8001/sse")

    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()


# --- 5. 核心 Agent 运行逻辑 ---
async def run_agent(user_input: str, mcp_config: Optional[dict] = None):
    # 1. 初始化 LLM
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    # 2. 获取工具
    tools = list(LOCAL_TOOLS)
    if mcp_config:
        try:
            mcp_client = MultiServerMCPClient(mcp_config)
            mcp_tools = await mcp_client.get_tools()
            tools.extend(mcp_tools)
        except Exception as e:
            logging.error(f"MCP Client Error: {e}")
            st.warning(f"无法连接到 MCP 服务: {e}")

    # 3. 创建 Agent
    # 使用 LangGraph 的 create_agent (ReAct)
    agent = create_agent(llm, tools)

    # 4. 运行
    input_messages = [
        SystemMessage(content="你是一个专业的投资研究分析师。请按照感知、建模、推理、决策、报告的流程进行分析。"),
        *st.session_state.messages
    ]

    result = await agent.ainvoke({"messages": input_messages})
    return result["messages"][-1]


# --- 6. UI 交互逻辑 ---
st.title("📈 智能投研助手 (Deliberative Agent)")

# 显示历史消息
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# 处理输入
if prompt := st.chat_input("例如：请分析新能源汽车行业在中期时间框架下的投资机会"):
    # 1. 记录并显示用户消息
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 运行 Agent 并显示回复
    with st.chat_message("assistant"):
        status_placeholder = st.status("🔄 正在进行深思熟虑分析...", expanded=True)

        mcp_config = None
        if use_mcp:
            if transport_type == "stdio":
                mcp_config = {
                    "external_server": {"command": stdio_cmd, "args": [a.strip() for a in stdio_args.split(",")],
                                        "transport": "stdio"}}
            else:
                mcp_config = {"external_server": {"url": sse_url, "transport": "sse"}}

        try:
            # 运行异步 Agent
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            ai_response = loop.run_until_complete(run_agent(prompt, mcp_config))

            if isinstance(ai_response, AIMessage):
                st.markdown(ai_response.content)
                st.session_state.messages.append(ai_response)

                # 尝试可视化 (如果返回了 JSON 数据)
                if "{" in ai_response.content:
                    try:
                        import re

                        json_match = re.search(r'\{.*\}', ai_response.content, re.DOTALL)
                        if json_match:
                            data = json.loads(json_match.group())
                            if isinstance(data, dict) and any(isinstance(v, (int, float)) for v in data.values()):
                                st.subheader("📊 数据分析图表")
                                df = pd.DataFrame(list(data.items()), columns=['指标', '数值'])
                                fig, ax = plt.subplots(figsize=(10, 5))
                                sns.barplot(x='指标', y='数值', data=df, ax=ax)
                                plt.xticks(rotation=45)
                                st.pyplot(fig)
                    except:
                        pass
            else:
                st.error("Agent 返回了非预期类型的响应。")

        except Exception as e:
            st.error(f"执行出错: {e}")
            logging.exception("Agent Execution Error")
        finally:
            status_placeholder.update(label="分析完成！", state="complete", expanded=False)

    # 强制刷新以保持 UI 同步
    st.rerun()
