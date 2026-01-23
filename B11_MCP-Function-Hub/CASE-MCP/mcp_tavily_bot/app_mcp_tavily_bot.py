import asyncio
import logging
import os
import platform

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
import nest_asyncio
# 允许在现有事件循环中嵌套运行（解决 Streamlit 中的异步冲突）
nest_asyncio.apply()

import logging

# 简单配置：设置根记录器的级别为 DEBUG
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# --- 1. 环境配置与多平台中文字体修复 ---
def setup_chinese_font():
    """
    设置 Matplotlib 中文字体，兼容 Linux, macOS 和 Windows。
    解决 'findfont: Font family not found' 和中文乱码问题。
    """
    system = platform.system()
    
    # 定义不同平台下的常用中文字体优先级
    if system == "Darwin":  # macOS
        fonts = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'STHeiti', 'sans-serif']
    elif system == "Windows":
        fonts = ['SimHei', 'Microsoft YaHei', 'STSong', 'sans-serif']
    else:  # Linux / Docker
        fonts = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
    
    plt.rcParams['font.sans-serif'] = fonts
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
    
    # 尝试设置 Seaborn 主题字体
    try:
        sns.set_theme(font=fonts[0])
    except:
        sns.set_theme()

setup_chinese_font()

# --- 2. Streamlit UI 配置 ---
st.set_page_config(page_title="Tavily 搜索智能助手", page_icon="🔍", layout="wide")
st.title("🔍 Tavily 搜索智能助手")
st.markdown("当前架构：**OpenAI + langchain-mcp-adapters (create_agent) + Streamlit**")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_value" not in st.session_state:
    st.session_state.input_value = ""

# --- 3. 侧边栏配置 ---
with st.sidebar:
    st.header("⚙️ 服务配置")
    st.success("该助手采用 langchain MultiServerMCPClient 连接 Tavily MCP Server 并运行 Agent")
    
    st.divider()
    st.header("💡 建议查询")
    suggestions = [
        '查找黄金相关的新闻',
        '搜索最新的AI技术发展趋势',
        '查找2026年经济预测相关文章',
        '搜索Python编程最佳实践',
        '分析当前全球半导体市场'
    ]
    for suggestion in suggestions:
        if st.button(suggestion, use_container_width=True):
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
async def run_agent(user_input: str):
    """
    使用 MultiServerMCPClient 连接 Tavily MCP Server 并运行 Agent
    """
    server_config = {
        "tavily_search": {
            "command": "npx",
            "args": ["-y", "tavily-mcp@0.1.4"],
            "transport": "stdio",
            "env": {**os.environ, "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", "")}
        }
    }
    
    client = MultiServerMCPClient(server_config)

    try:
        # 1. 异步获取工具
        tools = await client.get_tools()

        # 2. 创建 LLM
        llm = ChatOpenAI(model="gpt-4o", temperature=0)

        # 3. 使用 create_agent 创建 Agent
        agent = create_agent(llm, tools)

        # 4. 准备输入消息
        input_messages = st.session_state.messages

        # 5. 运行 Agent
        result = await agent.ainvoke({"messages": input_messages})

        # 6. 提取最终回复
        final_message = result["messages"][-1]
        return final_message
    except Exception as e:
        logging.error(f"run_agent error: {str(e)}")
        return f"Agent 执行出错: {str(e)}"

def extract_and_visualize(response_text: str):
    """尝试从响应中提取数据并可视化，已修复 Seaborn 警告"""
    if any(kw in response_text for kw in ["趋势", "分析", "图表", "数据"]):
        st.subheader("📊 数据可视化分析")
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # 模拟数据
        data = {"维度A": 80, "维度B": 95, "维度C": 70}
        df = pd.DataFrame(list(data.items()), columns=['指标', '数值'])
        
        # 修复 Seaborn 警告：指定 hue 并设置 legend=False
        sns.barplot(
            x='指标', 
            y='数值', 
            data=df, 
            ax=ax, 
            palette='viridis', 
            hue='指标', 
            legend=False
        )
        
        plt.title("搜索结果多维度分析 (多平台中文支持)")
        st.pyplot(fig)

# --- 6. 处理用户输入 ---
prompt = st.chat_input("请输入您的问题...")

if st.session_state.input_value:
    prompt = st.session_state.input_value
    st.session_state.input_value = ""

if prompt:
    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()

        try:
            status_placeholder.status("🔄 正在连接 Tavily MCP 并执行搜索任务...")

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                ai_response = loop.run_until_complete(run_agent(prompt))
            finally:
                loop.close()

            if isinstance(ai_response, AIMessage):
                response_content = ai_response.content
                response_placeholder.markdown(response_content)
                st.session_state.messages.append(ai_response)
                extract_and_visualize(response_content)
            else:
                st.error(f"❌ 执行失败: {ai_response}")

        except Exception as e:
            st.error(f"❌ 调用失败: {str(e)}")
        finally:
            status_placeholder.empty()

    st.rerun()
