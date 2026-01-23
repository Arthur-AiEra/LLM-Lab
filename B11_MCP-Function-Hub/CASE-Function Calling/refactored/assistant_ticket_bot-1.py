"""
门票助手 - Streamlit Web 界面版本
使用 OpenAI SDK 和 Streamlit 构建交互式 Web 应用
"""

import os
import json
import pandas as pd
import streamlit as st
from openai import OpenAI, APIError
from sqlalchemy import create_engine

import logging

# 简单配置：设置根记录器的级别为 DEBUG
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

logging.debug('这是一条 Debug 消息，只有在级别设置为 DEBUG 时才会显示。')
logging.info('这是一条 Info 消息。')

# ====== 页面配置 ======
st.set_page_config(
    page_title="门票助手",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== 系统提示词 ======
SYSTEM_PROMPT = """我是门票助手，以下是关于门票订单表相关的字段，我可能会编写对应的SQL(运行环境是MySQL8.0)，对数据进行查询
-- 门票订单表
CREATE TABLE tkt_orders (
    order_time DATETIME,             -- 订单日期
    account_id INT,                  -- 预定用户ID
    gov_id VARCHAR(18),              -- 商品使用人ID（身份证号）
    gender VARCHAR(10),              -- 使用人性别
    age INT,                         -- 年龄
    province VARCHAR(30),           -- 使用人省份
    SKU VARCHAR(100),                -- 商品SKU名
    product_serial_no VARCHAR(30),  -- 商品ID
    eco_main_order_id VARCHAR(20),  -- 订单ID
    sales_channel VARCHAR(20),      -- 销售渠道
    status VARCHAR(30),             -- 商品状态
    order_value DECIMAL(10,2),       -- 订单金额
    quantity INT                     -- 商品数量
);
一日门票，对应多种SKU：
Universal Studios Beijing One-Day Dated Ticket-Standard
Universal Studios Beijing One-Day Dated Ticket-Child
Universal Studios Beijing One-Day Dated Ticket-Senior
二日门票，对应多种SKU：
USB 1.5-Day Dated Ticket Standard
USB 1.5-Day Dated Ticket Discounted
一日门票、二日门票查询
SUM(CASE WHEN SKU LIKE 'Universal Studios Beijing One-Day%' THEN quantity ELSE 0 END) AS one_day_ticket_sales,
SUM(CASE WHEN SKU LIKE 'USB%' THEN quantity ELSE 0 END) AS two_day_ticket_sales
我将回答用户关于门票相关的问题
"""

# ====== OpenAI 工具定义 ======
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": "执行SQL查询语句，用于查询门票订单数据",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_query": {
                        "type": "string",
                        "description": "要执行的SQL查询语句"
                    },
                    "database": {
                        "type": "string",
                        "description": "数据库名称，默认为 'ubr'",
                        "default": "ubr"
                    }
                },
                "required": ["sql_query"]
            }
        }
    }
]

# ====== 数据库连接配置 ======
DB_CONFIG = {
    'host': 'localhost',
    'port': 3307,
    'user': 'student123',
    'password': 'student321',
    'charset': 'utf8mb4'
}

# ====== SQL 执行工具 ======
def execute_sql(sql_query: str, database: str = 'ubr') -> str:
    """执行 SQL 查询并返回结果"""
    try:
        connection_string = (
            f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
            f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{database}"
            f"?charset={DB_CONFIG['charset']}"
        )

        engine = create_engine(
            connection_string,
            connect_args={'connect_timeout': 10},
            pool_size=10,
            max_overflow=20
        )

        df = pd.read_sql(sql_query, engine)
        result = df.head(10).to_markdown(index=False)
        return result if result else "查询结果为空"

    except Exception as e:
        logging.error(f"SQL 执行出错: {str(e)}")
        return f"SQL 执行出错: {str(e)}"

# ====== 工具调用处理 ======
def process_tool_call(tool_name: str, tool_input: dict) -> str:
    """处理 OpenAI 返回的工具调用"""
    if tool_name == "execute_sql":
        return execute_sql(
            sql_query=tool_input.get('sql_query'),
            database=tool_input.get('database', 'ubr')
        )
    else:
        logging.error(f"未知的工具: {tool_name}")
        return f"未知的工具: {tool_name}"

# ====== 工具调用处理 ======
def process_query(input: str, model: str, temperature: float, max_tokens: int) :
    """处理 用户 query"""
    logging.info(f"process_query, input: {input}, model: {model}, temperature: {temperature}, max_tokens: {max_tokens}")
    if input:
        # 添加用户消息
        st.session_state.messages.append({
            "role": "user",
            "content": input
        })

        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(input)

        # 处理 OpenAI 响应
        with st.chat_message("assistant"):
            with st.spinner("正在处理您的请求..."):
                try:
                    # 构建消息列表
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        *st.session_state.messages
                    ]

                    # 调用 OpenAI API
                    response = st.session_state.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=TOOLS,
                        tool_choice="auto",
                        temperature=temperature,
                        max_tokens=max_tokens
                    )

                    assistant_message = response.choices[0].message

                    # 处理工具调用
                    if assistant_message.tool_calls:
                        # 执行工具调用
                        tool_results = []
                        for tool_call in assistant_message.tool_calls:
                            tool_name = tool_call.function.name
                            tool_input = json.loads(tool_call.function.arguments)
                            tool_result = process_tool_call(tool_name, tool_input)
                            tool_results.append(tool_result)

                        # 再次调用 API 以获得最终回复
                        messages.append({
                            "role": "assistant",
                            "content": assistant_message.content or "",
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments
                                    }
                                }
                                for tc in assistant_message.tool_calls
                            ]
                        })

                        for tool_result in tool_results:
                            messages.append({
                                "role": "user",
                                "content": tool_result
                            })

                        final_response = st.session_state.client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )

                        final_message = final_response.choices[0].message.content
                        st.markdown(final_message)

                        # 添加到消息历史
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": final_message
                        })
                    else:
                        # 直接返回文本回复
                        reply = assistant_message.content or "无法生成回复"
                        st.markdown(reply)

                        # 添加到消息历史
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": reply
                        })

                except APIError as e:
                    logging.error(f"❌ API 错误: {str(e)}")
                    st.error(f"❌ API 错误: {str(e)}")
                except Exception as e:
                    logging.error(f"❌ 发生错误: {str(e)}")
                    st.error(f"❌ 发生错误: {str(e)}")

# ====== 初始化 Streamlit 会话状态 ======
if "messages" not in st.session_state:
    st.session_state.messages = []

if "client" not in st.session_state:
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.error("❌ OPENAI_API_KEY 环境变量未设置")
        st.stop()
    st.session_state.client = OpenAI(api_key=api_key, base_url="https://api.fe8.cn/v1")

# ====== 页面标题和说明 ======
st.markdown("""
# 🎫 门票助手 (OpenAI 版本)

欢迎使用门票助手！我可以帮您查询门票订单信息、统计销售数据等。

**功能特性：**
- 🔍 智能查询门票订单数据
- 📊 生成销售统计报告
- 💬 自然语言交互
- 🔄 支持多轮对话
""")

print(f"reload page")

# ====== 侧边栏配置 ======
with st.sidebar:
    logging.info(f"reload sidebar")
    st.markdown("### ⚙️ 设置")

    model = st.selectbox(
        "选择模型",
        ["gpt-4.1-mini", "gpt-4.1-nano", "gemini-2.5-flash"],
        index=0
    )

    temperature = st.slider(
        "温度 (Creativity)",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1
    )

    max_tokens = st.slider(
        "最大令牌数",
        min_value=256,
        max_value=4096,
        value=2048,
        step=256
    )

    st.markdown("---")

    st.markdown("### 💡 建议问题")
    suggestions = [
        "2023年4、5、6月一日门票，二日门票的销量多少？帮我按照周进行统计",
        "2023年7月的不同省份的入园人数统计",
        "帮我查看2023年10月1-7日销售渠道订单金额排名",
    ]

    for i, suggestion in enumerate(suggestions):
        if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
            st.session_state.suggestion_clicked = suggestion

    st.markdown("---")

    if st.button("🗑️ 清空对话历史", use_container_width=True):
        st.session_state.messages = []
        # st.rerun()

# ====== 主聊天区域 ======
st.markdown("### 💬 对话")
logging.info(f"reload 主聊天区域")

# 显示对话历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
user_input = st.chat_input("请输入您的问题...")

if user_input:
    process_query(user_input, model, temperature, max_tokens)
elif "suggestion_clicked" in st.session_state:
    suggestion = st.session_state.pop("suggestion_clicked")
    process_query(suggestion, model, temperature, max_tokens)

# ====== 页脚 ======
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px;">
    门票助手 v1.0 | 由 OpenAI API 驱动 | 
    <a href="https://openai.com" target="_blank">OpenAI</a>
</div>
""", unsafe_allow_html=True)
