import os
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from openai import OpenAI
import streamlit as st
import matplotlib.font_manager as fm


# ====== Matplotlib 中文显示配置 ======
def set_matplot_zh_font():
    """设置 Matplotlib 的中文字体，解决乱码问题"""
    # 常见的中文字体名称列表
    zh_fonts = [
        'SimHei',  # Windows/Linux
        'Arial Unicode MS',  # macOS
        'PingFang SC',  # macOS
        'Heiti TC',  # macOS
        'STHeiti',  # macOS
        'Microsoft YaHei',  # Windows
        'WenQuanYi Micro Hei',  # Linux
        'Droid Sans Fallback'  # Linux
    ]

    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font in zh_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            return font

    # 如果没找到，尝试搜索包含 'CJK' 或 'Chinese' 的字体
    for f in fm.fontManager.ttflist:
        if 'CJK' in f.name or 'Chinese' in f.name:
            plt.rcParams['font.sans-serif'] = [f.name] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            return f.name
    return None


# 初始化字体设置
current_font = set_matplot_zh_font()

# ====== 配置与初始化 ======
# 优先从环境变量获取，如果没有则需要用户在界面输入
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# 数据库配置
# 数据库配置
# 注意：确保已安装 mysql-connector-python
DB_URL = "mysql+mysqlconnector://student123:student321@localhost:3307/ubr?charset=utf8mb4"

# 页面配置
st.set_page_config(page_title="门票助手", page_icon="🎫", layout="wide")

# ====== System Prompt ======
SYSTEM_PROMPT = """我是门票助手，以下是关于门票订单表相关的字段，我可能会编写对应的SQL，对数据进行查询
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

每当 exc_sql 工具返回数据时，我会结合表格和图片信息回答用户。
"""

# ====== 工具定义 (OpenAI Tool Spec) ======
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "exc_sql",
            "description": "执行生成的SQL语句进行查询，并自动生成可视化图表",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_input": {
                        "type": "string",
                        "description": "生成的SQL查询语句",
                    }
                },
                "required": ["sql_input"],
            },
        }
    }
]


# ====== 工具实现 ======
def exc_sql(sql_input):
    """执行 SQL 并返回结果和图表路径"""
    engine = create_engine(DB_URL, connect_args={'connect_timeout': 10}, pool_size=10, max_overflow=20)
    try:
        df = pd.read_sql(sql_input, engine)
        if df.empty:
            return {"text": "查询结果为空", "df": df, "image": None}

        # 自动推断x/y字段进行绘图
        x_candidates = df.select_dtypes(include=['object']).columns.tolist()
        if not x_candidates:
            x_candidates = df.columns.tolist()
        x = x_candidates[0]

        y_fields = df.select_dtypes(include=['number']).columns.tolist()

        if y_fields:
            plt.figure(figsize=(10, 6))
            bar_width = 0.35 if len(y_fields) > 1 else 0.6
            x_labels = df[x].astype(str)
            x_pos = range(len(df))

            for idx, y_col in enumerate(y_fields):
                plt.bar([p + idx * bar_width for p in x_pos], df[y_col], width=bar_width, label=y_col)

            plt.xlabel(x)
            plt.ylabel(','.join(y_fields))
            plt.title(f"{' & '.join(y_fields)} by {x}")
            plt.xticks([p + bar_width * (len(y_fields) - 1) / 2 for p in x_pos], x_labels, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()

            # 保存图片
            save_dir = "image_show"
            os.makedirs(save_dir, exist_ok=True)
            filename = f'bar_{int(time.time() * 1000)}.png'
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path)
            plt.close()
            return {"text": df.head(10).to_markdown(index=False), "df": df, "image": save_path}
        else:
            return {"text": df.head(10).to_markdown(index=False), "df": df, "image": None}

    except Exception as e:
        return {"text": f"SQL执行出错: {str(e)}", "df": None, "image": None}


# ====== Streamlit UI ======
st.title("🎫 门票助手 (OpenAI + Streamlit)")

# 侧边栏配置
with st.sidebar:
    st.header("配置")
    api_key = st.text_input("OpenAI API Key", value=OPENAI_API_KEY, type="password")
    base_url = st.text_input("OpenAI Base URL", value=OPENAI_BASE_URL)
    model = st.selectbox("选择模型", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"], index=0)

    if st.button("清空对话历史"):
        st.session_state.messages = []
        st.rerun()

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "results" in message:
            for res in message["results"]:
                if res.get("df") is not None:
                    st.dataframe(res["df"])
                if res.get("image"):
                    st.image(res["image"])

# 用户输入
if prompt := st.chat_input("请输入您的问题，例如：2023年7月的不同省份的入园人数统计"):
    if not api_key:
        st.error("请先在侧边栏配置 API Key")
        st.stop()

    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用 OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # 准备发送给 API 的消息
        api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in st.session_state.messages:
            api_messages.append({"role": m["role"], "content": m["content"]})

        try:
            # 第一次调用：判断是否需要调用工具
            response = client.chat.completions.create(
                model=model,
                messages=api_messages,
                tools=TOOLS,
                tool_choice="auto",
            )

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            results_to_store = []

            if tool_calls:
                # 如果有工具调用
                api_messages.append(response_message)

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    if function_name == "exc_sql":
                        st.info(f"正在执行 SQL: {function_args.get('sql_input')}")
                        result = exc_sql(function_args.get('sql_input'))

                        # 将结果反馈给模型
                        api_messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": result["text"],
                        })

                        # 存储结果用于 UI 显示
                        results_to_store.append(result)

                # 第二次调用：获取最终回答
                second_response = client.chat.completions.create(
                    model=model,
                    messages=api_messages,
                )
                full_response = second_response.choices[0].message.content
            else:
                # 没有工具调用，直接返回回答
                full_response = response_message.content

            # 显示最终回答和结果
            message_placeholder.markdown(full_response)
            for res in results_to_store:
                if res.get("df") is not None:
                    st.dataframe(res["df"])
                if res.get("image"):
                    st.image(res["image"])

            # 保存助手消息
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "results": results_to_store
            })

        except Exception as e:
            st.error(f"发生错误: {str(e)}")
