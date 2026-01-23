import os
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
import streamlit as st
from openai import OpenAI

# ==========================================
# 1. 解决中文显示问题
# ==========================================
# 优先使用系统中存在的 Noto Sans CJK SC 或 WenQuanYi Micro Hei
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 配置与常量
# ==========================================
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
我将回答用户关于门票相关的问题。

当用户的问题需要查询数据时，请调用 exc_sql 工具。
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "exc_sql",
            "description": "执行SQL查询并返回结果数据",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_input": {
                        "type": "string",
                        "description": "生成的SQL语句",
                    },
                    "database": {
                        "type": "string",
                        "description": "数据库名称，默认为 ubr",
                        "default": "ubr"
                    }
                },
                "required": ["sql_input"],
            },
        }
    }
]

def normalize_messages(msgs):
    """
    只保留 OpenAI 需要的字段，并确保每条消息都有 content 键（允许为空字符串）。
    同时保留 tool_calls / tool_call_id / name 以支持工具闭环。
    """
    clean = []
    for m in msgs:
        # 兼容：如果未来又混入 SDK 对象
        if hasattr(m, "model_dump"):
            m = m.model_dump(exclude_none=True)

        item = {"role": m.get("role", "")}

        # 确保 content 永远存在（tool_calls 场景允许为空）
        item["content"] = m.get("content", "") if m.get("content", "") is not None else ""

        # 这些字段在 tool calling 场景会用到
        if "tool_calls" in m:
            item["tool_calls"] = m["tool_calls"]
        if "tool_call_id" in m:
            item["tool_call_id"] = m["tool_call_id"]
        if "name" in m:
            item["name"] = m["name"]

        clean.append(item)
    return clean

# ==========================================
# 3. 工具函数实现
# ==========================================
def execute_sql(sql_input, database='ubr'):
    """执行SQL并返回DataFrame"""
    try:
        engine = create_engine(
            f'mysql+pymysql://student123:student321@localhost:3307/{database}?charset=utf8mb4',
            connect_args={'connect_timeout': 10}, pool_size=10, max_overflow=20
        )
        df = pd.read_sql(text(sql_input), engine)
        return df
    except Exception as e:
        return str(e)


def generate_chart(df):
    """生成图表并返回fig对象"""
    columns = df.columns
    if len(columns) < 2:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    # 获取object类型
    object_columns = df.select_dtypes(include='O').columns.tolist()
    if columns[0] in object_columns:
        object_columns.remove(columns[0])
    num_columns = df.select_dtypes(exclude='O').columns.tolist()

    if len(object_columns) > 0:
        # 透视表逻辑
        pivot_df = df.pivot_table(index=columns[0], columns=object_columns,
                                  values=num_columns,
                                  fill_value=0)
        bottoms = None
        for col in pivot_df.columns:
            label_str = str(col)
            ax.bar(pivot_df.index.astype(str), pivot_df[col], bottom=bottoms, label=label_str)
            if bottoms is None:
                bottoms = pivot_df[col].copy()
            else:
                bottoms += pivot_df[col]
    else:
        bottom = np.zeros(len(df))
        x = np.arange(len(df))
        for column in columns[1:]:
            ax.bar(df[columns[0]].astype(str), df[column], bottom=bottom, label=str(column))
            bottom += df[column]

    ax.legend()
    ax.set_title("销售统计")
    ax.set_xlabel(str(columns[0]))
    ax.set_ylabel("数值")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


# ==========================================
# 4. Streamlit 界面
# ==========================================
def main():
    st.set_page_config(page_title="门票助手", layout="wide")
    st.title("🎫 门票助手 (OpenAI + Streamlit)")

    # 初始化 OpenAI 客户端
    client = OpenAI()  # 环境变量已配置

    # 初始化会话状态
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 显示聊天历史（排除 system 消息）
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    st.markdown(content)
                elif "df" in msg:
                    st.dataframe(msg["df"])
                    if "fig" in msg:
                        st.pyplot(msg["fig"])

    # 用户输入
    if prompt := st.chat_input("请输入您的问题，例如：2023年7月的不同省份的入园人数统计"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            # 调用 OpenAI API
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=normalize_messages(st.session_state.messages),
                tools=TOOLS,
                tool_choice="auto"
            )

            response_message = response.choices[0].message

            # 处理 Tool Calls
            if response_message.tool_calls:
                st.session_state.messages.append(response_message.model_dump(exclude_none=True))

                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    if function_name == "exc_sql":
                        st.info(f"正在执行 SQL: {function_args.get('sql_input')}")
                        df = execute_sql(
                            sql_input=function_args.get("sql_input"),
                            database=function_args.get("database", "ubr")
                        )

                        if isinstance(df, pd.DataFrame):
                            # 将结果反馈给模型
                            st.session_state.messages.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": df.head(20).to_json(orient='records')
                            })

                            # 再次调用模型总结结果
                            second_response = client.chat.completions.create(
                                model="gpt-4.1-mini",
                                messages=normalize_messages(st.session_state.messages)
                            )
                            final_text = second_response.choices[0].message.content
                            st.markdown(final_text)

                            # 显示表格和图表
                            st.dataframe(df)
                            fig = generate_chart(df)
                            if fig:
                                st.pyplot(fig)

                            # 保存到会话状态
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": final_text,
                                "df": df,
                                "fig": fig
                            })
                        else:
                            st.error(f"SQL 执行出错: {df}")
                            st.session_state.messages.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": f"Error: {df}"
                            })
            else:
                full_response = response_message.content
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
