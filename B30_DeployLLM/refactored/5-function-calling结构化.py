import os
from enum import Enum
from pydantic import BaseModel

import openai
from openai import OpenAI

#t 146:03 https://gemini.google.com/app/054ebed098f1b878

# 定义数据库表的枚举类型
class Table(str, Enum):
    orders = "orders"  # 订单表
    customers = "customers"  # 客户表
    products = "products"  # 产品表

# 定义数据库列的枚举类型
class Column(str, Enum):
    id = "id"  # 唯一标识符
    status = "status"  # 订单状态
    expected_delivery_date = "expected_delivery_date"  # 预期交货日期
    delivered_at = "delivered_at"  # 实际交货日期
    shipped_at = "shipped_at"  # 发货日期
    ordered_at = "ordered_at"  # 订单日期
    canceled_at = "canceled_at"  # 取消日期

# 定义操作符的枚举类型
class Operator(str, Enum):
    eq = "="  # 等于
    gt = ">"  # 大于
    lt = "<"  # 小于
    le = "<="  # 小于等于
    ge = ">="  # 大于等于
    ne = "!="  # 不等于

# 定义排序方式的枚举类型
class OrderBy(str, Enum):
    asc = "asc"  # 升序
    desc = "desc"  # 降序

# 定义条件的数据模型
class Condition(BaseModel):
    column: str  # 列
    operator: Operator  # 操作符
    value: str  # 条件值

# 定义查询的数据模型
class Query(BaseModel):
    table_name: Table  # 表名
    columns: list[Column]  # 查询的列
    conditions: list[Condition]  # 查询的条件
    order_by: OrderBy  # 排序方式

# 创建DashScope客户端(兼容OpenAI协议)
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    base_url = "https://api.fe8.cn/v1"
)

# 通过Function Calling进行结构化输出
completion = client.beta.chat.completions.parse(
    model="qwen-max",
    messages=[
        {
            "role": "system",
            "content": "你是一个有用的助手。当前日期是2026年3月22日。你帮助用户通过调用查询函数来查找他们需要的数据。",
        },
        {
            "role": "user",
            "content": "查找去年5月所有已完成但未按时交付的订单",
        },
    ],
    tools=[
        openai.pydantic_function_tool(Query),  # 使用Pydantic模型作为工具
    ],
)

# 打印解析后的查询参数
print(completion.choices[0].message.tool_calls[0].function.parsed_arguments)
