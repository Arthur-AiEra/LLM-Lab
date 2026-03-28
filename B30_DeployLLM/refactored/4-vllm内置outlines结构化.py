#!/usr/bin/env python
# coding: utf-8

# vLLM 内置了结构化输出引擎(xgrammar/outlines)，可以在服务端进行结构化生成
# vLLM v0.12+ 废弃了 guided_json 等旧参数，改用 structured_outputs 或 response_format
# Qwen3 需要关闭思考模式，否则输出会包含 <think> 块

import json
from enum import Enum
from pydantic import BaseModel
from openai import OpenAI

# 连接本地 vLLM 服务
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

MODEL_NAME = "/root/autodl-tmp/models/qwen/Qwen3-0.6B"

# Qwen3 关闭思考模式的公共参数
EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}

# ============================================================
# 示例1: JSON结构化生成 - 角色信息
# 使用 response_format (json_schema) 约束输出
# ============================================================

class Name(str, Enum):
    john = "John"
    paul = "Paul"
    lisa = "Lisa"

class Character(BaseModel):
    name: Name
    age: int
    occupation: str

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "user", "content": "Generate a character named Paul who is 30 years old and works as an engineer."}
    ],
    max_tokens=256,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "character",
            "schema": Character.model_json_schema(),
        },
    },
    extra_body=EXTRA_BODY,
)

raw_output = response.choices[0].message.content
print("--- 示例1: JSON结构化(response_format json_schema) ---")
print("原始输出:", raw_output)
character = Character.model_validate_json(raw_output)
print("解析结果:", character)
print(f"  姓名: {character.name.value}, 年龄: {character.age}, 职业: {character.occupation}")


# ============================================================
# 示例2: 正则表达式约束生成
# 使用 structured_outputs.regex 约束输出
# ============================================================

ip_regex = r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)"

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "user", "content": "What is the IP address of Google DNS server?"}
    ],
    max_tokens=64,
    extra_body={
        **EXTRA_BODY,
        "structured_outputs": {"regex": ip_regex},
    },
)

print("\n--- 示例2: 正则约束(structured_outputs regex) ---")
print("IP地址:", response.choices[0].message.content)


# ============================================================
# 示例3: 多项选择约束
# 使用 structured_outputs.choice 约束输出
# ============================================================

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "user", "content": "这段评论的情感倾向是什么? 评论: 这个产品质量很好，非常满意!"}
    ],
    max_tokens=16,
    extra_body={
        **EXTRA_BODY,
        "structured_outputs": {"choice": ["positive", "negative", "neutral"]},
    },
)

print("\n--- 示例3: 多项选择(structured_outputs choice) ---")
print("情感分类:", response.choices[0].message.content)


# ============================================================
# 示例4: 复杂JSON结构 - 订单信息提取
# ============================================================

class OrderItem(BaseModel):
    product_name: str
    quantity: int
    unit_price: float

class OrderInfo(BaseModel):
    customer_name: str
    items: list[OrderItem]
    total_price: float

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "user",
            "content": "Extract order info: Customer Zhang Wei ordered 2 laptops at $999.99 each and 3 mice at $29.99 each."
        }
    ],
    max_tokens=512,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "order-info",
            "schema": OrderInfo.model_json_schema(),
        },
    },
    extra_body=EXTRA_BODY,
)

print("\n--- 示例4: 复杂JSON(订单提取) ---")
raw_output = response.choices[0].message.content
print("原始输出:", raw_output)
order = OrderInfo.model_validate_json(raw_output)
print(f"客户: {order.customer_name}")
for item in order.items:
    print(f"  商品: {item.product_name}, 数量: {item.quantity}, 单价: {item.unit_price}")
print(f"总价: {order.total_price}")
