#!/usr/bin/env python
# coding: utf-8

# In[1]:


from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="not-needed")

# 维护对话历史
messages = [
    {"role": "system", "content": "你是一个电商客服，帮助用户查询订单和处理退款。"}
]

# 模拟多轮对话
user_inputs = [
    "我的订单 2024001 到哪了？",
    "那这个订单可以退款吗？",         # 第二轮：复用第一轮的 KV Cache
    "退款需要多久到账？",              # 第三轮：复用前两轮的 KV Cache
]

for user_input in user_inputs:
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="Qwen3-0.6B",
        messages=messages,
        max_tokens=300,
        temperature=0.7
    )

    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})

    print(f"用户: {user_input}")
    print(f"客服: {reply}")
    print("---")

