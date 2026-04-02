#!/usr/bin/env python
# coding: utf-8

# In[1]:


from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="not-needed")

# 流式输出：stream=True
stream = client.chat.completions.create(
    model="Qwen3-0.6B",
    messages=[
        {"role": "user", "content": "写一首关于春天的五言绝句。"}
    ],
    stream=True,
    max_tokens=200
)

# 逐块接收并打印
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # 换行

