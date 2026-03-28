#!/usr/bin/env python
# coding: utf-8

# In[1]:


from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # vLLM 不验证 key，但必须填
)

response = client.chat.completions.create(
    model="/root/autodl-tmp/models/qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "你好，请介绍下自己"}],
    max_tokens=512
)

print(response.choices[0].message.content)

