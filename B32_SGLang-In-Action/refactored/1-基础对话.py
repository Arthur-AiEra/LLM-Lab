#!/usr/bin/env python
# coding: utf-8

# In[1]:


from openai import OpenAI

# 只需要改 base_url，其他代码完全不变
client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="not-needed"  # 本地部署不需要 key
)

response = client.chat.completions.create(
    model="Qwen3-0.6B",
    messages=[
        {"role": "system", "content": "你是一个有帮助的AI助手。"},
        {"role": "user", "content": "用一句话解释什么是大语言模型。"}
    ],
    temperature=0.7,
    max_tokens=200
)

print(response.choices[0].message.content)


# In[ ]:




