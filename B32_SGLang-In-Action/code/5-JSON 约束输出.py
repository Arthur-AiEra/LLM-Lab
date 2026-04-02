#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import json

# 使用 SGLang 原生 API 进行约束解码（通过 regex 参数）
response = requests.post(
    "http://localhost:8001/generate",
    json={
        "text": "Extract the person's information from the following text.\n\nText: My name is Zhang Wei, I am 28 years old, I work as a software engineer at ByteDance.\n\nOutput:",
        "sampling_params": {
            "max_new_tokens": 200,
            "temperature": 0,
            "regex": r'\{"name": "[^"]+", "age": \d+, "company": "[^"]+", "role": "[^"]+"\}'
        }
    }
)

result = json.loads(response.json()["text"])
print(json.dumps(result, ensure_ascii=False, indent=2))
# 输出一定是合法 JSON：
# {"name": "Zhang Wei", "age": 28, "company": "ByteDance", "role": "software engineer"}

