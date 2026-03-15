#!/usr/bin/env python
# coding: utf-8

import base64
import os

from openai import OpenAI

#t 37:59 https://gemini.google.com/app/a2010c5d9a45156a

# 1. 初始化 OpenAI 客户端 (可使用第三方代理或官方兼容接口)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.fe8.cn/v1" # 若用官方则换为 https://dashscope.aliyuncs.com/compatible-mode/v1
)

# 2. 本地图片转 Base64 的辅助函数
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

local_file_path = '2.jpg'

# 3. 读取本地图片并拼接成标准 Data URI
base64_image = encode_image(local_file_path)
base64_url = f"data:image/jpeg;base64,{base64_image}"

# 4. 构建标准的 OpenAI 兼容 messages
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "图片里有什么东西?"
            },
            {
                "type": "image_url",
                "image_url": {"url": base64_url}
            }
        ]
    }
]

# 5. 调用大模型 (注意模型名称)
response = client.chat.completions.create(
    model="qwen-vl-plus",
    messages=messages
)

# 6. 打印输出
print(response.choices[0].message.content)