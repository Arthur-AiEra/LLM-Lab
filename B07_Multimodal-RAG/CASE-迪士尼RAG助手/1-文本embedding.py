#!/usr/bin/env python
# coding: utf-8

# In[1]:


from openai import OpenAI
import json
import os
import logging

# 简单配置：设置根记录器的级别为 DEBUG
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

logging.debug('这是一条 Debug 消息，只有在级别设置为 DEBUG 时才会显示。')
logging.info('这是一条 Info 消息。')

OPENAI_KEY = os.getenv('OPENAI_API_KEY')

# 初始化 OpenAI 客户端
# 请确保已设置环境变量OPENAI_API_KEY或在此处直接设置
# client = OpenAI(
# #     # 如果环境变量中有API密钥，则不需要在此处设置
# #     # api_key=os.environ.get("OPENAI_API_KEY"),
# # # 如果需要直接设置API密钥，请取消下面这行的注释并填入您的API密钥
# #     # api_key="your-api-key"
# )

client = OpenAI(
        api_key=OPENAI_KEY,
        base_url="https://api.fe8.cn/v1" # OpenAI API 代理
    )

text = "上海迪士尼乐园门票分为一日票、两日票和特定日票三种类型。一日票可在购买时选定日期使用，价格根据季节浮动，平日成人票475元起"

# 调用OpenAI的embeddings API (新版本语法)
response = client.embeddings.create(
    model="text-embedding-ada-002",  # OpenAI的文本嵌入模型
    input=text
)

if response:
    result = {
        "status_code": 200,
        # "request_id": response.id,
        "output": {
            "embeddings": response.data[0].embedding
        },
        "usage": {
            "total_tokens": response.usage.total_tokens
        }
    }
    print(json.dumps(result, ensure_ascii=False, indent=4))
else:
    print("Error generating embeddings")
