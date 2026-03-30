import os

import outlines
import tiktoken
from openai import OpenAI

#t 127:11 https://gemini.google.com/app/054ebed098f1b878

# outlines 内部使用 tiktoken 获取 tokenizer，需要为 qwen 模型注册兼容的编码器
tiktoken.model.MODEL_TO_ENCODING["qwen-max"] = "cl100k_base"

model_name = "qwen-max"
api_key = os.getenv('OPENAI_API_KEY')
base_url = "https://api.fe8.cn/v1"

# 1. 先创建标准的 OpenAI 客户端实例
client = OpenAI(api_key=api_key, base_url=base_url)

# 2. 调用 outlines.models.openai 模块下的 OpenAI 类，并将 client 传入
model = outlines.models.openai.OpenAI(client, model_name)

# 3. 【修改点】直接调用 model() 或者使用 model.generate() 进行文本生成
# 这种写法在新版 API 中等价于原来的 outlines.generate.text
answer = model("你好，请介绍下自己")
print(answer)