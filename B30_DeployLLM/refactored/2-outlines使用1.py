import os
import tiktoken
import outlines
import outlines.models as models

# outlines 内部使用 tiktoken 获取 tokenizer，需要为 qwen 模型注册兼容的编码器
tiktoken.model.MODEL_TO_ENCODING["qwen-max"] = "cl100k_base"

model_name = "qwen-max"
api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

model = models.openai(model_name, api_key=api_key, base_url=base_url)
answer = outlines.generate.text(model)("你好，请介绍下自己")
print(answer)