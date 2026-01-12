import os
from openai import OpenAI

# client = OpenAI(
#     api_key=os.getenv("DASHSCOPE_API_KEY"),  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
# )

OPENAI_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(
        api_key=OPENAI_KEY,
        base_url="https://api.fe8.cn/v1" # OpenAI API 代理
    )

# 调用OpenAI的embeddings API (新版本语法)
response = client.embeddings.create(
    model="text-embedding-ada-002",  # OpenAI的文本嵌入模型
    input='我想知道迪士尼的退票政策',
    encoding_format="float"
)
print(response.model_dump_json())

# completion = client.embeddings.create(
#     model="text-embedding-v4",
#     input='我想知道迪士尼的退票政策',
#     dimensions=1024, # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
#     encoding_format="float"
# )

#print(completion.model_dump_json())