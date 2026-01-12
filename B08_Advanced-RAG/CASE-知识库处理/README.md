分析文件中的代码，梳理函数及其调用关系，并使用 Mermaid 流程图和中文详细解释各部分功能

==============================================================================

分析文件中的代码，梳理函数及其调用关系，帮我整理到 .html 可以用mermaid和svg

==============================================================================

比较一下 text-embedding-ada-002 和 text-embedding-v4 这两种 embedding 模型。
并说明为何在执行以下python 代码时，应用 text-embedding-ada-002 可以调用成功，但 应用 text-embedding-v4 会报”openai.InternalServerError: Error code: 500 - {'message': 'Server Error'}“

from openai import OpenAI
OPENAI_KEY = os.getenv('OPENAI_API_KEY')

TEXT_EMBEDDING_MODEL = "text-embedding-ada-002"

client = OpenAI(
        api_key=OPENAI_KEY,
        base_url="https://api.fe8.cn/v1" # OpenAI API 代理
    )

"""获取文本的 Embedding"""
    response = client.embeddings.create(
        model=TEXT_EMBEDDING_MODEL,
        input=text
    )