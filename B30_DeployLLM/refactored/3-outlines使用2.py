import os

import outlines
import tiktoken
from openai import OpenAI

#t 128:15 https://gemini.google.com/app/054ebed098f1b878
# 如果你使用 vLLM，你可以直接通过 API 参数（如 response_format 或 structured_outputs）白嫖它内置的结构化约束能力 。
# 如果你使用 Ollama，你需要自己在 Python 代码里 pip install outlines，然后把它作为一个外部的库，配合 Ollama/llama.cpp 的模型实例来控制结构化输出


# outlines内部使用tiktoken获取tokenizer，需要为qwen模型注册兼容的编码器
tiktoken.model.MODEL_TO_ENCODING["qwen-max"] = "cl100k_base"
tiktoken.model.MODEL_TO_ENCODING["qwen-turbo"] = "cl100k_base"

examples = [
    {"question": "37593 * 67 等于多少?", "code": "37593 * 67"},
    {
        "question": "小红的鸭子每天下16个蛋。她每天早餐吃3个，每天用4个做蛋糕给朋友。她把剩下的鸭蛋以每个2元的价格在市场上卖掉。她每天在市场上能赚多少钱?",
        "code": "(16-3-4)*2",
    },
    {
        "question": "做一件长袍需要2匹蓝色布料和一半数量的白色布料。总共需要多少匹布料?",
        "code": "2 + 2/2",
    },
]

question = "小明正在下载一个200GB的文件。他的下载速度是2GB/分钟，但下载到40%时下载失败了。然后小明不得不从头开始重新下载。他总共花了多少分钟下载完文件?"


@outlines.prompt
def answer_with_code_prompt(question, examples):
    """
    {% for example in examples %}
    QUESTION: {{example.question}}
    CODE: {{example.code}}

    {% endfor %}
    QUESTION: {{question}}
    CODE:"""

def extract_code(text):
    """从模型返回的文本中提取可执行的代码表达式"""
    import re
    # 尝试提取CODE:后面的代码
    match = re.search(r'CODE:\s*(.+)', text)
    if match:
        code_line = match.group(1).strip()
        # 去掉行尾注释
        code_line = re.split(r'\s*#', code_line)[0].strip()
        return code_line
    # 如果没有CODE:前缀，取第一行非空内容
    for line in text.strip().split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            return re.split(r'\s*#', line)[0].strip()
    return text.strip()


prompt = answer_with_code_prompt(question, examples)

# 使用DashScope API(兼容OpenAI协议)
model_name = "qwen-max"
api_key = os.getenv('OPENAI_API_KEY')
base_url = "https://api.fe8.cn/v1"

# 1. 先创建标准的 OpenAI 客户端实例
client = OpenAI(api_key=api_key, base_url=base_url)

# 2. 调用 outlines.models.openai 模块下的 OpenAI 类，并将 client 传入
model = outlines.models.openai.OpenAI(client, model_name)

answer = model(prompt)
print('answer=', answer)
code = extract_code(answer)
print('code=', code)
result = eval(code)
print(f"下载文件总共花费 {result:.0f} 分钟")
