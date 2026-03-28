import os
import json
from pydantic import BaseModel
from openai import OpenAI

class Step(BaseModel):
    explanation: str
    output: str

class MathResponse(BaseModel):
    steps: list[Step]
    final_answer: str


# 创建DashScope客户端(兼容OpenAI协议)
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 在system prompt中描述JSON格式，配合response_format=json_object使用
schema_desc = json.dumps(MathResponse.model_json_schema(), ensure_ascii=False, indent=2)

# 通过response_format进行结构化输出
completion = client.chat.completions.create(
    model="qwen-max",
    messages=[
        {
            "role": "system",
            "content": f"你是一个数学辅导老师。请严格按照以下JSON Schema格式返回解题过程:\n{schema_desc}",
        },
        {"role": "user", "content": "解方程 8x + 31 = 2"},
    ],
    response_format={"type": "json_object"},
)

# 手动解析JSON为Pydantic模型
message = completion.choices[0].message
result = MathResponse.model_validate_json(message.content)
print(result.steps)
print(result.final_answer)
