import os
import json

from pydantic import BaseModel
from openai import OpenAI

#t 152:33 https://gemini.google.com/app/054ebed098f1b878

# 食材
class Ingredient(BaseModel):
    name: str
    amount: str
    kcal: int

# 食谱
class Recipe(BaseModel):
    ingredients: list[Ingredient]
    instructions: str

# 创建DashScope客户端(兼容OpenAI协议)
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    base_url="https://api.fe8.cn/v1"
)

# 在system prompt中描述JSON格式
schema_desc = json.dumps(Recipe.model_json_schema(), ensure_ascii=False, indent=2)

# 通过response_format进行结构化输出
completion = client.chat.completions.create(
    model="qwen-turbo",
    messages=[
        {
            "role": "system",
            "content": f"你是一个食谱助手。请严格按照以下JSON Schema格式返回结果:\n{schema_desc}",
        },
        {"role": "user", "content": "请写一个苹果派的食谱"},
    ],
    response_format={"type": "json_object"},
)

apple_pie_recipe = Recipe.model_validate_json(completion.choices[0].message.content)

print(apple_pie_recipe)

#print(apple_pie_recipe.instructions)

#print(completion.choices[0].message.content)
