#!/usr/bin/env python
# coding: utf-8

# 保险场景 VLM使用

# In[1]:

#t 20:16 https://gemini.google.com/app/a2010c5d9a45156a

import os
from openai import OpenAI
import pandas as pd
import base64

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url="https://api.fe8.cn/v1")

# 调用VLM，得到推理结果
# user_prompt：用户想要分析的内容
# image_url：想要分析的图片
# 新增：一个将本地图片转换为 Base64 编码的辅助函数
# 如代理服务器发送大的单次请求，上传速度慢，为了防止网络阻塞，可以对大图片进行压缩/降采样，然后再进行 Base64 编码
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# 修改后的 VLM 推理函数
# 注意：此时传入的 image_url 应该对应本地的图片路径（例如：'./images/test.jpg'）
def get_response(user_prompt, image_path_input):
    # 得到 image_path_list，一张图片也放到[]中
    if image_path_input.startswith('[') and ',' in image_path_input:
        image_path_input = image_path_input.strip()[1:-1]
        image_path_list = [temp_path.strip() for temp_path in image_path_input.split(',')]
    else:
        image_path_list = [image_path_input]

    # 构建 messages 的 content 列表
    content = [{"type": "text", "text": f"{user_prompt}"}]

    for temp_path in image_path_list:
        # 读取本地图片并进行 Base64 编码
        base64_image = encode_image(temp_path)
        # 拼接成 API 要求的 Data URI 格式
        base64_url = f"data:image/jpeg;base64,{base64_image}"

        content.append({
            "type": "image_url",
            "image_url": {"url": base64_url}
        })

    messages = [{
        "role": "user",
        "content": content
    }]

    print(f'messages prepared for: {image_path_list}')  # 打印时为了控制台干净，建议不要打印巨大的 base64 字符串

    completion = client.chat.completions.create(
        model="qwen-vl-max", # 第三方中转代理（api.fe8.cn）不支持 qwen-vl-max-2025-04-08 这个带有具体日期的模型名称。第三方代理通常只维护标准名称的映射通道
        messages=messages
    )
    return completion


# In[2]:


df = pd.read_excel('./prompt_template_cn_local.xlsx')
df['response'] = ''
for index, row in df.iterrows():
    user_prompt = row['prompt']
    image_url = row['image']
    # 得到VLM推理结果
    completion = get_response(user_prompt, image_url)

    # 增加容错检查：如果 choices 不是 None 且里面有数据
    if hasattr(completion, 'choices') and completion.choices:
        response = completion.choices[0].message.content
        print(f"✅ 成功 | {index + 1} | {user_prompt} | {image_url}")
    else:
        # 如果报错了，打印出 API 真实返回的内容以便排查
        print(f"❌ 失败 | {index + 1} | {image_url}")
        print(f"⚠️ API 返回异常结果: {completion}")
        # ⚠️ API 返回异常结果: ChatCompletion(id=None, choices=None, created=None, model=None, object=None, service_tier=None, system_fingerprint=None, usage=None, error={'code': 'model_not_found', 'message': 'No available channel for model qwen-vl-max-2025-04-08 under group vip (distributor) (request id: 20260314080619876680697Qu9Hycy4)', 'type': 'new_api_error'})

        response = f"API 错误: {completion}"

    df.loc[index, 'response'] = response
    print(f"{index+1} {user_prompt} {image_url}")
df.to_excel('./prompt_template_cn_result-20260314.xlsx', index=False)


# In[6]:


df

