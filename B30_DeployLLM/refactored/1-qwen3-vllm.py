#!/usr/bin/env python
# coding: utf-8

# In[1]:


from openai import OpenAI

#t 86:33 https://gemini.google.com/app/054ebed098f1b878
# pip install vllm --upgrade
# 强制绑定本地 IP 和 Mac 的本地网卡接口
# export VLLM_HOST_IP=127.0.0.1
# export GLOO_SOCKET_IFNAME=lo0
# vllm serve /private/var/ifc/app_data/autodl-tmp/models/Qwen/Qwen3.5-0.8B \
# --enforce-eager \
# --gpu-memory-utilization 0.9
#
# curl http://localhost:8000/v1/models

client = OpenAI(
    base_url="http://localhost:8000/v1", # 需要在gpu上跑，vLLM 是为 Linux 环境下的 NVIDIA/AMD 显卡原生设计的，它并不支持 macOS 的 GPU 加速 (Metal/MPS)
    api_key="dummy"  # vLLM 不验证 key，但必须填
)

response = client.chat.completions.create(
    model="/private/var/ifc/app_data/autodl-tmp/models/Qwen/Qwen3.5-0.8B",
    messages=[{"role": "user", "content": "你好，请介绍下自己 /no_think"}],
    max_tokens=512
)

print(response.choices[0].message.content)

