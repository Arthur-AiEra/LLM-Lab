#!/usr/bin/env python
# coding: utf-8

"""
基础对话 - 验证 vLLM 服务是否正常运行
对应课件: vLLM 架构实现详解
"""

from openai import OpenAI

VLLM_URL = "http://localhost:8000/v1"
MODEL_NAME = "/root/autodl-tmp/models/qwen/Qwen3-0.6B"

client = OpenAI(
    base_url=VLLM_URL,
    api_key="not-needed"
)

# --- 单轮对话 ---
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": "你是一个有帮助的AI助手。"},
        {"role": "user", "content": "用一句话解释什么是KV Cache。"}
    ],
    temperature=0.7,
    max_tokens=200
)

print("=== 单轮对话 ===")
print(f"回复: {response.choices[0].message.content}")
print(f"输入tokens: {response.usage.prompt_tokens}")
print(f"输出tokens: {response.usage.completion_tokens}")
print()

# --- 多轮对话 (验证 Continuous Batching 下的多轮交互) ---
messages = [
    {"role": "system", "content": "你是一个AI架构师，擅长解释高并发系统设计。"}
]

questions = [
    "什么是PagedAttention?",
    "它和传统的KV Cache管理有什么区别?",
    "Continuous Batching又是什么?",
]

print("=== 多轮对话 ===")
for q in questions:
    messages.append({"role": "user", "content": q})
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=300,
        temperature=0.7
    )
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    print(f"用户: {q}")
    print(f"助手: {reply}")
    print("---")
