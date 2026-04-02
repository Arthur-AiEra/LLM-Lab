#!/usr/bin/env python
# coding: utf-8

# In[2]:


import concurrent.futures
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="not-needed")

# 待处理的文章列表
articles = [
    "人工智能正在改变医疗行业，通过深度学习算法，AI可以辅助医生进行影像诊断...",
    "区块链技术在供应链管理中的应用越来越广泛，它提供了透明的追溯机制...",
    "量子计算的突破使得传统加密算法面临挑战，后量子密码学成为研究热点...",
    # ... 更多文章
]

# 共享的评分 Prompt 模板（这部分的 KV Cache 只计算一次）
SYSTEM_PROMPT = """你是一个专业的文章评审员。请从以下三个维度评估文章：
1. 清晰度（1-10分）：逻辑是否清晰，表达是否流畅
2. 深度（1-10分）：是否有深入分析，不是泛泛而谈
3. 实用性（1-10分）：对读者是否有实际参考价值
请用 JSON 格式返回评分结果。"""

def score_article(article):
    response = client.chat.completions.create(
        model="Qwen3-0.6B",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"请评估以下文章：\n\n{article}"}
        ],
        max_tokens=1000,
        temperature=0.3
    )
    return response.choices[0].message.content

# 并发发送请求（SGLang 自动识别共享前缀并复用 KV Cache）
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(score_article, articles))

for i, result in enumerate(results):
    print(f"文章 {i+1} 评分: {result}")

