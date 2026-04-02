#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import requests

def test_performance(prompt, n_requests=10):
    """简单的性能测试：测量 TTFT 和总延迟"""
    url = "http://localhost:8001/v1/completions"
    latencies = []

    for i in range(n_requests):
        start = time.time()
        response = requests.post(url, json={
            "model": "Qwen3-0.6B",
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.7
        })
        elapsed = time.time() - start
        latencies.append(elapsed)

    avg_latency = sum(latencies) / len(latencies)
    print(f"平均延迟: {avg_latency*1000:.0f}ms")
    print(f"最小延迟: {min(latencies)*1000:.0f}ms")
    print(f"最大延迟: {max(latencies)*1000:.0f}ms")

# 测试 1：冷启动（第一次请求，无缓存）
print("=== 冷启动测试 ===")
test_performance("请解释什么是机器学习", n_requests=1)

# 测试 2：缓存命中（重复相同前缀的请求）
print("=== 缓存命中测试 ===")
test_performance("请解释什么是机器学习", n_requests=10)

# 测试 3：批量共享前缀
print("=== 共享前缀测试 ===")
import concurrent.futures
prompts = [f"你是一个专业的AI助手。请回答以下问题：问题{i}" for i in range(20)]
start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
    futures = [ex.submit(
        requests.post, "http://localhost:8001/v1/completions",
        json={"model": "Qwen3-0.6B", "prompt": p, "max_tokens": 50}
    ) for p in prompts]
    [f.result() for f in futures]
print(f"20 个请求总耗时: {(time.time()-start)*1000:.0f}ms")

