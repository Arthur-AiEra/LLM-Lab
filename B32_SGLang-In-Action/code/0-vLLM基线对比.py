#!/usr/bin/env python
# coding: utf-8

"""
vLLM 基线对比 - 与 SGLang 做性能对比的基准测试
对应课件: SGLang vs vLLM
  - vLLM: 通用推理引擎 (像 MySQL)
  - SGLang: 复杂LLM程序运行时 (像 Redis+Lua)

使用方式:
  1. 先启动 vLLM 服务 (端口 8000)
  2. 运行本脚本，记录 vLLM 的性能基线
  3. 再启动 SGLang 服务 (端口 8001)
  4. 运行 6-性能测试.py，对比两者差异

前置:
  python -m vllm.entrypoints.openai.api_server \
      --model /root/autodl-tmp/models/qwen/Qwen3-0.6B \
      --port 8000 --dtype float16 --enforce-eager --trust-remote-code
"""

import time
import json
import concurrent.futures
import requests
from openai import OpenAI

VLLM_URL = "http://localhost:8000"
VLLM_API = f"{VLLM_URL}/v1"
MODEL_NAME = "Qwen3-0.6B"

client = OpenAI(base_url=VLLM_API, api_key="not-needed")


# =============================================================================
# 测试1: 冷启动 vs 缓存命中 (与SGLang的RadixAttention对比)
# =============================================================================

def test_caching():
    """
    vLLM 的 prefix caching vs SGLang 的 RadixAttention
    课件要点: SGLang 通过 Radix 树自动识别共享前缀并复用 KV Cache
    """
    url = f"{VLLM_URL}/v1/completions"
    prompt = "请解释什么是机器学习"

    print("=" * 60)
    print("测试1: 冷启动 vs 缓存命中 (vLLM)")
    print("=" * 60)

    # 冷启动
    start = time.time()
    requests.post(url, json={
        "model": MODEL_NAME, "prompt": prompt,
        "max_tokens": 100, "temperature": 0.7
    })
    cold_latency = (time.time() - start) * 1000
    print(f"冷启动延迟: {cold_latency:.0f}ms")

    # 缓存命中 (重复相同前缀)
    latencies = []
    for _ in range(10):
        start = time.time()
        requests.post(url, json={
            "model": MODEL_NAME, "prompt": prompt,
            "max_tokens": 100, "temperature": 0.7
        })
        latencies.append((time.time() - start) * 1000)

    avg = sum(latencies) / len(latencies)
    print(f"缓存命中平均延迟: {avg:.0f}ms (10次)")
    print(f"加速比: {cold_latency / avg:.2f}x")
    print()
    return {"cold_ms": cold_latency, "warm_avg_ms": avg}


# =============================================================================
# 测试2: 批量共享前缀 (RadixAttention 的优势场景)
# =============================================================================

def test_shared_prefix():
    """
    课件要点: SGLang 通过 RadixAttention 只计算一次共享前缀的 KV Cache
    vLLM 需要 --enable-prefix-caching 才能实现类似效果
    """
    print("=" * 60)
    print("测试2: 批量共享前缀 (vLLM)")
    print("=" * 60)

    # 所有 prompt 共享相同前缀
    shared_prefix = "你是一个专业的AI助手。请回答以下问题："
    prompts = [f"{shared_prefix}问题{i}" for i in range(20)]

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        futures = [
            ex.submit(
                requests.post,
                f"{VLLM_URL}/v1/completions",
                json={"model": MODEL_NAME, "prompt": p, "max_tokens": 50}
            )
            for p in prompts
        ]
        [f.result() for f in futures]
    elapsed = (time.time() - start) * 1000
    print(f"20个共享前缀请求总耗时: {elapsed:.0f}ms")
    print(f"平均每请求: {elapsed / 20:.0f}ms")
    print()
    return {"total_ms": elapsed, "avg_ms": elapsed / 20}


# =============================================================================
# 测试3: 多轮对话 (KV Cache 复用场景)
# =============================================================================

def test_multi_turn():
    """
    课件要点: 多轮对话是 SGLang 的优势场景
    RadixAttention 自动缓存之前轮次的 KV Cache
    vLLM 也支持，但 SGLang 的 Radix 树结构在多会话并发时命中率更高
    """
    print("=" * 60)
    print("测试3: 多轮对话延迟 (vLLM)")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "你是一个电商客服，帮助用户查询订单和处理退款。"}
    ]
    user_inputs = [
        "我的订单 2024001 到哪了？",
        "那这个订单可以退款吗？",
        "退款需要多久到账？",
    ]

    turn_latencies = []
    for i, user_input in enumerate(user_inputs):
        messages.append({"role": "user", "content": user_input})
        start = time.time()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=200,
            temperature=0.7
        )
        elapsed = (time.time() - start) * 1000
        turn_latencies.append(elapsed)

        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        print(f"第{i+1}轮延迟: {elapsed:.0f}ms")

    print()
    return {"turn_latencies_ms": turn_latencies}


# =============================================================================
# 测试4: 并发吞吐 (基线数据)
# =============================================================================

def test_throughput():
    """与 SGLang 6-性能测试.py 的并发测试做对比"""
    print("=" * 60)
    print("测试4: 并发吞吐 (vLLM, 8并发)")
    print("=" * 60)

    prompts = [
        "什么是深度学习?", "解释反向传播算法", "什么是注意力机制",
        "Transformer架构的核心是什么", "什么是梯度下降",
    ] * 4  # 20个请求

    results = []
    start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                requests.post,
                f"{VLLM_URL}/v1/chat/completions",
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": p}],
                    "max_tokens": 100
                },
                timeout=60
            )
            for p in prompts
        ]
        for f in concurrent.futures.as_completed(futures):
            try:
                resp = f.result()
                data = resp.json()
                results.append(data["usage"]["completion_tokens"])
            except Exception:
                pass

    total_time = time.perf_counter() - start
    total_tokens = sum(results)
    throughput = total_tokens / total_time
    qps = len(results) / total_time

    print(f"成功请求: {len(results)}/20")
    print(f"总耗时: {total_time:.2f}s")
    print(f"QPS: {qps:.2f} req/s")
    print(f"吞吐量: {throughput:.1f} tokens/s")
    print()
    return {"qps": qps, "throughput_tokens_s": throughput, "total_time_s": total_time}


# =============================================================================
# 主流程
# =============================================================================

print("=" * 60)
print("vLLM 基线性能测试")
print(f"服务地址: {VLLM_URL}")
print(f"模型: {MODEL_NAME}")
print("=" * 60)
print()

all_results = {}
all_results["caching"] = test_caching()
all_results["shared_prefix"] = test_shared_prefix()
all_results["multi_turn"] = test_multi_turn()
all_results["throughput"] = test_throughput()

# 保存结果,方便和 SGLang 对比
with open("vllm_baseline.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print("=" * 60)
print("vLLM 基线结果已保存到 vllm_baseline.json")
print("接下来启动 SGLang 服务,运行 6-性能测试.py 做对比")
print("=" * 60)

# 打印汇总
print()
print("汇总 (vLLM 基线):")
print(f"  冷启动延迟:       {all_results['caching']['cold_ms']:.0f}ms")
print(f"  缓存命中延迟:     {all_results['caching']['warm_avg_ms']:.0f}ms")
print(f"  共享前缀(20请求): {all_results['shared_prefix']['total_ms']:.0f}ms")
print(f"  并发吞吐(8并发):  {all_results['throughput']['throughput_tokens_s']:.1f} tok/s")
print(f"  QPS(8并发):       {all_results['throughput']['qps']:.2f} req/s")
