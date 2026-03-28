#!/usr/bin/env python
# coding: utf-8

"""
并发压测 - 模拟高并发场景,测量吞吐量和延迟分布
对应课件:
  - Batch的概念: 把多个请求打包送入GPU
  - 静态Batch vs Continuous Batching: GPU利用率对比
  - 关键性能指标: Throughput, Goodput, TTFT, TPOT
"""

import time
import json
import numpy as np
import concurrent.futures
import requests

VLLM_URL = "http://localhost:8000"
MODEL_NAME = "/root/autodl-tmp/models/qwen/Qwen3-0.6B"


def send_chat_request(prompt, max_tokens=100):
    """
    发送一个聊天请求,返回详细的延迟数据
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是一个有帮助的AI助手。"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False
    }

    start = time.perf_counter()
    try:
        resp = requests.post(f"{VLLM_URL}/v1/chat/completions", json=payload, timeout=60)
        elapsed = time.perf_counter() - start

        if resp.status_code != 200:
            return {"success": False, "error": resp.status_code, "latency_ms": elapsed * 1000}

        data = resp.json()
        prompt_tokens = data["usage"]["prompt_tokens"]
        completion_tokens = data["usage"]["completion_tokens"]

        return {
            "success": True,
            "latency_ms": elapsed * 1000,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {"success": False, "error": str(e), "latency_ms": elapsed * 1000}


def run_concurrent_benchmark(prompts, concurrency, max_tokens=100):
    """
    以指定并发度发送请求,收集性能数据
    """
    print(f"\n{'='*60}")
    print(f"并发压测: {concurrency} 并发, {len(prompts)} 个请求")
    print(f"{'='*60}")

    all_results = []
    bench_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_chat_request, p, max_tokens)
            for p in prompts
        ]
        for f in concurrent.futures.as_completed(futures):
            all_results.append(f.result())

    bench_end = time.perf_counter()
    total_time = bench_end - bench_start

    # 统计成功/失败
    success_results = [r for r in all_results if r["success"]]
    fail_count = len(all_results) - len(success_results)

    if not success_results:
        print("所有请求都失败了!")
        return None

    # 延迟分布
    latencies = np.array([r["latency_ms"] for r in success_results])
    completion_tokens_list = [r["completion_tokens"] for r in success_results]
    total_completion_tokens = sum(completion_tokens_list)
    total_prompt_tokens = sum(r["prompt_tokens"] for r in success_results)

    # 吞吐量
    throughput = total_completion_tokens / total_time  # tokens/s
    qps = len(success_results) / total_time            # queries/s

    # 延迟分位数
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    print(f"\n--- 结果汇总 ---")
    print(f"成功请求:         {len(success_results)}/{len(all_results)}")
    print(f"失败请求:         {fail_count}")
    print(f"总耗时:           {total_time:.2f} s")
    print(f"QPS:              {qps:.2f} req/s")
    print(f"吞吐量:           {throughput:.1f} tokens/s")
    print(f"总输入tokens:     {total_prompt_tokens}")
    print(f"总输出tokens:     {total_completion_tokens}")
    print()
    print(f"--- 延迟分布 (端到端) ---")
    print(f"平均延迟:         {np.mean(latencies):.1f} ms")
    print(f"P50延迟:          {p50:.1f} ms")
    print(f"P95延迟:          {p95:.1f} ms")
    print(f"P99延迟:          {p99:.1f} ms")
    print(f"最小延迟:         {np.min(latencies):.1f} ms")
    print(f"最大延迟:         {np.max(latencies):.1f} ms")

    return {
        "concurrency": concurrency,
        "num_requests": len(all_results),
        "success": len(success_results),
        "total_time_s": total_time,
        "qps": qps,
        "throughput_tokens_s": throughput,
        "latency_avg_ms": float(np.mean(latencies)),
        "latency_p50_ms": float(p50),
        "latency_p95_ms": float(p95),
        "latency_p99_ms": float(p99),
    }


# =============================================================================
# 生成测试用的 Prompt 集合
# =============================================================================

# 模拟真实场景: 不同长度的问题
short_prompts = [
    "你好",
    "今天天气怎么样?",
    "1+1等于几?",
    "什么是AI?",
    "介绍一下Python",
] * 4  # 20个短prompt

medium_prompts = [
    "请详细解释什么是Transformer架构,以及它在NLP中的应用。",
    "请解释深度学习中的反向传播算法,并说明梯度消失问题。",
    "什么是PagedAttention? 它是如何解决KV Cache内存碎片问题的?",
    "请解释Continuous Batching和静态Batching的区别和优势。",
    "推测解码(Speculative Decoding)的核心思想是什么?",
] * 4  # 20个中等prompt

# 混合负载: 更贴近真实场景
mixed_prompts = short_prompts[:10] + medium_prompts[:10]


# =============================================================================
# 测试1: 不同并发度下的性能表现
# 课件要点: GPU擅长并行计算,处理1个请求和8个请求耗时差不多,但吞吐翻8倍
# =============================================================================

print("=" * 60)
print("实验: 不同并发度下的性能对比")
print("观察: 并发度增加时, 吞吐量(tokens/s)如何变化")
print("=" * 60)

concurrency_levels = [1, 2, 4, 8, 16]
comparison_results = []

for c in concurrency_levels:
    # 每种并发度发送20个请求
    result = run_concurrent_benchmark(mixed_prompts[:20], concurrency=c, max_tokens=100)
    if result:
        comparison_results.append(result)
    time.sleep(2)

# 汇总对比
print("\n" + "=" * 60)
print("并发度 vs 性能汇总")
print("=" * 60)
print(f"{'并发度':>6} | {'QPS':>8} | {'吞吐(tok/s)':>12} | {'P50延迟ms':>10} | {'P99延迟ms':>10}")
print("-" * 60)
for r in comparison_results:
    print(f"{r['concurrency']:>6} | {r['qps']:>8.2f} | {r['throughput_tokens_s']:>12.1f} | "
          f"{r['latency_p50_ms']:>10.1f} | {r['latency_p99_ms']:>10.1f}")

print()
print("分析:")
print("- 并发度从1增到8, 吞吐量应该有明显提升 (Continuous Batching的效果)")
print("- 继续增大并发度, P99延迟会上升 (GPU算力接近饱和)")
print("- 吞吐量增长放缓时, 说明GPU已经从Memory-Bound转向Compute-Bound")
