#!/usr/bin/env python
# coding: utf-8

"""
流式输出与延迟测量 - 精确测量 TTFT 和 TPOT
对应课件: 关键性能指标 KPIs
  - TTFT (Time To First Token): 从请求发送到第一个token输出的时间
  - TPOT (Time Per Output Token): 相邻输出token间隔
"""

import time
from openai import OpenAI

VLLM_URL = "http://localhost:8000/v1"
MODEL_NAME = "/root/autodl-tmp/models/qwen/Qwen3-0.6B"

client = OpenAI(base_url=VLLM_URL, api_key="not-needed")


def measure_streaming_latency(prompt, max_tokens=200):
    """
    通过流式输出精确测量 TTFT 和每个 token 的 TPOT
    """
    messages = [
        {"role": "system", "content": "你是一个有帮助的AI助手。"},
        {"role": "user", "content": prompt}
    ]

    token_timestamps = []
    generated_text = ""

    request_start = time.perf_counter()

    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        stream=True,
        max_tokens=max_tokens,
        temperature=0.7
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            now = time.perf_counter()
            token_timestamps.append(now)
            content = chunk.choices[0].delta.content
            generated_text += content
            print(content, end="", flush=True)

    print()
    request_end = time.perf_counter()

    if len(token_timestamps) < 2:
        print("生成的 token 太少，无法计算 TPOT")
        return

    # 计算 TTFT: 从发送请求到收到第一个 token
    ttft = (token_timestamps[0] - request_start) * 1000

    # 计算每个 token 的间隔 (TPOT)
    token_intervals = []
    for i in range(1, len(token_timestamps)):
        interval = (token_timestamps[i] - token_timestamps[i - 1]) * 1000
        token_intervals.append(interval)

    total_time = (request_end - request_start) * 1000
    num_tokens = len(token_timestamps)

    # 对 TPOT 排序，计算分位数
    sorted_intervals = sorted(token_intervals)
    p50_idx = int(len(sorted_intervals) * 0.50)
    p95_idx = min(int(len(sorted_intervals) * 0.95), len(sorted_intervals) - 1)
    p99_idx = min(int(len(sorted_intervals) * 0.99), len(sorted_intervals) - 1)

    print(f"\n{'='*50}")
    print(f"性能指标报告")
    print(f"{'='*50}")
    print(f"Prompt长度:        {prompt[:30]}...")
    print(f"生成token数:       {num_tokens}")
    print(f"总耗时:            {total_time:.1f} ms")
    print(f"TTFT (首token延迟): {ttft:.1f} ms")
    print(f"TPOT (平均):        {sum(token_intervals)/len(token_intervals):.1f} ms")
    print(f"TPOT P50:           {sorted_intervals[p50_idx]:.1f} ms")
    print(f"TPOT P95:           {sorted_intervals[p95_idx]:.1f} ms")
    print(f"TPOT P99:           {sorted_intervals[p99_idx]:.1f} ms")
    print(f"吞吐量:            {num_tokens / (total_time / 1000):.1f} tokens/s")
    print(f"{'='*50}")

    return {
        "ttft_ms": ttft,
        "tpot_avg_ms": sum(token_intervals) / len(token_intervals),
        "tpot_p50_ms": sorted_intervals[p50_idx],
        "tpot_p95_ms": sorted_intervals[p95_idx],
        "num_tokens": num_tokens,
        "total_ms": total_time,
        "throughput": num_tokens / (total_time / 1000)
    }


# =============================================================================
# 测试不同长度的 Prompt，观察 TTFT 变化
# 课件要点: TTFT 受 Prefill 阶段影响，Prompt 越长 TTFT 越高
# =============================================================================

print("=" * 60)
print("测试1: 短Prompt (Prefill快)")
print("=" * 60)
r1 = measure_streaming_latency("你好", max_tokens=100)

print()
print("=" * 60)
print("测试2: 中等Prompt")
print("=" * 60)
r2 = measure_streaming_latency(
    "请详细解释Transformer架构中的自注意力机制,包括Query、Key、Value的计算过程",
    max_tokens=200
)

print()
print("=" * 60)
print("测试3: 长Prompt (Prefill慢, 观察TTFT变化)")
print("=" * 60)
long_prompt = """请阅读以下技术背景,然后回答问题。
背景: 在大语言模型的推理服务中,KV Cache是关键的性能瓶颈。以LLaMA-2-7B为例,
模型参数包含隐藏维度4096,层数32,注意力头数32。在FP16精度下,每个token的
KV Cache占用约0.5MB。当上下文长度为4096 tokens时,单个请求需要2GB显存。
如果并发100个请求,总共需要200GB,远超单卡A100的80GB显存容量。
PagedAttention通过将KV Cache分割成固定大小的Block(通常16 tokens/块),
实现按需分配和非连续存储,类似操作系统的虚拟内存管理。
问题: 基于以上背景,请解释PagedAttention如何解决KV Cache的内存碎片问题,
以及写时复制(Copy-on-Write)机制在Beam Search中的作用。"""
r3 = measure_streaming_latency(long_prompt, max_tokens=300)

# 对比结果
if r1 and r2 and r3:
    print()
    print("=" * 60)
    print("TTFT 对比 (Prompt长度 vs 首token延迟)")
    print("=" * 60)
    print(f"短Prompt TTFT:   {r1['ttft_ms']:.1f} ms")
    print(f"中Prompt TTFT:   {r2['ttft_ms']:.1f} ms")
    print(f"长Prompt TTFT:   {r3['ttft_ms']:.1f} ms")
    print("结论: Prompt越长, Prefill计算量越大, TTFT越高")
