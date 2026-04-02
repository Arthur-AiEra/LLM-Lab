#!/usr/bin/env python
# coding: utf-8

"""
vLLM 与 SGLang 性能对比 - 相同任务下的性能差异
对应课件: SGLang vs vLLM 对比表
  - 缓存策略: PagedAttention(块级复用) vs RadixAttention(前缀树复用)
  - 多轮对话场景 SGLang 缓存命中率更高
  - 批量任务(共享前缀) SGLang 更省计算

前置条件:
  - vLLM 服务运行在 localhost:8000
  - SGLang 服务运行在 localhost:8001
  - 两者加载相同模型 Qwen3-0.6B
"""

import time
import json
import concurrent.futures
import requests

VLLM_URL = "http://localhost:8000/v1"
SGLANG_URL = "http://localhost:8001/v1"
MODEL_NAME = "Qwen3-0.6B"


def benchmark_single(base_url, prompt, max_tokens=100, n_requests=10):
    """对指定服务进行单请求延迟测试"""
    latencies = []
    for _ in range(n_requests):
        start = time.time()
        requests.post(f"{base_url}/completions", json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7
        })
        latencies.append((time.time() - start) * 1000)
    return {
        "avg_ms": sum(latencies) / len(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
    }


def benchmark_concurrent(base_url, prompts, concurrency=8, max_tokens=100):
    """对指定服务进行并发吞吐测试"""
    results = []
    start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                requests.post,
                f"{base_url}/chat/completions",
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": p}],
                    "max_tokens": max_tokens
                },
                timeout=60
            )
            for p in prompts
        ]
        for f in concurrent.futures.as_completed(futures):
            try:
                data = f.result().json()
                results.append(data["usage"]["completion_tokens"])
            except Exception:
                pass

    total_time = time.perf_counter() - start
    total_tokens = sum(results)
    return {
        "success": len(results),
        "total_time_s": total_time,
        "qps": len(results) / total_time,
        "throughput_tokens_s": total_tokens / total_time,
    }


def benchmark_shared_prefix(base_url, n_requests=20, concurrency=10):
    """批量共享前缀测试 (RadixAttention的核心优势场景)"""
    shared_prefix = "你是一个专业的AI助手。请回答以下问题："
    prompts = [f"{shared_prefix}问题{i}" for i in range(n_requests)]

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [
            ex.submit(
                requests.post, f"{base_url}/completions",
                json={"model": MODEL_NAME, "prompt": p, "max_tokens": 50}
            )
            for p in prompts
        ]
        [f.result() for f in futures]

    total_ms = (time.time() - start) * 1000
    return {"total_ms": total_ms, "avg_ms": total_ms / n_requests}


# =============================================================================
# 对比测试
# =============================================================================

def run_comparison():
    """运行完整的对比测试"""

    services = {"vLLM": VLLM_URL, "SGLang": SGLANG_URL}
    all_results = {}

    for name, url in services.items():
        print(f"\n{'='*60}")
        print(f"  测试 {name} (地址: {url})")
        print(f"{'='*60}")

        # 预热
        try:
            requests.post(f"{url}/completions", json={
                "model": MODEL_NAME, "prompt": "hello", "max_tokens": 10
            }, timeout=10)
        except Exception as e:
            print(f"  {name} 连接失败: {e}")
            print(f"  请确认 {name} 服务已启动")
            continue

        results = {}

        # 测试1: 单请求延迟
        print(f"\n  [1/3] 单请求延迟...")
        results["single"] = benchmark_single(url, "什么是深度学习?", n_requests=10)
        print(f"    平均延迟: {results['single']['avg_ms']:.0f}ms")

        # 测试2: 共享前缀 (RadixAttention优势场景)
        print(f"\n  [2/3] 共享前缀 (20请求)...")
        results["shared_prefix"] = benchmark_shared_prefix(url)
        print(f"    总耗时: {results['shared_prefix']['total_ms']:.0f}ms")

        # 测试3: 并发吞吐
        print(f"\n  [3/3] 并发吞吐 (8并发, 20请求)...")
        prompts = [
            "什么是深度学习?", "解释反向传播", "什么是注意力机制",
            "Transformer架构的核心是什么", "什么是梯度下降",
        ] * 4
        results["concurrent"] = benchmark_concurrent(url, prompts)
        print(f"    QPS: {results['concurrent']['qps']:.2f}")
        print(f"    吞吐: {results['concurrent']['throughput_tokens_s']:.1f} tok/s")

        all_results[name] = results

    # 对比汇总
    if len(all_results) == 2:
        print(f"\n{'='*70}")
        print("对比汇总: vLLM vs SGLang")
        print(f"{'='*70}")
        print(f"{'指标':>20} | {'vLLM':>12} | {'SGLang':>12} | {'差异':>10}")
        print("-" * 70)

        v = all_results["vLLM"]
        s = all_results["SGLang"]

        rows = [
            ("单请求延迟(ms)", v["single"]["avg_ms"], s["single"]["avg_ms"]),
            ("共享前缀耗时(ms)", v["shared_prefix"]["total_ms"], s["shared_prefix"]["total_ms"]),
            ("QPS(8并发)", v["concurrent"]["qps"], s["concurrent"]["qps"]),
            ("吞吐(tok/s)", v["concurrent"]["throughput_tokens_s"], s["concurrent"]["throughput_tokens_s"]),
        ]

        for label, vv, sv in rows:
            if "延迟" in label or "耗时" in label:
                diff = f"{sv/vv:.2f}x" if vv > 0 else "N/A"
            else:
                diff = f"{sv/vv:.2f}x" if vv > 0 else "N/A"
            print(f"{label:>20} | {vv:>12.1f} | {sv:>12.1f} | {diff:>10}")

        # 保存结果
        with open("vllm_vs_sglang.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到 vllm_vs_sglang.json")

        print(f"\n分析:")
        print(f"- 共享前缀场景: SGLang 的 RadixAttention 自动识别前缀并复用 KV Cache")
        print(f"- 单请求延迟: 两者差异不大 (底层都使用高效的 Attention 实现)")
        print(f"- 吞吐量: 取决于各自的调度策略和配置参数")
    else:
        print("\n只有一个服务可用，无法对比。请确认 vLLM (8000) 和 SGLang (8001) 都已启动。")


if __name__ == "__main__":
    run_comparison()
