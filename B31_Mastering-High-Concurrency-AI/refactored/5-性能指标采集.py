#!/usr/bin/env python
# coding: utf-8

"""
性能指标采集 - 从 vLLM Prometheus 端点采集并解析关键指标
对应课件: 监控工具栈
  - vLLM 内置 Metrics (Prometheus格式)
  - 通过 --enable-metrics 暴露
  - curl http://localhost:8000/metrics 查看

前置条件: 启动 vLLM 服务时需要添加 --enable-metrics 参数
"""

import time
import re
import requests
import concurrent.futures

VLLM_URL = "http://localhost:8000"
MODEL_NAME = "/root/autodl-tmp/models/qwen/Qwen3-0.6B"


# =============================================================================
# 第一部分: 解析 Prometheus 指标
# =============================================================================

def fetch_raw_metrics():
    """从 vLLM /metrics 端点获取原始 Prometheus 格式的指标"""
    resp = requests.get(f"{VLLM_URL}/metrics", timeout=10)
    return resp.text


def parse_prometheus_metrics(raw_text):
    """
    解析 Prometheus 格式的文本, 提取课件中提到的关键指标
    课件对应:
      vllm:prompt_tokens_total           (输入token累计)
      vllm:generation_tokens_total       (生成token累计)
      vllm:time_to_first_token_seconds   (TTFT分布)
      vllm:time_per_output_token_seconds (TPOT分布)
      vllm:gpu_kv_cache_usage_percent    (KV Cache利用率)
      vllm:num_requests_running          (运行中的请求数)
      vllm:num_requests_waiting          (等待中的请求数)
    """
    metrics = {}

    # 提取 Counter/Gauge 类型的单值指标
    # 格式: metric_name{labels} value  或  metric_name value
    patterns = {
        "prompt_tokens_total": r'vllm:prompt_tokens_total[^\n]*?\s+([\d.]+)',
        "generation_tokens_total": r'vllm:generation_tokens_total[^\n]*?\s+([\d.]+)',
        "num_requests_running": r'vllm:num_requests_running\s+([\d.]+)',
        "num_requests_waiting": r'vllm:num_requests_waiting\s+([\d.]+)',
        "num_requests_swapped": r'vllm:num_requests_swapped\s+([\d.]+)',
        "gpu_cache_usage_perc": r'vllm:gpu_cache_usage_perc\s+([\d.]+)',
        "cpu_cache_usage_perc": r'vllm:cpu_cache_usage_perc\s+([\d.]+)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, raw_text)
        if match:
            metrics[key] = float(match.group(1))

    # 提取 Histogram 的分位数 (TTFT, TPOT)
    # 格式: vllm:time_to_first_token_seconds_bucket{le="0.5"} 123
    histogram_keys = [
        "time_to_first_token_seconds",
        "time_per_output_token_seconds",
        "e2e_request_latency_seconds",
    ]

    for hkey in histogram_keys:
        count_match = re.search(rf'vllm:{hkey}_count\s+([\d.]+)', raw_text)
        sum_match = re.search(rf'vllm:{hkey}_sum\s+([\d.]+)', raw_text)
        if count_match and sum_match:
            count = float(count_match.group(1))
            total = float(sum_match.group(1))
            if count > 0:
                metrics[f"{hkey}_avg"] = total / count
                metrics[f"{hkey}_count"] = count

        # 提取各 bucket 的值
        buckets = re.findall(
            rf'vllm:{hkey}_bucket\{{le="([^"]+)"\}}\s+([\d.]+)',
            raw_text
        )
        if buckets:
            metrics[f"{hkey}_buckets"] = [
                (float(le) if le != "+Inf" else float("inf"), float(val))
                for le, val in buckets
            ]

    return metrics


def compute_percentile_from_buckets(buckets, percentile):
    """
    从 Prometheus Histogram 的 bucket 数据中估算分位数
    buckets: [(le_bound, cumulative_count), ...]
    percentile: 0-100
    """
    if not buckets:
        return None

    total = buckets[-1][1]
    if total == 0:
        return None

    target = total * (percentile / 100.0)

    prev_bound = 0
    prev_count = 0
    for bound, count in buckets:
        if bound == float("inf"):
            return prev_bound
        if count >= target:
            # 线性插值
            if count == prev_count:
                return bound
            fraction = (target - prev_count) / (count - prev_count)
            return prev_bound + fraction * (bound - prev_bound)
        prev_bound = bound
        prev_count = count

    return buckets[-2][0] if len(buckets) > 1 else 0


def print_metrics_report(metrics):
    """格式化打印指标报告"""
    print("=" * 60)
    print("vLLM 性能指标报告 (来自 Prometheus /metrics)")
    print("=" * 60)

    # 请求状态
    print("\n--- 请求状态 ---")
    print(f"运行中的请求:  {metrics.get('num_requests_running', 'N/A')}")
    print(f"等待中的请求:  {metrics.get('num_requests_waiting', 'N/A')}")
    print(f"换出的请求:    {metrics.get('num_requests_swapped', 'N/A')}")

    # Token 统计
    print("\n--- Token 统计 ---")
    print(f"输入token累计: {metrics.get('prompt_tokens_total', 'N/A')}")
    print(f"生成token累计: {metrics.get('generation_tokens_total', 'N/A')}")

    # KV Cache 利用率 (课件要点: PagedAttention 管理的核心指标)
    print("\n--- KV Cache 利用率 ---")
    gpu_cache = metrics.get('gpu_cache_usage_perc', None)
    if gpu_cache is not None:
        bar_len = int(gpu_cache * 40)
        bar = "#" * bar_len + "-" * (40 - bar_len)
        print(f"GPU KV Cache:  [{bar}] {gpu_cache*100:.1f}%")
    cpu_cache = metrics.get('cpu_cache_usage_perc', None)
    if cpu_cache is not None:
        print(f"CPU KV Cache:  {cpu_cache*100:.1f}%")

    # TTFT 延迟分布
    print("\n--- TTFT (首token延迟) ---")
    ttft_avg = metrics.get('time_to_first_token_seconds_avg')
    if ttft_avg is not None:
        print(f"平均 TTFT:     {ttft_avg*1000:.1f} ms")
        print(f"请求总数:      {metrics.get('time_to_first_token_seconds_count', 'N/A'):.0f}")

    ttft_buckets = metrics.get('time_to_first_token_seconds_buckets')
    if ttft_buckets:
        p50 = compute_percentile_from_buckets(ttft_buckets, 50)
        p95 = compute_percentile_from_buckets(ttft_buckets, 95)
        p99 = compute_percentile_from_buckets(ttft_buckets, 99)
        if p50 is not None:
            print(f"TTFT P50:      {p50*1000:.1f} ms")
        if p95 is not None:
            print(f"TTFT P95:      {p95*1000:.1f} ms")
        if p99 is not None:
            print(f"TTFT P99:      {p99*1000:.1f} ms")

    # TPOT 延迟分布
    print("\n--- TPOT (Token间延迟) ---")
    tpot_avg = metrics.get('time_per_output_token_seconds_avg')
    if tpot_avg is not None:
        print(f"平均 TPOT:     {tpot_avg*1000:.1f} ms")

    tpot_buckets = metrics.get('time_per_output_token_seconds_buckets')
    if tpot_buckets:
        p50 = compute_percentile_from_buckets(tpot_buckets, 50)
        p95 = compute_percentile_from_buckets(tpot_buckets, 95)
        if p50 is not None:
            print(f"TPOT P50:      {p50*1000:.1f} ms")
        if p95 is not None:
            print(f"TPOT P95:      {p95*1000:.1f} ms")

    # 端到端延迟
    print("\n--- 端到端请求延迟 ---")
    e2e_avg = metrics.get('e2e_request_latency_seconds_avg')
    if e2e_avg is not None:
        print(f"平均延迟:      {e2e_avg*1000:.1f} ms")

    e2e_buckets = metrics.get('e2e_request_latency_seconds_buckets')
    if e2e_buckets:
        p50 = compute_percentile_from_buckets(e2e_buckets, 50)
        p99 = compute_percentile_from_buckets(e2e_buckets, 99)
        if p50 is not None:
            print(f"E2E P50:       {p50*1000:.1f} ms")
        if p99 is not None:
            print(f"E2E P99:       {p99*1000:.1f} ms")


# =============================================================================
# 第二部分: 持续监控模式 (每隔几秒采集一次)
# =============================================================================

def monitor_during_load(duration_seconds=30, interval=3):
    """
    在压测期间持续采集指标, 观察指标随时间的变化
    """
    print(f"\n{'='*60}")
    print(f"开始持续监控 (每{interval}秒采集一次, 共{duration_seconds}秒)")
    print(f"{'='*60}")

    snapshots = []
    start = time.time()

    while time.time() - start < duration_seconds:
        try:
            raw = fetch_raw_metrics()
            m = parse_prometheus_metrics(raw)
            elapsed = time.time() - start

            running = m.get('num_requests_running', 0)
            waiting = m.get('num_requests_waiting', 0)
            gpu_cache = m.get('gpu_cache_usage_perc', 0)
            gen_tokens = m.get('generation_tokens_total', 0)

            snapshot = {
                "time_s": elapsed,
                "running": running,
                "waiting": waiting,
                "gpu_cache_pct": gpu_cache * 100 if gpu_cache else 0,
                "gen_tokens": gen_tokens,
            }
            snapshots.append(snapshot)

            print(f"[{elapsed:5.1f}s] "
                  f"运行: {running:.0f}  "
                  f"等待: {waiting:.0f}  "
                  f"KV Cache: {snapshot['gpu_cache_pct']:.1f}%  "
                  f"生成tokens: {gen_tokens:.0f}")

        except Exception as e:
            print(f"采集失败: {e}")

        time.sleep(interval)

    return snapshots


# =============================================================================
# 第三部分: 执行采集
# =============================================================================

# 步骤1: 先采集一次当前状态
print("步骤1: 采集当前 vLLM 指标")
print("-" * 40)
try:
    raw = fetch_raw_metrics()
    metrics = parse_prometheus_metrics(raw)
    print_metrics_report(metrics)
except Exception as e:
    print(f"连接失败: {e}")
    print("请确认 vLLM 服务已启动且开启了 --enable-metrics")
    exit(1)


# 步骤2: 发送一批请求, 同时持续监控指标变化
print("\n\n步骤2: 发送20个并发请求, 同时监控指标变化")
print("-" * 40)

def send_batch_requests():
    """后台线程: 发送一批并发请求"""
    prompts = [
        "请解释什么是KV Cache", "什么是PagedAttention", "什么是Continuous Batching",
        "推测解码的原理是什么", "请解释EAGLE推理框架", "什么是MEDUSA",
        "解释PD分离架构", "什么是TTFT和TPOT", "Chunked Prefill有什么用",
        "GPU利用率低怎么解决",
    ] * 2  # 20个请求

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                requests.post,
                f"{VLLM_URL}/v1/chat/completions",
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": p}],
                    "max_tokens": 150,
                },
                timeout=60
            )
            for p in prompts
        ]
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except Exception:
                pass

import threading
load_thread = threading.Thread(target=send_batch_requests, daemon=True)
load_thread.start()

# 在负载期间持续监控
snapshots = monitor_during_load(duration_seconds=30, interval=2)

load_thread.join(timeout=5)


# 步骤3: 负载结束后再采集一次指标, 对比前后变化
print("\n\n步骤3: 负载结束后的指标")
print("-" * 40)
time.sleep(2)
raw = fetch_raw_metrics()
metrics_after = parse_prometheus_metrics(raw)
print_metrics_report(metrics_after)

# 计算增量
gen_before = metrics.get('generation_tokens_total', 0)
gen_after = metrics_after.get('generation_tokens_total', 0)
print(f"\n--- 本次测试增量 ---")
print(f"新增生成tokens: {gen_after - gen_before:.0f}")
