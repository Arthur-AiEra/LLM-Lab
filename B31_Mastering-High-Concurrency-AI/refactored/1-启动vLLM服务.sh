#!/bin/bash
# =============================================================================
# vLLM 服务启动脚本 - 配合课件《AI服务核心：高并发原理与性能监控调优》
# 硬件: RTX 4090D (24GB)
# 模型: Qwen3-0.6B
# =============================================================================

#t 140:15 https://gemini.google.com/app/21d39e366127ce06
# bash 1-启动vLLM服务.sh basic => 需要gpu环境，auto dl(https://autodl.com/home)


MODEL_NAME="/private/var/ifc/app_data/autodl-tmp/models/Qwen/Qwen3.5-0.8B"
PORT=8000

# ---- 场景A: 基础版 (验证环境可用) ----
# 最简配置，用于快速验证模型加载和推理是否正常
start_basic() {
    /usr/local/bin/python3.12 -m vllm.entrypoints.openai.api_server \
        --model $MODEL_NAME \
        --port $PORT \
        --dtype float16 \
        --enforce-eager \
        --trust-remote-code
}

# ---- 场景B: 高吞吐配置 (最大化 tokens/s) ----
# 对应课件「目标A: 最高吞吐」
# 增大 batch、开启前缀缓存、提高显存利用率
start_high_throughput() {
    python -m vllm.entrypoints.openai.api_server \
        --model $MODEL_NAME \
        --port $PORT \
        --dtype float16 \
        --max-model-len 4096 \
        --max-num-seqs 64 \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.92 \
        --enable-prefix-caching \
        --enforce-eager \
        --trust-remote-code
}

# ---- 场景C: 低延迟配置 (最小化 TTFT/TPOT) ----
# 对应课件「目标B: 最低延迟」
# 开启 Chunked Prefill，限制 batch 大小
start_low_latency() {
    python -m vllm.entrypoints.openai.api_server \
        --model $MODEL_NAME \
        --port $PORT \
        --dtype float16 \
        --max-model-len 4096 \
        --max-num-seqs 16 \
        --max-num-batched-tokens 512 \
        --enable-chunked-prefill \
        --gpu-memory-utilization 0.90 \
        --enforce-eager \
        --trust-remote-code
}

# ---- 场景D: 省显存配置 (解决 OOM) ----
# 对应课件「目标C: 节省显存」
# 缩短序列长度、限制并发数
start_save_memory() {
    python -m vllm.entrypoints.openai.api_server \
        --model $MODEL_NAME \
        --port $PORT \
        --dtype float16 \
        --max-model-len 2048 \
        --max-num-seqs 16 \
        --gpu-memory-utilization 0.85 \
        --preemption-mode swap \
        --enforce-eager \
        --trust-remote-code
}

# =============================================================================
# 使用方式:
#   bash 1-启动vLLM服务.sh basic           # 基础版
#   bash 1-启动vLLM服务.sh high_throughput  # 高吞吐
#   bash 1-启动vLLM服务.sh low_latency     # 低延迟
#   bash 1-启动vLLM服务.sh save_memory     # 省显存
# =============================================================================

case "${1:-basic}" in
    basic)           start_basic ;;
    high_throughput) start_high_throughput ;;
    low_latency)     start_low_latency ;;
    save_memory)     start_save_memory ;;
    *)
        echo "用法: bash $0 {basic|high_throughput|low_latency|save_memory}"
        exit 1
        ;;
esac
