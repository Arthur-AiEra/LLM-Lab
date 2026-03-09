#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速测试 macOS Metal GPU 兼容性"""

import torch
import sys

print("=" * 60)
print("macOS Metal GPU 兼容性测试")
print("=" * 60)

# 测试 PyTorch
print("\n✓ PyTorch 版本:", torch.__version__)

# 测试 MPS 支持
if torch.backends.mps.is_available():
    print("✓ Metal GPU (MPS) 可用")
    device = torch.device("mps")
else:
    print("✗ Metal GPU (MPS) 不可用")
    device = torch.device("cpu")

print(f"✓ 使用设备: {device}")

# 测试 CUDA
if torch.cuda.is_available():
    print(f"✓ CUDA 可用: {torch.cuda.get_device_name(0)}")
else:
    print("- CUDA 不可用（正常）")

# 创建简单的张量操作
print("\n测试张量操作...")
x = torch.randn(10, 10).to(device)
y = torch.randn(10, 10).to(device)
z = torch.matmul(x, y)
print(f"✓ 矩阵乘法成功: {z.shape}")

# 测试梯度计算
print("\n测试梯度计算...")
x = torch.randn(10, requires_grad=True, device=device)
y = (x ** 2).sum()
y.backward()
print(f"✓ 梯度计算成功: {x.grad is not None}")

print("\n" + "=" * 60)
print("所有测试通过！准备加载大型模型...")
print("=" * 60)
