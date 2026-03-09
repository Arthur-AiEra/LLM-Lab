# macOS 兼容版本使用指南

## 文件说明

- **Qwen2_5_(7B)_Alpaca.py** - 原始版本（仅支持 NVIDIA GPU 的 Unsloth 框架）
- **Qwen2_5_(7B)_Alpaca_macOS.py** - macOS 兼容版本（使用 Transformers + PEFT）
- **requirements.txt** - 原始依赖（包含 Unsloth）
- **requirements_macos.txt** - macOS 依赖

## 主要改动

### 1. 替换框架
- ❌ `unsloth` → ✅ `transformers` + `peft`
- Unsloth 依赖 NVIDIA GPU，不支持 macOS

### 2. 设备支持
- **Metal GPU (M 芯片 Mac)**: 自动检测并使用 MPS 加速 ⚡
- **CPU**: 自动降级到 CPU（较慢）
- **CUDA**: 仍然支持有 NVIDIA GPU 的设备

### 3. 内存优化
- 减小批大小：2 → 1
- 增加梯度累积：4 → 8（保持等效更新）
- 移除 4-bit 量化（macOS Metal 不支持）

### 4. 数据加载
- 支持本地 JSON 数据集
- 支持从 HuggingFace 下载

## 安装步骤

### 第 1 步：创建虚拟环境
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 第 2 步：安装依赖
```bash
pip install -r requirements_macos.txt
```

### 第 3 步：准备模型和数据

#### 方案 A：本地模型路径
编辑脚本中的 `model_path` 变量：
```python
model_path = "/path/to/your/Qwen2.5-7B-Instruct"
```

#### 方案 B：从 HuggingFace 自动下载
脚本会自动下载 `Qwen/Qwen2.5-7B-Instruct`（需要 ~15GB 存储空间）

#### 数据集
脚本会搜索本地 `【数据集】alpaca-cleaned/alpaca_data_cleaned.json`
如果不存在，会从 HuggingFace 自动下载

### 第 4 步：运行训练
```bash
python Qwen2_5_(7B)_Alpaca_macOS.py
```

## 性能对比

| 设备 | 框架 | 速度 | 显存 |
|------|------|------|------|
| M2 Max (30GB) | Transformers + Metal | 基准 (1x) | ~20GB |
| NVIDIA 3090 | Unsloth | 5-10x | ~10GB |
| NVIDIA A100 | Unsloth | 10-40x | ~8GB |

## 常见问题

### Q1: 训练太慢了？
**A**: 这是正常的。macOS Metal 性能远低于 NVIDIA GPU。
- 对于完整训练，建议使用云 GPU（AutoDL、Lambda Labs 等）
- 这个版本适合测试和开发

### Q2: 内存不足？
**A**: 尝试以下方法：
1. 减少 `batch_size`（脚本已设为 1）
2. 减少 `max_seq_length`
3. 使用更小的模型（如 Qwen2.5-3B）
4. 启用梯度检查点：`gradient_checkpointing=True`

### Q3: 无法加载本地模型？
**A**: 脚本有自动降级机制
- 本地路径失败 → 自动从 HuggingFace 下载
- 需要 ~15GB 空间和稳定网络连接

### Q4: 如何使用 Metal GPU？
**A**: 自动检测。确保：
- PyTorch 版本 ≥ 1.12
- 使用 M 芯片 Mac（M1/M2/M3 及以上）
- 检查输出信息：`Using Metal GPU (MPS) acceleration`

### Q5: 如何加速训练？
**A**: 最直接的方式是使用远程 GPU：
```bash
# 使用原始 requirements.txt 和 Qwen2_5_(7B)_Alpaca.py
pip install -r requirements.txt
python Qwen2_5_(7B)_Alpaca.py
```

## 输出示例

```
Device: mps, Dtype: torch.float16
✓ Loading model from Qwen/Qwen2.5-7B-Instruct...
✓ Model loaded successfully
trainable params: 41,943,040 || all params: 7,613,513,728 || trainable%: 0.551
✓ Dataset loaded successfully
Dataset size: 52002
Starting training...

[1/1, Epoch 0]: loss=2.341, learning_rate=0.000200
[2/1, Epoch 0]: loss=2.287, learning_rate=0.000200
...

✓ Training completed in 3600.45 seconds (60.01 minutes)

Test 1: Fibonacci Sequence
--------------------------------------------------
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Continue the fibonacci sequence.

### Input:
1, 1, 2, 3, 5, 8

### Response:
13, 21, 34, 55, 89, 144...

✓ Model saved to lora_model_macos
```

## 高级配置

### 修改训练参数
编辑 `training_args`：
```python
training_args = TrainingArguments(
    per_device_train_batch_size=1,      # 批大小
    num_train_epochs=1,                  # 训练轮数
    learning_rate=2e-4,                  # 学习率
    max_steps=100,                       # 最大步数
    logging_steps=1,                     # 日志频率
    save_steps=50,                       # 保存频率
)
```

### 调整 LoRA 参数
编辑 `peft_config`：
```python
peft_config = LoraConfig(
    r=16,              # LoRA 秩（越小越快，效果越差）
    lora_alpha=16,     # LoRA 缩放因子
    lora_dropout=0.05, # Dropout（过拟合时增加）
    target_modules=[...],  # 目标模块
)
```

## 与原始版本的功能对比

| 功能 | 原始 (Unsloth) | macOS 版本 |
|------|--------|---------|
| 4-bit 量化 | ✅ | ❌ (不支持) |
| LoRA 微调 | ✅ | ✅ |
| Metal GPU | ❌ | ✅ |
| Stream 推理 | ✅ | ✅ |
| 模型保存 | ✅ | ✅ |
| 性能 | ⚡⚡⚡ | ⚡ |

## 推荐工作流

### 开发/测试阶段
```bash
# 在 macOS 上本地开发和测试
python Qwen2_5_(7B)_Alpaca_macOS.py
```

### 完整训练
```bash
# 连接到远程 GPU，使用原始脚本
ssh user@autodl-instance
python Qwen2_5_(7B)_Alpaca.py
```

## 更多帮助

- [PyTorch Metal 文档](https://pytorch.org/docs/stable/notes/mps.html)
- [PEFT 文档](https://huggingface.co/docs/peft/)
- [Transformers 文档](https://huggingface.co/docs/transformers/)
- [Qwen 模型文档](https://huggingface.co/Qwen)
