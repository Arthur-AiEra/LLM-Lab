# Qwen2-VL 3B macOS 兼容微调指南

## 问题诊断

您的原始脚本存在以下问题：
```
NotImplementedError: Unsloth currently only works on NVIDIA, AMD and Intel GPUs.
```

**原因**：`unsloth` 库只支持 NVIDIA/AMD/Intel 的独立 GPU，不支持 macOS 的 Metal GPU 或 CPU。

## 解决方案

已为您创建 **macOS 兼容版本**：`qwen_vl_car_insurance_train_macOS.py`

### 主要改动

| 项目 | 原始版本 | macOS 版本 |
|------|--------|---------|
| GPU 框架 | Unsloth | Transformers + PEFT |
| 设备支持 | 仅 NVIDIA GPU | Metal GPU (M 芯片) / CPU |
| 内存优化 | 批大小: 2 | 批大小: 1 (减少内存占用) |
| 梯度累积 | 4 步 | 8 步 (保持等效更新) |
| 优化器 | adamw_8bit | adamw_torch (macOS 兼容) |
| Attention | flash_attention_2 | eager attention (macOS 兼容) |

## 安装依赖

### 1. 创建虚拟环境（如果还未创建）
```bash
cd /private/var/www/github/Arthur-AiEra/LLM-Lab/B26_LLM-Distill-Finetune-Lab/refactored
python3 -m venv .venv
source .venv/bin/activate
```

### 2. 安装 macOS 依赖
```bash
pip install -r requirements_macos.txt
```

## 运行脚本

### 方案 A: 使用本地模型（推荐，快速）
1. 编辑 `qwen_vl_car_insurance_train_macOS.py` 第 61 行，设置本地模型路径：
```python
model_path = "/your/local/path/to/Qwen2.5-VL-3B-Instruct"
```

2. 运行脚本：
```bash
python3 qwen_vl_car_insurance_train_macOS.py
```

### 方案 B: 从 Hugging Face 下载（自动）
脚本会自动尝试从 Hugging Face 下载（如果本地路径不存在）：
```bash
python3 qwen_vl_car_insurance_train_macOS.py
```

## 数据准备

### 步骤 1: 准备 Excel 文件
创建 `qwen-vl-train.xlsx`，包含以下列：
- `image`: 图片文件路径
- `prompt`: 输入提示
- `response`: 预期输出

示例：
```
image                                  | prompt                  | response
images/1-vehicle-odometer-reading.jpg | 识别里程表数字          | 里程表显示: 12345km
```

### 步骤 2: 检查测试图片
确保存在 `images/1-vehicle-odometer-reading.jpg` 用于测试

### 步骤 3: 运行脚本
脚本会自动：
- 加载 Excel 数据
- 处理图片
- 进行微调
- 测试推理
- 保存模型到 `car_insurance_lora_model_macos/`

## 性能预期

| 硬件配置 | 训练时间 | 内存占用 |
|---------|---------|---------|
| M1/M2/M3 (Metal GPU) | 中等 (~1-2h) | ~8-12GB RAM |
| Intel Mac (CPU) | 慢 (~4-6h) | ~6-8GB RAM |

## 常见问题

### Q: 如何检查是否使用了 Metal GPU？
**A**: 运行脚本时，输出的第一行会显示：
- ✓ `Using Metal GPU (MPS) acceleration` → 已启用
- ⚠ `Using CPU(training will be slow)` → 未检测到，使用 CPU

### Q: 脚本显示 "MPS is not available"？
**A**: 
1. 确保 PyTorch 版本支持 M 芯片：
```bash
python3 -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"
```

2. 更新PyTorch：
```bash
pip install --upgrade torch torchvision
```

### Q: 内存不足？
**A**: 在脚本中调整以下参数：
```python
# 第 310 行
per_device_train_batch_size=1,  # 改为 1（已是最小）
gradient_accumulation_steps=8,  # 改为 4（减少累积）
max_steps=30,                   # 改为 10（减少步数）
```

### Q: 如何继续使用原始脚本？
**A**: 如果您有 NVIDIA/AMD/Intel GPU 的设备，继续使用原始版本：
```bash
python3 qwen_vl_car_insurance_train.py
```

## 模型输出位置

训练完成后，微调后的模型保存在：
- `car_insurance_lora_model_macos/adapter_model.safetensors`
- `car_insurance_lora_model_macos/adapter_config.json`

## 下一步

### 1. 使用微调模型
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
adapter_id = "car_insurance_lora_model_macos"

model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=False, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_id)
```

### 2. 合并适配器（可选）
```python
merged_model = model.merge_and_unload()
merged_model.save_pretrained("car_insurance_merged_model")
```

## 技术细节

### 为什么使用 PEFT 而不是 Unsloth？
- ✅ 完全兼容 macOS/CPU
- ✅ 支持多种硬件
- ✅ 更好的社区支持和文档
- ✅ 与主流框架集成紧密

### 关键代码改动
1. **设备检测**：自动识别最可用的设备（MPS > CUDA > CPU）
2. **数据加载**：支持 Vision Transformer 的多模态输入
3. **优化器**：使用 `adamw_torch` 替代 `adamw_8bit`
4. **Attention**：使用 eager attention 而非 flash attention

## 联系与反馈

如有问题，请检查：
1. Python 版本 ≥ 3.8
2. PyTorch ≥ 2.0
3. 磁盘空间充足（模型 ~7GB）
4. 网络连接正常（用于下载模型）

---
**Happy Training on macOS! 🎉**
