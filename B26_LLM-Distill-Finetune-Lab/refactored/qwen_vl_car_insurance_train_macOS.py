# -*- coding: utf-8 -*-
"""
Qwen2-VL 3B 视觉模型微调 - 汽车保险承保专家 (macOS 兼容版本)
课程：LLM模型蒸馏与微调实操
功能：使用 Transformers + PEFT 框架对 Qwen2.5-VL-3B 进行车辆里程表识别任务的微调
环境：macOS with Metal GPU 加速（或 CPU）
改动：移除 Unsloth 依赖，添加 Metal GPU 支持

重要提示：
- Qwen2.5-VL 是多模态模型，需要 torch 和 transformers 的新版本
- 某些操作在 MPS 上可能不完全支持，脚本包含自动回退到 CPU 的机制
- 首次运行会下载较大的模型文件（~7GB），需要足够的磁盘空间和网络连接
"""

import json
import os
import torch
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration

# ========================================
# Step 0: 设备检测
# ========================================

if torch.backends.mps.is_available():
    device = "mps"
    print("✓ Using Metal GPU (MPS) acceleration")
    dtype = torch.float16
elif torch.cuda.is_available():
    device = "cuda"
    print("✓ Using CUDA GPU")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
else:
    device = "cpu"
    print("⚠ Using CPU (training will be slow)")
    dtype = torch.float32

print(f"Device: {device}, Dtype: {dtype}")

# ========================================
# Step 1: 模型加载
# ========================================

from transformers import AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model

print("正在加载Qwen2.5-VL-3B模型...")

model_path = "/private/var/ifc/app_data/autodl-tmp/models/Qwen/Qwen2___5-VL-3B-Instruct"

try:
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    # 使用 AutoModel 而不是 AutoModelForCausalLM，因为这是多模态视觉模型
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,  # 使用 dtype 而不是 torch_dtype
        torch_dtype=torch.float16,
        device_map="mps",
        attn_implementation="eager",  # 对macOS使用eager attention
    )
    # 将模型移到指定设备
    model = model.to(device)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load model from {model_path}: {e}")
    print("Trying to download from Hugging Face...")
    try:
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            trust_remote_code=True,
            dtype=dtype,
            device_map=None,
            attn_implementation="eager",
        )
        model = model.to(device)
        print("✓ Model downloaded and loaded successfully")
    except Exception as e2:
        print(f"✗ Failed to load model: {e2}")
        exit(1)

print("配置模型微调参数...")
# 对于Qwen2.5-VL，启用梯度检查点以节省内存
try:
    model.gradient_checkpointing_enable()
    print("✓ 已启用梯度检查点")
except Exception as e:
    print(f"⚠ 无法启用梯度检查点: {e}")

# 计算可训练参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数数: {total_params:,}")
print(f"可训练参数数: {trainable_params:,}")
print(f"参数比例: {100 * trainable_params / total_params:.2f}%")

# 确保模型处于评估模式以进行推理测试
model.eval()


# ========================================
# Step 2: 数据准备（Excel格式）
# ========================================

import pandas as pd

print("加载训练数据...")

def process_vision_info(messages):
    """处理消息中的视觉信息"""
    image_inputs = []
    video_inputs = []
    
    for message in messages:
        if isinstance(message.get("content"), list):
            for content in message["content"]:
                if content.get("type") == "image":
                    image_inputs.append(content.get("image"))
                elif content.get("type") == "video":
                    video_inputs.append(content.get("video"))
    
    return image_inputs, video_inputs


def load_excel_dataset(file_path):
    """加载Excel格式的数据集"""
    try:
        df = pd.read_excel(file_path)
        print(f"Excel文件列名: {list(df.columns)}")
        print(f"数据集形状: {df.shape}")
        return df
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        return None


def convert_excel_to_training_format(df):
    """将Excel格式转换为训练格式"""
    converted_data = []
    
    for idx, row in df.iterrows():
        image_path = row["image"]
        prompt = row["prompt"]
        response = row["response"]
        
        if pd.notna(image_path) and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert('RGB')
                conversation = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image", "image": image}
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": response}
                            ]
                        }
                    ]
                }
                converted_data.append(conversation)
                print(f"成功处理样本 {idx + 1}: {image_path}")
            except Exception as e:
                print(f"处理图片 {image_path} 时出错: {e}")
        else:
            print(f"警告：图片文件不存在或路径为空 {image_path}")
    
    return converted_data


# 尝试加载数据集
if os.path.exists("qwen-vl-train.xlsx"):
    train_df = load_excel_dataset("qwen-vl-train.xlsx")
    if train_df is not None:
        converted_dataset = convert_excel_to_training_format(train_df)
    else:
        print("无法加载数据集，请确保 qwen-vl-train.xlsx 存在")
        exit(1)
    print(f"成功加载 {len(converted_dataset)} 个训练样本")
else:
    print("警告：qwen-vl-train.xlsx 不存在，创建示例数据集...")
    # 创建示例数据集用于演示
    converted_dataset = []
    if os.path.exists("images/1-vehicle-odometer-reading.jpg"):
        for i in range(2):
            image = Image.open("images/1-vehicle-odometer-reading.jpg").convert('RGB')
            conversation = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "你是一名汽车保险承保专家。这里有一张车辆里程表的图片。请从中提取关键信息。"},
                            {"type": "image", "image": image}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "这张图片显示车辆里程表读数为示例数据。"}
                        ]
                    }
                ]
            }
            converted_dataset.append(conversation)
    print(f"已创建 {len(converted_dataset)} 个示例训练样本")


# ========================================
# Step 3: 训练前推理测试
# ========================================

print("\n训练前模型推理测试...")

if os.path.exists("images/1-vehicle-odometer-reading.jpg"):
    try:
        model.eval()
        with torch.no_grad():
            test_image = Image.open("images/1-vehicle-odometer-reading.jpg").convert('RGB')
            test_instruction = "你是一名汽车保险承保专家。这里有一张车辆里程表的图片。请从中提取关键信息。"

            # 🔴 关键修复：构建 messages 格式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"}, # 告诉模板这里有一张图
                        {"type": "text", "text": test_instruction}
                    ]
                }
            ]
            # 🔴 使用 apply_chat_template 自动插入图像占位符
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # 使用 processor 处理
            inputs = processor(
                text=[text], # 注意这里要传列表
                images=[test_image],
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            from transformers import TextStreamer
            text_streamer = TextStreamer(processor.tokenizer, skip_prompt=True)
            print("训练前模型输出:")
            _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=64,
                              use_cache=True, temperature=1.0)
    except Exception as e:
        print(f"⚠ 推理测试失败: {e}")
        print("继续进行训练...")
else:
    print("⚠ 测试图片不存在，跳过推理测试")


# ========================================
# Step 4: 数据处理函数
# ========================================

def collate_fn(batch):
    """自定义数据整理函数 - 严格遵循 Qwen-VL 的 messages 模板"""
    texts = []
    images = []

    for item in batch:
        messages = item.get("messages", [])

        clean_messages = []
        for msg in messages:
            clean_content = []
            for content_item in msg["content"]:
                if content_item["type"] == "image":
                    images.append(content_item["image"])  # 提取 PIL Image 对象
                    clean_content.append({"type": "image"})  # 仅保留类型声明供模板使用
                else:
                    clean_content.append(content_item)
            clean_messages.append({"role": msg["role"], "content": clean_content})

        # 转化为带 <|image_pad|> 占位符的字符串
        text = processor.apply_chat_template(clean_messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    # 统一交给 processor 处理
    processed = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        max_length=2048,
        return_tensors="pt"
    )

    # 🔴 核心：生成正确的 labels，让模型知道正确答案是什么
    labels = processed["input_ids"].clone()

    # 将 padding 设为 -100 (不计算 loss)
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Qwen-VL: 忽略图像占位符的 loss
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if image_token_id is not None:
        labels[labels == image_token_id] = -100

    processed["labels"] = labels
    return processed


# ========================================
# Step 5: 模型训练
# ========================================

print("\n开始训练模型...")
from transformers import Trainer, TrainingArguments

# 为macOS调整训练参数
training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=1,  # macOS上减小批大小
    gradient_accumulation_steps=8,  # 增加梯度累积步数
    warmup_steps=2,
    num_train_epochs=1,  # 首先尝试1个epoch
    learning_rate=1e-4,  # 降低学习率以适应直接微调
    logging_steps=1,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    optim="adamw_torch",  # macOS上使用adamw_torch而不是adamw_8bit
    seed=3407,
    report_to="none",
    remove_unused_columns=False,
    save_strategy="epoch",
    save_total_limit=2,
    dataloader_pin_memory=False,  # macOS上禁用pin_memory
    use_mps_device=(device == "mps"),  # 明确指定使用MPS设备
    bf16=False,  # macOS MPS 不支持bfloat16
    fp16=False,  # 禁用fp16以避免MPS上的gradient scaler问题
)

trainer = Trainer(
    model=model,
    tokenizer=processor.tokenizer,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=converted_dataset,
)

# 显存/内存信息
if device == "cuda":
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. 最大显存 = {max_memory} GB.")
    print(f"{start_gpu_memory} GB 显存已使用.")
else:
    print(f"使用设备: {device}")
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        print("Metal GPU (MPS) 已启用")

# 确保模型处于训练模式
model.train()

# 执行训练
print("开始训练（这可能需要一段时间）...")
try:
    trainer_stats = trainer.train()
    print("✓ 训练完成")
except RuntimeError as e:
    if "MPS" in str(e):
        print(f"⚠ MPS 兼容性问题: {e}")
        print("尝试切换到 CPU 继续训练...")
        model.to("cpu")
        device = "cpu"
        trainer = Trainer(
            model=model,
            tokenizer=processor.tokenizer,
            args=training_args,
            data_collator=collate_fn,
            train_dataset=converted_dataset,
        )
        trainer_stats = trainer.train()
    else:
        raise

# 训练统计
if device == "cuda":
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    print(f"峰值显存使用: {used_memory} GB")
    print(f"LoRA训练显存使用: {used_memory_for_lora} GB")
    print(f"显存使用率: {used_percentage}%")

print(f"训练用时: {trainer_stats.metrics.get('train_runtime', 0)} 秒")
print(f"训练用时: {round(trainer_stats.metrics.get('train_runtime', 0)/60, 2)} 分钟")

# ========================================
# Step 6: 训练后推理测试
# ========================================

print("\n训练后模型推理测试...")

if os.path.exists("images/1-vehicle-odometer-reading.jpg"):
    try:
        model.eval()
        with torch.no_grad():
            test_image = Image.open("images/1-vehicle-odometer-reading.jpg").convert('RGB')
            test_instruction = "从这张车辆里程表图片中提取信息。"

            # 🔴 构建 messages 格式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": test_instruction}
                    ]
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = processor(
                text=[text],
                images=[test_image],
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            from transformers import TextStreamer

            text_streamer = TextStreamer(processor.tokenizer, skip_prompt=True)
            print("训练后模型输出:")
            _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128,
                               use_cache=True, temperature=1.0)
    except Exception as e:
        print(f"⚠ 推理测试失败: {e}")
else:
    print("⚠ 跳过训练后推理测试")


# ========================================
# Step 7: 保存模型
# ========================================

print("\n保存模型和处理器...")
model.save_pretrained("car_insurance_lora_model_macos")
processor.save_pretrained("car_insurance_lora_model_macos")
print("✓ 训练完成! 模型已保存到 car_insurance_lora_model_macos 目录")
