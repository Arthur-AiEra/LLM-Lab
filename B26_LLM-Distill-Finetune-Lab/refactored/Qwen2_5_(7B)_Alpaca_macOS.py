# -*- coding: utf-8 -*-
"""
Qwen2.5-7B SFT微调（Alpaca数据集）- macOS 兼容版本
课程：LLM模型蒸馏与微调实操
功能：使用标准 Transformers + PEFT 框架对 Qwen2.5-7B 进行 Alpaca 格式的监督微调
环境：macOS with Metal GPU 加速（或 CPU）
"""

#t 81:20 https://gemini.google.com/app/b0c8e221ea7ba4ac

import torch
import warnings
warnings.filterwarnings("ignore")

# ========================================
# Step 1: 模型加载
# ========================================

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

max_seq_length = 1024  # 最大序列长度（减小以节省内存）

# 检测设备
if torch.backends.mps.is_available():
    device = "mps"
    print("✓ Using Metal GPU (MPS) acceleration")
    dtype = torch.float16
elif torch.cuda.is_available():
    device = "cuda"
    print("✓ Using CUDA GPU")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
else:
    device = "cpu"
    print("⚠ Using CPU (training will be slow)")
    dtype = torch.float32

print(f"Device: {device}, Dtype: {dtype}")

# 模型路径（请根据实际路径修改）
model_path = "/private/var/ifc/app_data/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct"

# 如果本地模型路径不存在，可以使用HuggingFace上的模型
# 例如: model_path = "Qwen/Qwen2.5-7B-Instruct"

try:
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,  # 不使用自动device_map，手动处理设备
    )
    # 将模型移到指定设备
    model = model.to(device)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    print("Trying to download from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,
    )
    model = model.to(device)

# ========================================
# Step 2: LoRA 适配器配置
# ========================================

peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,  # LoRA秩
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    modules_to_save=None,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ========================================
# Step 3: Alpaca 数据集准备
# ========================================

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token if tokenizer.eos_token else "</s>"
# 设置pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

def tokenize_function(examples):
    """对文本进行标记化和准备标签"""
    texts = examples["text"]
    
    # 标记化
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
        return_tensors=None,
    )
    
    # 将input_ids作为labels（因果语言模型）
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

from datasets import load_dataset

# 本地数据集路径
dataset_path = "【数据集】alpaca-cleaned"

try:
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset('json', data_files=f'{dataset_path}/alpaca_data_cleaned_compact.json', split='train')
    print("✓ Dataset loaded successfully")
except Exception as e:
    print(f"✗ Failed to load local dataset: {e}")
    print("Trying to download from Hugging Face...")
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")

# 数据处理 - 先格式化，再标记化
dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=2, remove_columns=["instruction", "input", "output"])
dataset = dataset.map(tokenize_function, batched=True, num_proc=2, remove_columns=["text"])

print(f"Dataset size: {len(dataset)}")
print(f"Sample keys: {list(dataset[0].keys())}")

# ========================================
# Step 4: 训练配置
# ========================================

from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

# macOS 兼容的训练参数
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # 减小批大小以适应 macOS 内存
    gradient_accumulation_steps=8,  # 增加梯度累积
    warmup_steps=2,
    num_train_epochs=1,  # 单轮训练演示
    learning_rate=2e-4,
    logging_steps=1,
    optim="adamw_torch",  # macOS 兼容的优化器
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    report_to="none",
    save_strategy="steps",
    save_steps=50,
    eval_strategy="no",
    remove_unused_columns=False,  # 保留所有列，由data_collator处理
    bf16=False,  # 禁用bfloat16（Metal GPU不支持）
    fp16=False,  # 禁用float16（Metal GPU可能有问题）
    dataloader_pin_memory=False,  # Metal GPU上禁用pin memory
)

# 简单的数据整理器
data_collator = DataCollatorWithPadding(tokenizer, padding="longest")


class MetalCompatibleTrainer(Trainer):
    """macOS Metal GPU 兼容的Trainer"""
    def _prepare_inputs(self, inputs):
        """将输入移到正确的设备"""
        inputs = super()._prepare_inputs(inputs)
        # 确保张量在正确的设备上
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(device)
        return inputs

# ========================================
# Step 5: 训练
# ========================================

trainer = MetalCompatibleTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

print("Starting training...")
import time
start_time = time.time()

trainer.train()

end_time = time.time()
training_time = end_time - start_time

print(f"\n✓ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

# ========================================
# Step 6: 推理验证
# ========================================

print("\n" + "="*50)
print("Testing inference...")
print("="*50)

model.eval()

def generate_response(instruction, input_text=""):
    prompt = alpaca_prompt.format(instruction, input_text, "")
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 移到正确的设备
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,  # 减小token长度以加快推理
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 测试推理
print("\n Test 1: Fibonacci Sequence")
print("-" * 50)
response = generate_response("Continue the fibonacci sequence.", "1, 1, 2, 3, 5, 8")
print(response)

print("\n Test 2: Paris Tower")
print("-" * 50)
response = generate_response("What is a famous tall tower in Paris?")
print(response)

# ========================================
# Step 7: 模型保存
# ========================================

print("\n" + "="*50)
print("Saving model...")
print("="*50)

output_dir = "lora_model_macos"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✓ Model saved to {output_dir}")

# 保存完整模型（可选，需要更多存储空间）
# model.merge_and_unload()
# model.save_pretrained(f"{output_dir}_merged")
# tokenizer.save_pretrained(f"{output_dir}_merged")

print("\n✓ All done!")
