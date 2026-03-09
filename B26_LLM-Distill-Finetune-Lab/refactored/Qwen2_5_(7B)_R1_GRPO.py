# -*- coding: utf-8 -*-
"""
Qwen2.5-7B GRPO强化学习训练R1模型 - macOS 兼容版本
课程：LLM模型蒸馏与微调实操
功能：使用GRPO（Group Relative Policy Optimization）训练Qwen2.5-7B的推理能力
环境：macOS with Metal GPU 加速（或 CPU）
依赖：pip install transformers peft trl datasets
"""

#t 107:32 https://gemini.google.com/app/b0c8e221ea7ba4ac

# ========================================
# Step 1: 模型加载
# ========================================

import torch
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

max_seq_length = 1024  # 可以增加以获得更长的推理轨迹
lora_rank = 32  # 更大的rank让模型更智能，但训练更慢

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

# 模型路径
model_path = "/private/var/ifc/app_data/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct"

try:
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if device != "cpu" else None,
        trust_remote_code=True,
    )
    
    if device == "cpu":
        model = model.to(device)
    
    print("✓ Model loaded successfully")
    
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    exit(1)

# ========================================
# Step 2: LoRA配置
# ========================================

# LoRA配置
lora_config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 应用LoRA
model = get_peft_model(model, lora_config)
print("✓ LoRA applied to model")

# 添加TRL需要的属性
if not hasattr(model, 'warnings_issued'):
    model.warnings_issued = {}


# ========================================
# Step 3: GSM8K数据准备
# ========================================

import re
from datasets import load_dataset, Dataset

# 系统提示词：定义推理输出格式
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def extract_xml_answer(text: str) -> str:
    """从XML格式文本中提取答案"""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    """从####标记文本中提取答案"""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def get_gsm8k_questions(split="train") -> Dataset:
    """加载GSM8K数据集"""
    data = load_dataset('/private/var/www/github/Arthur-AiEra/LLM-Lab/B26_LLM-Distill-Finetune-Lab/refactored/【数据集】gsm8k', 'main')[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    return data


dataset = get_gsm8k_questions()


# ========================================
# Step 4: 奖励函数设计
# ========================================

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """正确性奖励：检查答案是否正确（权重最高）"""
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-' * 20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}",
          f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    """整数奖励：检查答案是否为整数"""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """严格格式奖励：完全符合XML格式"""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """宽松格式奖励：基本符合XML格式"""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    """计算XML标签完整性得分"""
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """XML标签计数奖励"""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


# ========================================
# Step 5: GRPOTrainer训练
# ========================================

max_prompt_length = 256

from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    learning_rate=5e-6, # 比SFT低一个数量级，强化学习需稳定
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_torch",  # 使用标准AdamW
    logging_steps=1,
    per_device_train_batch_size=2,  # 增加batch size以满足generation_batch_size要求
    gradient_accumulation_steps=1,
    num_generations=2,  # 每个问题生成2个候选答案（GRPO需要至少2个）
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    max_steps=10,
    save_steps=10,
    max_grad_norm=0.1,
    report_to="none",
    output_dir="outputs",
    use_vllm=False,  # 禁用vLLM以避免兼容性问题
    # MPS兼容性设置
    dataloader_pin_memory=False,  # MPS不支持pin_memory
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)

# 开始训练
trainer.train()


# ========================================
# Step 6: 模型测试与保存
# ========================================

# 保存LoRA参数
model.save_pretrained("grpo_saved_lora")
tokenizer.save_pretrained("grpo_saved_lora")
print("✓ LoRA adapters saved to 'grpo_saved_lora'")

# 测试模型推理
# 1. 关闭梯度检查点（把模型从“省显存训练状态”解放出来）
model.gradient_checkpointing_disable()

# 2. 将模型切换到评估/推理模式（关闭 Dropout 等训练专属层）
model.eval()

text = tokenizer.apply_chat_template([
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Calculate pi."},
], tokenize=False, add_generation_prompt=True)

inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.8,
        top_p=0.95,
        do_sample=True,
        use_cache = True, # 在调用 model.generate() 生成文本时，大模型极其依赖 KV Cache（键值缓存）  来记住它上一秒刚说过的话
        pad_token_id=tokenizer.eos_token_id,
    )

generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print("Generated response(Calculate pi):")
print(generated_text)


# ========================================
# 模型导出选项（按需取消注释）
# ========================================

# 保存为16bit浮点合并模型
# from peft import merge_and_unload
# merged_model = merge_and_unload(model, model.base_model)
# merged_model.save_pretrained("model_merged", safe_serialization=True)
# tokenizer.save_pretrained("model_merged")

# 保存为4bit量化模型（需要bitsandbytes）
# model.save_pretrained("model_4bit", safe_serialization=True)
# tokenizer.save_pretrained("model_4bit")

# 仅保存LoRA适配器
# model.save_pretrained_merged("model", tokenizer, save_method="lora")

# 保存为GGUF q4_k_m格式
# model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
