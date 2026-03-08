# -*- coding: utf-8 -*-
"""
Qwen3.5-0.8B 中文医疗模型SFT微调 (CPU版)
课程: 高质量微调数据工程与评估
环境: 无需GPU，约需3.2GB内存，训练10步约需1-3分钟
依赖: pip install transformers peft trl datasets pandas
"""

import os
import json
import glob
import torch
import pandas as pd
from datasets import Dataset
import logging

# 简单配置：设置根记录器的级别为 DEBUG
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

#t 108:55 LoRA（Low-Rank Adaptation，低秩微调） https://gemini.google.com/app/dc27954cd53de868

# ========================================
# 配置
# ========================================

MEDICAL_PROMPT = """你是一个专业的医疗助手。请根据患者的问题提供专业、准确的回答。

### 问题：
{}

### 回答：
{}"""

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "【数据集】中文医疗数据")
_PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data")
_LOCAL_MODEL_DIR = "/private/var/ifc/app_data/autodl-tmp/models/Qwen/Qwen3___5-0___8B"
if os.path.exists(_LOCAL_MODEL_DIR):
    MODEL_NAME = _LOCAL_MODEL_DIR
else:
    MODEL_NAME = "Qwen/Qwen3.5-0.8B"
logging.info(f"MODEL_NAME: {MODEL_NAME}")

def read_csv_with_encoding(file_path):
    """尝试使用不同编码读取CSV，gb18030是GBK超集优先尝试"""
    import io
    for encoding in ['gb18030', 'utf-8', 'gbk', 'gb2312']:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    with open(file_path, 'r', encoding='gb18030', errors='replace') as f:
        return pd.read_csv(io.StringIO(f.read()))


def load_from_jsonl(jsonl_path):
    """从JSONL文件加载数据"""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_from_csv(data_dir, max_samples=2000):
    """从原始CSV文件加载数据"""
    data = []
    departments = {
        'IM_内科': '内科',
        'Surgical_外科': '外科',
        'Pediatric_儿科': '儿科',
        'Oncology_肿瘤科': '肿瘤科',
        'OAGD_妇产科': '妇产科',
        'Andriatria_男科': '男科'
    }
    for dept_dir, dept_name in departments.items():
        dept_path = os.path.join(data_dir, dept_dir)
        if not os.path.exists(dept_path):
            continue
        csv_files = [f for f in os.listdir(dept_path) if f.endswith('.csv')]
        for csv_file in csv_files:
            file_path = os.path.join(dept_path, csv_file)
            try:
                df = read_csv_with_encoding(file_path)
                for _, row in df.iterrows():
                    question = str(row.get('question', row.get('ask', ''))).strip()
                    answer = str(row.get('answer', row.get('response', ''))).strip()
                    if not question or not answer or question == 'nan' or answer == 'nan':
                        continue
                    if len(question) < 5 or len(answer) < 10:
                        continue
                    data.append({"input": question, "output": answer})
                    if len(data) >= max_samples:
                        break
            except Exception as e:
                continue
            if len(data) >= max_samples:
                break
        if len(data) >= max_samples:
            break
    return data


def prepare_training_data():
    """准备训练数据，优先检查processed_data/*.jsonl，否则回退到CSV"""
    candidates = [
        os.path.join(_PROCESSED_DIR, "medical_alpaca_train.jsonl"),
        os.path.join(_PROCESSED_DIR, "medical_alpaca_sampled.jsonl"),
        os.path.join(_PROCESSED_DIR, "medical_alpaca_full.jsonl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return load_from_jsonl(path)
    for path in sorted(glob.glob(os.path.join(_PROCESSED_DIR, "*.jsonl"))):
        return load_from_jsonl(path)
    return load_from_csv(_DATA_DIR, max_samples=2000)


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer

    # 1. 加载模型
    print("CPU模式: float32加载 (约需3.2GB内存)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained( # 基础模型
        MODEL_NAME,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 配置LoRA
    lora_config = LoraConfig( # ΔW=BA
        r=8, # 设置秩 r=8， r=8 意味着这两个小矩阵的中间维度只有 8
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # 指定了 LoRA 只更新注意力机制相关的权重参数 ：Query, Key, Value, Output 投影层
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    # 应用配置， PEFT 高效微调
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. 准备数据并格式化为带EOS的文本
    data_list = prepare_training_data()
    eos_token = tokenizer.eos_token or ""

    def formatting_func(examples):
        texts = []
        for inp, out in zip(examples["input"], examples["output"]):
            text = MEDICAL_PROMPT.format(inp, out) + eos_token
            texts.append(text)
        return {"text": texts}

    dataset = Dataset.from_list(data_list)
    dataset = dataset.map(formatting_func, batched=True)
    print(f"训练数据: {len(dataset)} 条")

    # 4. SFT训练
    train_kwargs = dict(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2, # 梯度累积步数2
        max_steps=10, # 最大训练步数
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        # 优化器： 指导模型如何利用刚才算出来的错误（梯度）去更新参数
        # 它像一个智能指南针，不仅知道该往哪个方向走，还能根据路况（不同参数的特性）自动为你调节步伐大小，防止模型在学习时“走火入魔”（过拟合）
        optim="adamw_torch",
        use_cpu=True,
        output_dir="outputs_medical",
        report_to="none",
        logging_steps=1,
    )

    from trl import SFTConfig

    sft_config = SFTConfig(
        **train_kwargs,
        max_length=512,  # <--- 修改处：从 max_seq_length 改为 max_length
        dataset_text_field="text",
        dataset_num_proc=1,
        packing=False,
    )
    trainer = SFTTrainer( # SFTTrainer（监督微调训练器）是“老师”
        model=model, # 基础模型是“学生”
        processing_class=tokenizer,  # 传译员: 负责把课本翻译成数字，把学生的回答翻译成人类语言
        train_dataset=dataset, # 数据集是“课本”
        args=sft_config, # 训练参数，就是老师制定的“教学计划”
    )
    # 学习方法 (LoRA)：老师允许学生不用把整本百科全书重抄一遍，只用“贴便利贴”的方式（低秩矩阵）快速记住核心的医学知识
    trainer.train() # 启动训练

    # 5. 测试推理
    test_questions = [
        "我最近总是感觉头晕，应该怎么办？",
        "感冒发烧应该吃什么药？",
        "高血压患者需要注意什么？",
    ]
    # 训练完成后，将模型设为 eval() 模式 => 模型内部层（如 Dropout）的物理行为，确保计算逻辑是针对测试环境的
    # 1. 关闭 Dropout（随机失活）机制; 2. 冻结 Normalization（归一化）层的统计数据
    model.eval()
    for q in test_questions:
        prompt = MEDICAL_PROMPT.format(q, "")
        inputs = tokenizer(prompt, return_tensors="pt") # 把患者的问题（prompt）扔给分词器，并且要求它直接输出 PyTorch 张量（return_tensors="pt"）
        with torch.no_grad(): # 改变的是底层引擎的记录行为。它告诉 PyTorch：“现在是测试，不需要通过反向传播更新权重了，请停止记录梯度。” 这样做可以极大地节省显存（或内存）并成倍提升生成速度。
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "### 回答：" in response:
            answer = response.split("### 回答：")[-1].strip()
        elif "### 回答:" in response:
            answer = response.split("### 回答:")[-1].strip()
        else:
            answer = response
        print(f"Q: {q}")
        print(f"A: {answer[:200]}")
        print()

    # 6. 保存LoRA模型
    save_dir = "lora_model_medical"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"LoRA模型已保存: {save_dir}")
