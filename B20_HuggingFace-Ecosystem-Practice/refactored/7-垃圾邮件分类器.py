#!/usr/bin/env python
# coding: utf-8

# ## Step1，准备环境与数据

# In[1]:


import os
# 必须在导入 pipeline 之前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/private/var/ifc/app_data/autodl-tmp/models"  # 同时控制 model 和 tokenizer 缓存

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# 1. 模拟一个私有数据集 (真实场景用 load_dataset 加载 CSV)
data = [
    {"text": "今晚有空一起吃饭吗？", "label": 0},  # 正常
    {"text": "恭喜您获得500万大奖，点击领取", "label": 1}, # 垃圾
    {"text": "您的验证码是1234，请勿泄露", "label": 0},
    {"text": "澳门首家线上赌场上线啦", "label": 1},
    {"text": "项目进度怎么样了？需不需要开会", "label": 0},
    {"text": "独家内幕消息，股票必涨，加群", "label": 1},
]
# 转为 HF Dataset 对象
dataset = Dataset.from_list(data)

# 2. 划分训练集和测试集
dataset = dataset.train_test_split(test_size=0.2)


# ## Step2：数据预处理 (Map & Tokenize)

# In[2]:


checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def preprocess_function(examples):
    # Truncation=True: 截断过长的
    # Padding=False: 这里先不补齐！留给 DataCollator 动态补齐
    return tokenizer(examples["text"], truncation=True, max_length=128)

# 批量处理
tokenized_datasets = dataset.map(preprocess_function, batched=True)


# ## Step3：DataCollator 与 模型加载

# In[3]:


from transformers import DataCollatorWithPadding

# 动态补齐工具
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 加载带分类头的模型 (num_labels=2: 二分类)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)


# ## Step4：配置参数并开始训练

# In[5]:


import numpy as np
import evaluate  # pip install evaluate

# 加载评估指标
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    """计算评估指标"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="/private/var/ifc/app_data/autodl-tmp/models/spam-bert-finetuned", # 模型保存路径
    eval_strategy="epoch",               # 每个 epoch 结束后评估一次
    save_strategy="epoch",               # 每个 epoch 结束后保存一次
    learning_rate=2e-5,                  # 学习率 (微调通常很小)
    per_device_train_batch_size=4,      # 批次大小 (显存小就调小)
    num_train_epochs=3,                  # 训练轮数
    weight_decay=0.01,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    # tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 开始训练！
trainer.train()


# ## Step5：模型推理 (验证效果)

# In[8]:


# 模拟一条新数据
text = "澳门线上赌场，特价酬宾"
inputs = tokenizer(text, return_tensors="pt").to(model.device) # 确保数据也在 GPU 上

with torch.no_grad():
    logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()

print(f"输入文本: {text}")
print(f"预测类别: {'垃圾邮件' if predicted_class_id == 1 else '正常邮件'}")


# ## Step6：模型推理 (验证效果)

# In[9]:


# 模拟一条新数据
text = "澳门线上赌场，特价酬宾"
inputs = tokenizer(text, return_tensors="pt").to(model.device) # 确保数据也在 GPU 上

with torch.no_grad():
    logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()

print(f"输入文本: {text}")
print(f"预测类别: {'垃圾邮件' if predicted_class_id == 1 else '正常邮件'}")

