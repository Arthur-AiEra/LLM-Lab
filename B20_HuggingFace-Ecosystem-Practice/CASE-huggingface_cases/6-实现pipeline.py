#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
# 必须在导入 pipeline 之前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/models"  # 同时控制 model 和 tokenizer 缓存

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. 准备工作
checkpoint = "uer/roberta-base-finetuned-dianping-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# 2. 原始文本
text = "这家餐厅太难吃了，服务员态度还差！"

# 3. Step 1: Tokenize (预处理)
inputs = tokenizer(text, return_tensors="pt")
# inputs 包含 input_ids 和 attention_mask

# 4. Step 2: Model Inference (模型推理)
# 不需要计算梯度，节省内存
with torch.no_grad():
    outputs = model(**inputs)
    # 获取 Logits (未归一化的分数)
    logits = outputs.logits
    print(f"模型原始输出 (Logits): {logits}")

# 5. Step 3: Post-processing (后处理)
# 使用 Softmax 将 logits 转换为概率
probabilities = F.softmax(logits, dim=-1)
print(f"概率分布: {probabilities}")

# 获取概率最大的标签索引
predicted_id = torch.argmax(probabilities).item()
# 使用 model.config.id2label 查表得到人类可读标签
predicted_label = model.config.id2label[predicted_id]

print(f"最终结果: {predicted_label}")

