#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
# 必须在导入 pipeline 之前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/models"  # 同时控制 model 和 tokenizer 缓存

from transformers import pipeline

# 加载零样本分类器 (推荐使用支持多语言的模型)
classifier = pipeline(
    task="zero-shot-classification",
    model="facebook/bart-large-mnli" # 这个模型虽然是英文的，但对简单中文也能理解，或者换用 multilingual 模型
)

text = "特斯拉发布了最新的自动驾驶技术，股价大涨。"
candidate_labels = ["体育", "财经", "娱乐", "科技"]

# 让模型自己去匹配
result = classifier(text, candidate_labels)
print(f"文本内容：{text}")
print(f"预测标签：{result['labels'][0]} (置信度: {result['scores'][0]:.4f})")
# 输出示例：科技 (0.85) 或 财经

