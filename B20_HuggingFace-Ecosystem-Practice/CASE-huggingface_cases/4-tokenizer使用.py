#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
# 必须在导入 pipeline 之前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/models"  # 同时控制 model 和 tokenizer 缓存

from transformers import AutoTokenizer

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# 模拟一个 Batch（两个长度不一样的句子）
sentences = ["我爱AI", "HuggingFace真好用"]

# 调用分词器
inputs = tokenizer(
    sentences,
    padding=True,      # 自动填充到最长句子的长度
    truncation=True,   # 超过最大长度就截断
    max_length=10,     # 设置最大长度
    return_tensors="pt" # 返回 PyTorch 张量
)

print(inputs)
# 重点观察：
# 'input_ids': 你的字对应的数字
# 'attention_mask': 1代表是真实的字，0代表是填充的(Padding)

