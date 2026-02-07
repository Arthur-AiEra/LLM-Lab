#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/models"  # 同时控制 model 和 tokenizer 缓存

from transformers import pipeline

generator = pipeline(
    task="text-generation", 
    model="uer/gpt2-chinese-cluecorpussmall",
    device="cuda"
)

text = "在一个风雨交加的夜晚，程序员小李打开了电脑，突然"
result = generator(text, max_length=100, truncation=True, do_sample=True)
print(result[0]['generated_text'])

