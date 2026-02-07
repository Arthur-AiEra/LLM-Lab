#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
# 必须在导入 pipeline 之前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/models"  # 同时控制 model 和 tokenizer 缓存

from transformers import AutoModel, AutoModelForSequenceClassification

# 1. 纯净版模型
base_model = AutoModel.from_pretrained("bert-base-chinese")
base_model
# 输出形状：(Batch_Size, Sequence_Length, Hidden_Size) -> (2, 10, 768)


# In[2]:


# 2. 带分类头的模型
cls_model = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
# 输出形状：(Batch_Size, Num_Labels) -> (2, 2)
cls_model

