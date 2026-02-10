#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
# 必须在导入 pipeline 之前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/private/var/ifc/app_data/autodl-tmp/models"  # 同时控制 model 和 tokenizer 缓存

from transformers import pipeline

# 1. 加载 pipeline
# 指定 task="sentiment-analysis"
# 💡 技巧：如果不指定 model，默认下载英文模型。
# 这里我们指定一个中文微调过的模型，效果更好。
cache_dir = '/private/var/ifc/app_data/autodl-tmp/models'
classifier = pipeline(
    task="sentiment-analysis",
    model="uer/roberta-base-finetuned-dianping-chinese"
    #model="bert-base-chinese"
)

# 2. 预测
result = classifier("这个手机屏幕太烂了，反应很慢！")
print(result)
# 输出示例：[{'label': 'negative (negative)', 'score': 0.98}]

result2 = classifier("物流很快，包装很精美，五星好评。")
print(result2)
# 输出示例：[{'label': 'positive (positive)', 'score': 0.99}]

