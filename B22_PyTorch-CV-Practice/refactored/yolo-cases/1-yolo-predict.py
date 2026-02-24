#!/usr/bin/env python
# coding: utf-8

#t 151:05

# In[ ]:


## 使用yolo12进行预测


# In[1]:


from ultralytics import YOLO

# 如果本地的 yolo12n.pt 权重不兼容，如删除后运行以下命令重新下载
# yolo predict model=yolo12n.pt source=000000000139.jpg
model = YOLO('yolo12n.pt')
# 使用model进行目标检测
results = model("./000000000139.jpg")
results[0].show()


# In[2]:


# 在验证集上评估
# 只是为了练习和测试 YOLOv12 的代码流程，完全没有必要在本地下载几十 GB 的完整 COCO 数据集。Ultralytics 官方提供了一个极其轻量的测试数据集 coco8.yaml
metrics = model.val(data='coco8.yaml', save_json=True)
print(metrics.box.map)  # 打印mAP指标

