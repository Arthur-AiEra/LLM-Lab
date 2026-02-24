#!/usr/bin/env python
# coding: utf-8

#t 152:48

# In[1]:


from ultralytics import YOLO

# 从 YAML 配置文件初始化 YOLO12n 模型
# model = YOLO("yolov12.yaml")

# 如果有预训练模型，可以使用下面的方式加载
model = YOLO("yolo12n.pt")

print("############ 显示模型信息 BEGIN ############")
# 显示模型信息
model.info()
print("############ 显示模型信息 END ############")

# In[2]:

print("############ 模型训练 ############")
# 模型训练
results = model.train(
  data='coco8.yaml',
  epochs=1,           # 训练轮数
  batch=8,         # 批次大小
  imgsz=640,         # 输入图像尺寸
  scale=0.5,         # 图像缩放比例 (S:0.9; M:0.9; L:0.9; X:0.9)

  # 数据增强参数
  mosaic=1.0,        # 马赛克数据增强概率
  mixup=0.0,         # 混合数据增强概率 (S:0.05; M:0.15; L:0.15; X:0.2)
  copy_paste=0.1,    # 复制粘贴增强概率 (S:0.15; M:0.4; L:0.5; X:0.6)
  device="cpu",        # 使用的 GPU 设备号
)

print("############ 使用模型进行目标检测 ############")
# 使用模型进行目标检测
results = model("./000000000139.jpg")
results[0].show()  # 显示检测结果


