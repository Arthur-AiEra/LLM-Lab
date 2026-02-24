import torch
import torch.nn as nn
import numpy as np
#t 99:50
# 这个代码示例展示了卷积神经网络的核心操作：如何将一个简单的图像通过自定义卷积核进行卷积运算，生成对应的特征图

image = np.array([[1,1,1,0,0],
         [0,1,1,1,0],
         [0,0,1,1,1],
         [0,0,1,1,0],
         [0,1,1,0,0]])
filter_1 = np.array([[1, 0, 1],
                    [0, 1, 0], 
                    [1, 0, 1]])
filters = np.array([filter_1])
image = image.astype('float32')
# 将NumPy数组转换为PyTorch张量
image = (torch.from_numpy(image)
         .unsqueeze(0) # 添加批次维度（batch dimension）
         .unsqueeze(1)) # 添加通道维度（channel dimension）
print(f"image.shape: {image.shape}") # 将2D图像(5×5)转换为4D张量
# 将滤波器转换为PyTorch张量，添加输入通道维度，并确保其数据类型为FloatTensor
weight = (torch.from_numpy(filters)
          .unsqueeze(1) # 添加输入通道维度
          .type(torch.FloatTensor))
print(f"weight: {weight}")
# 使用卷积对四维数据进行处理，这里只有一张图片
conv = nn.Conv2d(1,1, kernel_size=(3,3), bias=False)
conv.weight = torch.nn.Parameter(weight) # 用我们之前定义的滤波器权重替换卷积层的默认随机初始化权重
conv_output = conv(image) # 执行卷积操作：将图像输入卷积层进行前向计算，得到输出特征图
print(f"conv_output: {conv_output}")
