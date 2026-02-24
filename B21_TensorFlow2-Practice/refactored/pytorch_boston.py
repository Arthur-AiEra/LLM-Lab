import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 数据加载（从housing.csv读取，无表头，空格分隔）
data = pd.read_csv('housing.csv', header=None, delim_whitespace=True)
x = data.iloc[:, :-1].values  # 前13列为特征
y = data.iloc[:, -1].values  # 最后一列为目标
print(x.shape)
print(y.shape)

# 将y转换形状
y = y.reshape(-1, 1)

# 数据规范化
ss_input = MinMaxScaler()
x = ss_input.fit_transform(x)

# 转为torch张量
torch_x = torch.from_numpy(x).type(torch.FloatTensor)
torch_y = torch.from_numpy(y).type(torch.FloatTensor)

# 划分训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(torch_x, torch_y, test_size=0.25)

# 构建神经网络
model = nn.Sequential(
    nn.Linear(13, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练
max_epoch = 300
iter_loss = []
for i in range(max_epoch):
    # 前向传播
    y_pred = model(train_x)
    # 计算损失
    loss = criterion(y_pred, train_y)
    iter_loss.append(loss.item())
    # 清空梯度
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 更新参数
    optimizer.step()

# 绘制loss曲线
plt.plot(np.arange(max_epoch), np.array(iter_loss))
plt.title('Loss Value in all iterations')
plt.xlabel('Iteration')
plt.ylabel('Mean Loss Value')
plt.show()

# 测试
output = model(test_x)
predict_list = output.detach().numpy()
print(predict_list)

# 真实值与预测值的散点图
x_idx = np.arange(test_x.shape[0])
y1 = np.array(predict_list)
y2 = np.array(test_y)
line1 = plt.scatter(x_idx, y1, c='red', label='predict')
line2 = plt.scatter(x_idx, y2, c='yellow', label='real')
plt.legend(loc='best')
plt.title('Prediction VS Real')
plt.ylabel('House Price')
plt.show() 