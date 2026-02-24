# 使用numpy实现一个神经网络
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体，解决中文显示乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# n为样本大小，d_in为输入维度,h为隐藏层维度,d_out为输出维度
n, d_in, h, d_out = 64, 1000, 100, 10

# 随机生成输入数据x和目标输出y
x = np.random.randn(n, d_in)      # 输入数据，形状为(64, 1000)
y = np.random.randn(n, d_out)     # 目标输出，形状为(64, 10)

# 随机初始化权重参数
# 输入层到隐藏层的权重（1000，100）
w1 = np.random.randn(d_in, h)
# 隐藏层到输出层的权重（100，10）
w2 = np.random.randn(h, d_out)
# 设置学习率
learning_rate = 1e-6

# 用于记录每次迭代的loss值
loss_history = []

# 训练500次
for t in range(500):
    # 前向传播
    temp = x.dot(w1)                  # 输入层到隐藏层的线性变换
    temp_relu = np.maximum(temp, 0)   # ReLU激活函数，隐藏层输出
    y_pred = temp_relu.dot(w2)        # 隐藏层到输出层的线性变换，得到预测值

    # 计算损失函数（均方误差和）
    loss = np.square(y_pred - y).sum()
    loss_history.append(loss)  # 记录loss值
    print(t, loss)

    # 反向传播，计算梯度
    grad_y_pred = 2.0 * (y_pred - y)              # 损失对预测输出的梯度
    #print('grad_y_pred=', grad_y_pred.shape) #(64, 10)
    grad_w2 = temp_relu.T.dot(grad_y_pred)        # 损失对w2的梯度
    grad_temp_relu = grad_y_pred.dot(w2.T)        # 损失对隐藏层输出的梯度
    grad_temp = grad_temp_relu.copy()             # 复制一份用于ReLU处理
    grad_temp[temp<0] = 0                         # ReLU小于0的部分梯度置零
    grad_w1 = x.T.dot(grad_temp)                  # 损失对w1的梯度

    # 更新权重参数
    w1 = w1 - learning_rate * grad_w1
    w2 = w2 - learning_rate * grad_w2

# 绘制Loss曲线
plt.figure(figsize=(10, 6))
plt.plot(loss_history, 'b-', linewidth=2)
plt.title('训练过程中的Loss变化曲线', fontsize=14)
plt.xlabel('迭代次数', fontsize=12)
plt.ylabel('Loss值', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

# 输出最终训练得到的权重参数
print(w1, w2)
# print(w1) 
# print(w2) 