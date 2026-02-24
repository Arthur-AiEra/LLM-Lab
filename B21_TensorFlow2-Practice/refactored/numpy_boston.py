# 使用numpy实现Boston房价预测
import numpy as np
import pandas as pd  # 新增pandas导入
import matplotlib.pyplot as plt  # 绘制误差曲线

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'STHeiti', 'Heiti TC', 'DejaVu Sans']  # macOS中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据加载（从housing.csv读取，无表头，空格分隔）
data = pd.read_csv('housing.csv', header=None, sep=r'\s+')
X = data.iloc[:, :-1].values  # 前13列为特征
y = data.iloc[:, -1].values  # 最后一列为目标
# 将y转为列向量，方便后续矩阵运算
y = y.reshape(-1, 1)

# 数据规范化（标准化处理，使每一列特征均值为0，方差为1）
x = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
print(x)  # 打印标准化后的特征矩阵
n_feature = x.shape[1]  # 特征数量（13）
print(n_feature)

n_hidden = 10  # 隐藏层神经元数量
# 初始化输入层到隐藏层的权重和偏置
w1 = np.random.randn(n_feature, n_hidden)
b1 = np.zeros(n_hidden)
# 初始化隐藏层到输出层的权重和偏置
w2 = np.random.randn(n_hidden, 1)
b2 = np.zeros(1)

def Relu(x):
    # ReLU激活函数，小于0的部分置为0
    result = np.where(x>0, x, 0)
    return result

# 设置学习率
learning_rate = 1e-5

def MSE_loss(y, y_hat):
    # 均方误差损失函数
    return np.mean(np.square(y_hat - y))

def Linear(X, W1, b1):
    # 线性变换：X @ W1 + b1
    return X.dot(W1) + b1

# 输入特征维度13，隐藏层10维，输出1维（房价）
losses = []  # 保存每个迭代的损失
for t in range(5000):
    # 前向传播：输入->隐藏层->输出
    l1 = Linear(x, w1, b1)      # 输入层到隐藏层的线性变换
    s1 = Relu(l1)               # 隐藏层激活
    y_pred = Linear(s1, w2, b2) # 隐藏层到输出层的线性变换，得到预测值

    # 计算损失
    loss = MSE_loss(y, y_pred)
    losses.append(loss)  # 保存损失
    print(t, loss)

    # 反向传播：计算梯度
    grad_y_pred = 2.0 * (y_pred - y)           # 损失对预测输出的梯度
    grad_w2 = s1.T.dot(grad_y_pred)            # 损失对w2的梯度
    grad_temp_relu = grad_y_pred.dot(w2.T)     # 损失对隐藏层输出的梯度
    grad_temp_relu[l1<0] = 0                   # ReLU小于0的部分梯度置零
    grad_w1 = x.T.dot(grad_temp_relu)          # 损失对w1的梯度

    # 更新权重参数
    w1 = w1 - learning_rate * grad_w1
    w2 = w2 - learning_rate * grad_w2

# 输出最终训练得到的权重参数
print('w1={} \n w2={}'.format(w1, w2))

# 绘制误差曲线
plt.figure(figsize=(10, 6))
plt.plot(losses, linewidth=2)
plt.xlabel('迭代次数', fontsize=12)
plt.ylabel('均方误差 (MSE)', fontsize=12)
plt.title('模型训练过程中的误差变化', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
