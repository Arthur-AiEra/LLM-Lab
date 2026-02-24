import numpy as np

# 初始化网络（初始权重和偏置）
def init_network():
    network = dict()
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

# sigmoid激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
  
# 恒等函数，作为输出层的激活函数
def identity_function(x):
    return x

# 前向传播
def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    # hidden layer 1
    a1 = np.dot(x, W1) + b1
	# 前向传播 使用sigmoid
    z1 = sigmoid(a1)

    # hidden layer 2
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    # output layer
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y

# 初始化网络
network = init_network()
# 设置输入值
x = np.array([1.0, 0.5])
# 前向传播
y = forward(network, x)
print(y)
