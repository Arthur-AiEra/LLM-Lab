import numpy as np
import pandas as pd
import tensorflow as tf
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

# 划分训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25)

# 使用MirroredStrategy实现数据并行
strategy = tf.distribute.MirroredStrategy()
print('设备数量:', strategy.num_replicas_in_sync)

with strategy.scope():
    # 构建神经网络（TensorFlow Sequential API）
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(13,)),
        tf.keras.layers.Dense(1)
    ])
    # 定义损失函数和优化器
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss='mse')

# 训练
max_epoch = 300
history = model.fit(train_x, train_y, epochs=max_epoch, verbose=0)

# 绘制loss曲线
plt.plot(np.arange(max_epoch), history.history['loss'])
plt.title('Loss Value in all iterations')
plt.xlabel('Iteration')
plt.ylabel('Mean Loss Value')
plt.show()

# 测试
predict_list = model.predict(test_x)
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