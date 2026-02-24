# 二手车价格预测 - TensorFlow 神经网络模型

## 项目概述

这是一个使用 TensorFlow 构建的二层神经网络，用于预测二手车价格。

## 模型架构

```
输入层 (29 features)
    ↓
Dense Layer 1: 128 neurons, ReLU activation
    ↓
Dense Layer 2: 64 neurons, ReLU activation
    ↓
Output Layer: 1 neuron (regression)
```

**总参数数**: 12,161

## 数据集

- **训练数据**: `used_car_train_20200313.csv`
- **测试数据**: `used_car_testB_20200421.csv`
- **样本提交**: `used_car_sample_submit.csv`

### 数据处理

- 自动提取所有数值特征
- 移除空值行
- 使用 StandardScaler 进行标准化缩放
- 训练集:验证集 = 90%:10%

## 依赖安装

```bash
pip install -r requirements.txt
```

### 依赖项

- tensorflow
- pandas
- scikit-learn
- joblib

## 训练脚本

### 基本用法

```bash
python3 train_model.py
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data` | str | `used_car_train_20200313.csv` | 训练数据文件路径 |
| `--epochs` | int | `200` | 训练轮数 |
| `--batch_size` | int | `32` | 批量大小 |
| `--lr` | float | `1e-4` | 学习率（建议较小值） |
| `--quick` | flag | - | 快速测试模式（子集 + 5 轮） |

### 示例

#### 快速测试
```bash
python3 train_model.py --quick
```

#### 完整训练（200轮）
```bash
python3 train_model.py --epochs 200 --lr 1e-4
```

#### 自定义参数
```bash
python3 train_model.py --epochs 300 --batch_size 64 --lr 5e-5 --data used_car_train_20200313.csv
```

## 训练配置

- **优化器**: Adam
- **损失函数**: Mean Squared Error (MSE)
- **验证指标**: Mean Absolute Error (MAE)
- **早停法**: 
  - 监控指标: `val_mae`
  - 耐心值: 10 (10轮无改进后停止)
  - 恢复最佳权重: 是

## 模型输出

训练完成后，会保存以下文件：

- `model_two_layer.h5` - 训练好的模型文件
- `scaler.joblib` - 特征标准化器（用于预测）

## 训练监控

训练日志会输出：

```
Epoch 1/200
[Training metrics displaying loss and MAE for training and validation]
...
Final validation MAE: XXXX.XXXX, MSE: XXXXXX.XXXX
Model saved to model_two_layer.h5, scaler saved to scaler.joblib
```

## 性能改进建议

1. **学习率**: 默认使用 `1e-4`（较小），可根据验证损失调整
2. **训练轮数**: 默认 200 轮，配合早停法自动终止
3. **批量大小**: 默认 32，可根据显存调整
4. **早停耐心值**: 可在代码中修改 `patience` 参数

## 文件说明

| 文件 | 说明 |
|------|------|
| `train_model.py` | 主训练脚本 |
| `requirements.txt` | 依赖包列表 |
| `used_car_train_20200313.csv` | 训练数据（空格分隔） |
| `model_two_layer.h5` | 训练后的模型 |
| `scaler.joblib` | 特征标准化器 |

## 注意事项

- CSV 文件使用**空格**作为分隔符（不是逗号）
- 所有非数值列会被自动忽略
- 包含空值的行会被过滤掉
- 建议在有 GPU 的环境运行以加速训练
