import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib


def predict_prices(test_data_path, model_path='model_two_layer.keras', scaler_path='scaler.joblib'):
    """
    使用训练好的模型预测二手车价格
    
    Args:
        test_data_path: 测试数据CSV文件路径
        model_path: 模型文件路径
        scaler_path: 标准化器文件路径
    
    Returns:
        DataFrame with SaleID and predicted prices
    """
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print(f"Loading scaler from {scaler_path}...")
    scaler = joblib.load(scaler_path)
    
    print(f"Loading test data from {test_data_path}...")
    df = pd.read_csv(test_data_path, sep=' ')
    
    # 保存SaleID用于输出
    sale_ids = df['SaleID'].values
    
    # 提取数值特征（排除price，如果存在）
    num_features = df.select_dtypes(include=[np.number]).copy()
    cols_to_drop = []
    if 'price' in num_features.columns:
        cols_to_drop.append('price')
    
    if cols_to_drop:
        X = num_features.drop(columns=cols_to_drop).values
    else:
        X = num_features.values
    
    print(f"Test data shape: {X.shape}")
    
    # 使用保存的scaler进行标准化
    X_scaled = scaler.transform(X)
    
    # 进行预测
    print("Making predictions...")
    prices = model.predict(X_scaled, verbose=0)
    
    # 创建结果DataFrame
    results = pd.DataFrame({
        'SaleID': sale_ids,
        'price': prices.flatten()
    })
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', default='used_car_testB_20200421.csv', help='Test data path')
    parser.add_argument('--model', default='model_two_layer.keras', help='Model path')
    parser.add_argument('--scaler', default='scaler.joblib', help='Scaler path')
    parser.add_argument('--output', default='predictions.csv', help='Output CSV path')
    args = parser.parse_args()
    
    # 进行预测
    results = predict_prices(
        args.test,
        model_path=args.model,
        scaler_path=args.scaler
    )
    
    # 保存结果
    results.to_csv(args.output, index=False, sep=' ')
    print(f"\nPredictions saved to {args.output}")
    print(f"\nFirst 10 predictions:")
    print(results.head(10))
    print(f"\nStatistics:")
    print(results['price'].describe())


if __name__ == '__main__':
    main()
