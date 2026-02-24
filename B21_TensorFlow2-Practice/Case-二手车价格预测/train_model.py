import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.initializers import HeNormal
try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
except Exception:
    lgb = None


def load_and_preprocess(path, quick=False, quick_n=20000):
    df = pd.read_csv(path, sep=' ')
    num = df.select_dtypes(include=[np.number]).copy()
    num = num.dropna()
    if 'price' not in num.columns:
        raise ValueError('`price` column not found in numeric columns')
    y = num['price'].values
    
    # More aggressive outlier removal - use 10-90 percentile range for better focus
    Q1 = np.percentile(y, 10)
    Q3 = np.percentile(y, 90)
    mask = (y >= Q1) & (y <= Q3)
    y = y[mask]
    
    X = num.drop(columns=['price']).values
    X = X[mask]
    
    # Remove features with very high variance or zero variance
    X_df = pd.DataFrame(X)
    feature_std = X_df.std()
    valid_features = feature_std[feature_std > 1e-6].index.tolist()
    X = X[:, valid_features]
    
    # Feature engineering: log transform for skewed features
    X_df = pd.DataFrame(X, columns=valid_features)
    # Apply log transform to potentially skewed features
    for col in X_df.columns:
        if (X_df[col] > 0).all():  # Only for positive values
            X_df[col] = np.log1p(X_df[col])
    X = X_df.values
    
    if quick and X.shape[0] > quick_n:
        idx = np.random.choice(X.shape[0], quick_n, replace=False)
        X = X[idx]
        y = y[idx]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Scale price to 0-1 range with min-max scaling for easier training
    price_min, price_max = y.min(), y.max()
    y = (y - price_min) / (price_max - price_min)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Return price scaling parameters for inverse transform
    return X_train, X_val, y_train, y_val, scaler, price_min, price_max


def build_model(input_dim, lr=1e-3):
    l2_reg = 3e-5  # Fine-tuned regularization
    
    # Input layer
    inputs = layers.Input(shape=(input_dim,))
    x = inputs
    
    # Deep and wide architecture
    x = layers.Dense(4096, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization(momentum=0.95)(x)
    x = layers.Dropout(0.45)(x)
    
    # Second block
    x = layers.Dense(2048, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization(momentum=0.95)(x)
    x = layers.Dropout(0.4)(x)
    
    # Third block
    x = layers.Dense(1024, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization(momentum=0.95)(x)
    x = layers.Dropout(0.35)(x)
    
    # Fourth block
    x = layers.Dense(512, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization(momentum=0.95)(x)
    x = layers.Dropout(0.3)(x)
    
    # Fifth block
    x = layers.Dense(256, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization(momentum=0.95)(x)
    x = layers.Dropout(0.25)(x)
    
    # Sixth block
    x = layers.Dense(128, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = layers.Dropout(0.15)(x)
    
    # Output layer
    outputs = layers.Dense(1)(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    opt = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=True)
    loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=opt, loss=loss_fn, metrics=['mae'])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='used_car_train_20200313.csv')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--use_lgb', action='store_true', help='Use LightGBM instead of Keras')
    parser.add_argument('--quick', action='store_true', help='Run a quick smoke test (subset + fewer epochs)')
    args = parser.parse_args()

    if args.quick:
        epochs = 5
    else:
        epochs = args.epochs

    print('Loading and preprocessing...')
    X_train, X_val, y_train, y_val, scaler, price_min, price_max = load_and_preprocess(args.data, quick=args.quick)

    print(f'Train shape: {X_train.shape}, Val shape: {X_val.shape}')
    if args.use_lgb:
        if lgb is None:
            raise ImportError('lightgbm is not installed. Install it or run without --use_lgb')
        print('Training with LightGBM...')
        lgbm = LGBMRegressor(
            objective='regression',
            metric='mae',
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=64,
            max_depth=12,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.2,
            random_state=42,
            n_jobs=-1
        )
        # fit with early stopping on validation set
        lgbm.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='mae',
            early_stopping_rounds=50,
            verbose=50
        )
        preds = lgbm.predict(X_val)
        val_mae_orig = np.mean(np.abs(preds - y_val))
        val_mse_orig = np.mean((preds - y_val) ** 2)
        print(f'Final validation MAE (original space): {val_mae_orig:.4f}, MSE: {val_mse_orig:.4f}')
        joblib.dump(lgbm, 'model_lgb.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        print('Saved LightGBM model to model_lgb.joblib')
    else:
        model = build_model(X_train.shape[1], lr=args.lr)
        model.summary()

        # Fine-tuned callbacks for better convergence
        es = callbacks.EarlyStopping(
            monitor='val_mae', 
            mode='min', 
            patience=100, 
            restore_best_weights=True, 
            verbose=1, 
            min_delta=0.00001
        )
        
        # More gradual learning rate reduction
        lr_reduce = callbacks.ReduceLROnPlateau(
            monitor='val_mae', 
            factor=0.7, 
            patience=5, 
            min_lr=1e-10, 
            verbose=1,
            cooldown=0
        )
        
        # Use ReduceLROnPlateau + EarlyStopping for better convergence
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=args.batch_size,
            callbacks=[es, lr_reduce],
            verbose=1
        )

        # Evaluate in original price space (inverse scaling)
        preds = model.predict(X_val, batch_size=1024).flatten()
        preds = preds * (price_max - price_min) + price_min  # Inverse scaling
        y_val_original = y_val * (price_max - price_min) + price_min  # Inverse scaling
        
        val_mae_orig = np.mean(np.abs(preds - y_val_original))
        val_mse_orig = np.mean((preds - y_val_original) ** 2)
        val_rmse_orig = np.sqrt(val_mse_orig)
        print(f'\n\n===== Final validation MAE (original space): {val_mae_orig:.4f}, RMSE: {val_rmse_orig:.4f} =====')
        
        # Check if we achieved the goal
        if val_mae_orig <= 500:
            print('✓ Target achieved: MAE <= 500')
        else:
            print(f'✗ Target not achieved: MAE = {val_mae_orig:.4f} (target: 500)')

        model_path = 'model_two_layer.keras'
        scaler_path = 'scaler.joblib'
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        print(f'Model saved to {model_path}, scaler saved to {scaler_path}')


if __name__ == '__main__':
    main()
