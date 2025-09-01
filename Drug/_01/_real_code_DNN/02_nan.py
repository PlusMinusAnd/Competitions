import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras import layers, models

# 데이터 로드
df = pd.read_csv('./Drug/_engineered_data_DNN/train_with_rdkit_features.csv')

# 타겟 제거, 수치형 피처만 사용
X_all = df.drop(columns=['Inhibition'])
X_all = X_all.select_dtypes(include=[np.number])

# 결측치가 있는 컬럼만
missing_cols = X_all.columns[X_all.isnull().any()]

# 결측치 채운 데이터 저장용
X_filled = X_all.copy()

# DNN 모델 정의 함수
def build_dnn(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# 각 결측 컬럼에 대해 반복
for col in missing_cols:
    print(f"Filling NaN for: {col}")
    
    # 타겟 설정
    y = X_all[col]
    X = X_all.drop(columns=[col])
    
    # 훈련용: 결측치 없는 부분
    notnull_mask = y.notnull()
    X_train = X[notnull_mask]
    y_train = y[notnull_mask]
    
    # 예측용: 결측치 있는 부분
    null_mask = y.isnull()
    X_pred = X[null_mask]
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_pred_scaled = scaler.transform(X_pred)
    
    # DNN 훈련
    model = build_dnn(X_train_scaled.shape[1])
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
    
    # 예측 및 대체
    y_pred = model.predict(X_pred_scaled).flatten()
    X_filled.loc[null_mask, col] = y_pred

# 채워진 데이터와 원래 ID/SMILES/Inhibition 붙이기
result = pd.concat([df.iloc[:, [0, 1]], X_filled, df['Inhibition']], axis=1)

# 저장
result.to_csv('./Drug/_engineered_data_DNN/train_filled_dnn.csv', index=False)
