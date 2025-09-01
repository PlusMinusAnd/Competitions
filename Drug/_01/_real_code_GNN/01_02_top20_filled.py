import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# 🔥 데이터 로드
df = pd.read_csv("./Drug/data/features_rdkit_top20.csv")

# 🔍 결측 피처와 입력 피처 구분
target_columns_with_nan = ['BCUT2D_MRLOW', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_LOGPHI']
input_columns = [col for col in df.columns if col not in target_columns_with_nan + ['index', 'SMILES', 'Inhibition']]

# ✅ 복사본 생성
filled_df = df.copy()

# 🔁 각 결측 피처에 대해 MLPRegressor로 채우기
for target_col in target_columns_with_nan:
    # 결측 없는 행만 학습에 사용
    train_data = filled_df[filled_df[target_col].notna()]
    X_train = train_data[input_columns]
    y_train = train_data[target_col]

    # 표준화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 모델 구성 및 학습
    model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # 결측값 위치 예측
    missing_rows = filled_df[filled_df[target_col].isna()]
    if not missing_rows.empty:
        X_missing = scaler.transform(missing_rows[input_columns])
        predicted_values = model.predict(X_missing)
        filled_df.loc[filled_df[target_col].isna(), target_col] = predicted_values

# ✅ 결측값 확인 및 저장
print(filled_df.isna().sum())
filled_df.to_csv("./Drug/_engineered_data/filled_train_final.csv", index=False)
print("✔️ 저장 완료: filled_train_final.csv")
