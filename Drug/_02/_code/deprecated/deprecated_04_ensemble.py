##### DNN 데이터 제작 및 앙상블(with DMPNN, DNN) #####

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# 시드 고정
np.random.seed(73)

### 1. DNN 예측 (RandomForest) ###
print("🔹 DNN 학습 및 예측 시작")

# RDKit descriptor 기반 데이터 로드
train = pd.read_csv("./Drug/_02/0_dataset/train_descriptor.csv")
test = pd.read_csv("./Drug/_02/0_dataset/test_descriptor.csv")

X = train.drop(columns=["Inhibition"])
y = train["Inhibition"]
X_test = test.copy()

# 결측값 처리
X = X.fillna(X.mean())
X_test = X_test.fillna(X.mean())

# 모델 정의
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=73)

# Cross-validation + 예측
kf = KFold(n_splits=5, shuffle=True, random_state=73)
oof = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model.fit(X_train, y_train)
    oof[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / kf.n_splits

# 평가
rmse = np.sqrt(mean_squared_error(y, oof))
print(f"✅ DNN (RF) CV RMSE: {rmse:.4f}")

### 2. DMPNN 결과 로드 ###
print("🔹 DMPNN 예측 로드")
dmpnn_preds = pd.read_csv("./Drug/_02/1_dmpnn/submission_dmpnn.csv")["Inhibition"].values

### 3. 앙상블 ###
print("🔹 앙상블 수행 (0.5 DNN + 0.5 DMPNN)")
final_preds = 0.5 * test_preds + 0.5 * dmpnn_preds

# OOF 및 test 예측 저장
np.save("./Drug/_02/2_npy/dnn_oof.npy", oof)
np.save("./Drug/_02/2_npy/dnn_preds.npy", test_preds)
print("✅ oof 저장 완료")

# 저장
submission = pd.DataFrame({
    "ID": pd.read_csv("./Drug/test.csv")["ID"],
    "Inhibition": final_preds
})
submission.to_csv("./Drug/_02/3_submission/submission_ensemble.csv", index=False)
print("✅ submission_ensemble.csv 저장 완료")


