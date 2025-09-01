
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import os

# 시드 고정
np.random.seed(73)

# 리더보드 점수 출력 함수
def print_scores(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    pearson = pearsonr(y_true, y_pred)[0]
    score = 0.5 * (1 - min(nrmse, 1)) + 0.5 * np.clip(pearson, 0, 1)

    print(f"📊 {label} 결과")
    print(f"RMSE     : {rmse:.5f}")
    print(f"NRMSE    : {nrmse:.5f}")
    print(f"Pearson  : {pearson:.5f}")
    print(f"Score📈  : {score:.5f}")

### 1. DNN 예측 (RandomForest) ###
print("🔹 DNN 학습 및 예측 시작")
train = pd.read_csv("./Drug/_02/0_dataset/train_descriptor.csv")
test = pd.read_csv("./Drug/_02/0_dataset/test_descriptor.csv")
X = train.drop(columns=["Inhibition"])
y = train["Inhibition"].values
X_test = test.copy()

# 모델 정의
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=73)

kf = KFold(n_splits=5, shuffle=True, random_state=73)
oof = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model.fit(X_train, y_train)
    oof[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / kf.n_splits

# DNN 점수 출력
print_scores(y, oof, label="DNN")

# 저장
np.save("./Drug/_02/2_npy/dnn_oof.npy", oof)
np.save("./Drug/_02/2_npy/dnn_preds.npy", test_preds)
print("✅ dnn_oof.npy, dnn_preds.npy 저장 완료")

### 2. DMPNN 결과 로드 ###
print("🔹 DMPNN 예측 로드")
dmpnn_preds = pd.read_csv("./Drug/_02/1_dmpnn/submission_dmpnn.csv")["Inhibition"].values
dmpnn_oof_path = "./Drug/_02/2_npy/dmpnn_oof.npy"
if os.path.exists(dmpnn_oof_path):
    dmpnn_oof = np.load(dmpnn_oof_path)
else:
    dmpnn_oof = np.zeros_like(oof)
    print("⚠️ dmpnn_oof.npy 파일이 없어 0으로 대체됨")

### 3. 앙상블 (alpha 튜닝) ###
print("🔹 앙상블 가중치 탐색")
best_score = -np.inf
best_alpha = 0.5
for alpha in np.linspace(0, 1, 21):
    blended = alpha * dmpnn_oof + (1 - alpha) * oof
    rmse = np.sqrt(mean_squared_error(y, blended))
    nrmse = rmse / (np.max(y) - np.min(y))
    pearson = pearsonr(y, blended)[0]
    score = 0.5 * (1 - min(nrmse, 1)) + 0.5 * np.clip(pearson, 0, 1)
    if score > best_score:
        best_score = score
        best_alpha = alpha

print(f"✅ 최적 alpha: {best_alpha:.2f} | 최종 리더보드 Score: {best_score:.5f}")

# 최종 앙상블 예측
final_preds = best_alpha * dmpnn_preds + (1 - best_alpha) * test_preds
final_oof = best_alpha * dmpnn_oof + (1 - best_alpha) * oof

# 점수 출력
print_scores(y, final_oof, label=f"Ensemble α={best_alpha:.2f}")

# 저장
submission = pd.DataFrame({
    "ID": pd.read_csv("./Drug/test.csv")["ID"],
    "Inhibition": final_preds
})
submission.to_csv("./Drug/_02/3_submission/submission_ensemble.csv", index=False)
print("✅ submission_ensemble.csv 저장 완료")
