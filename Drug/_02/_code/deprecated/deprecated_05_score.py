import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import os

# 평가 수식 정의
def leaderboard_score(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    norm_rmse = rmse / (np.max(y_true) - np.min(y_true))
    A = min(norm_rmse, 1.0)

    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        B = 0.0
    else:
        B = np.clip(pearsonr(y_true, y_pred)[0], 0, 1)

    score = 0.5 * (1 - A) + 0.5 * B
    return {
        "RMSE": rmse,
        "Normalized RMSE": A,
        "Pearson": B,
        "Leaderboard Score": score
    }

# 데이터 불러오기
train = pd.read_csv("./Drug/train.csv")
y = train["Inhibition"].values

# OOF 예측 불러오기
dnn_oof = np.load("./Drug/_02/2_npy/dnn_oof.npy")
dmpnn_oof = np.load("./Drug/_02/2_npy/dmpnn_oof.npy") if os.path.exists("./Drug/_02/2_npy/dmpnn_oof.npy") else np.zeros_like(dnn_oof)

# 앙상블 OOF 예측
alpha = 0.5  # 가중치 비율 (원하면 튜닝 가능)
final_oof = alpha * dmpnn_oof + (1 - alpha) * dnn_oof

# 점수 출력
score_result = leaderboard_score(y, final_oof)
print("🔍 리더보드 기준 앙상블 평가 결과")
for k, v in score_result.items():
    print(f"{k}: {v:.5f}")
