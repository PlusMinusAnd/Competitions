import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import pandas as pd

# 🎯 고정 시드
r = 394
np.random.seed(r)

# ✅ 두 모델의 OOF 및 예측 결과 (예시로 불러온다고 가정)
gnn_oof = np.load("./Drug/_02/full_pipeline/gat_oof.npy")
gnn_pred = np.load("./Drug/_02/full_pipeline/gat_preds.npy")
pycaret_oof = np.load("./Drug/_02/full_pipeline/pycaret_oof.npy")
pycaret_pred = np.load("./Drug/_02/full_pipeline/pycaret_preds.npy")

# ✅ 앙상블 가중치 설정 (가중 평균, 가중치는 실험적으로 조정)
w1 = 0.6  # GNN
w2 = 0.4  # PyCaret

# ✅ 앙상블 결과 계산
final_oof = w1 * gnn_oof + w2 * pycaret_oof
final_pred = w1 * gnn_pred + w2 * pycaret_pred

# ✅ 평가 함수 정의
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    pearson = pearsonr(y_true, y_pred)[0]
    score = 0.5 * (1 - min(nrmse, 1)) + 0.5 * np.clip(pearson, 0, 1)
    return rmse, nrmse, pearson, score

# ✅ 최종 평가
submission_df = pd.read_csv("./Drug/_02/full_pipeline/0_dataset/train_descriptor.csv")
y_true = submission_df["Inhibition"].values  # 실제값 불러오기
rmse, nrmse, pearson, score = evaluate(y_true, final_oof)

print(f"📊 GNN + PyCaret 앙상블 성능:")
print(f"RMSE     : {rmse:.5f}")
print(f"NRMSE    : {nrmse:.5f}")
print(f"Pearson  : {pearson:.5f}")
print(f"Score    : {score:.5f}")

# ✅ Submission 저장
import pandas as pd
submission = pd.read_csv("./Drug/sample_submission.csv")
submission["Inhibition"] = final_pred
submission.to_csv("submission_gnn_pycaret.csv", index=False)
