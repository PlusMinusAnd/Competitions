import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import datetime
import random

# 시드 고정
r = 394 #random.randint(1,1000)
random.seed(r)
np.random.seed(r)
# torch.manual_seed(r)

def print_scores(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    pearson = pearsonr(y_true, y_pred)[0]
    score = 0.5 * (1 - min(nrmse, 1)) + 0.5 * np.clip(pearson, 0, 1)
    print(f"📊 {label}")
    print(f"RMSE     : {rmse:.5f}")
    print(f"NRMSE    : {nrmse:.5f}")
    print(f"Pearson  : {pearson:.5f}")
    print(f"Score📈  : {score:.5f}")
    return score

# 🔹 데이터 로드
y = pd.read_csv("./Drug/train.csv")["Inhibition"].values
gat_oof = np.load("./Drug/_02/full_pipeline/gat_oof.npy")              # Set2Set 기반 GAT OOF
pycaret_oof = np.load("./Drug/_02/full_pipeline/pycaret_oof.npy")
gat_preds = np.load("./Drug/_02/full_pipeline/gat_preds.npy")
pycaret_preds = np.load("./Drug/_02/full_pipeline/pycaret_preds.npy")
test_id = pd.read_csv("./Drug/test.csv")["ID"]

# 🔍 최적 alpha 탐색 (GAT vs PyCaret)
best_score = -np.inf
best_alpha = 0.5

for alpha in np.linspace(0, 1, 21):
    final_oof = alpha * gat_oof + (1 - alpha) * pycaret_oof
    score = print_scores(y, final_oof, label=f"α(GAT)={alpha:.2f}, β(PyCaret)={1-alpha:.2f}")
    if score > best_score:
        best_score = score
        best_alpha = alpha

# ✅ 최적 가중치 적용
a = best_alpha
b = 1 - a

final_oof = a * gat_oof + b * pycaret_oof
final_preds = a * gat_preds + b * pycaret_preds

# 🔹 최종 점수 출력
final_score = print_scores(y, final_oof, label=f"Final Ensemble GAT+PyCaret α={a:.2f}")

# 🔹 제출 저장
now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
filename = f"submission_{r}_GAT_Set2Set_PyCaret({now}).csv"
submission = pd.DataFrame({
    "ID": test_id,
    "Inhibition": final_preds
})
submission.to_csv(f"./Drug/_02/full_pipeline/{filename}", index=False)

print(f"✅ 최종 앙상블 저장 완료 → {filename}")
print(f"랜덤 시드 : {r}")
