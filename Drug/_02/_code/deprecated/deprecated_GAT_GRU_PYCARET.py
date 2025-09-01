import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import datetime

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

# 데이터 로드
r = 238
y = pd.read_csv("./Drug/train.csv")["Inhibition"].values
gat_oof = np.load("./Drug/_02/full_pipeline/gat_oof.npy")
boost_oof = np.load("./Drug/_02/full_pipeline/boost_oof.npy")
pycaret_oof = np.load("./Drug/_02/full_pipeline/pycaret_oof.npy")       # ✅ 추가
gat_preds = np.load("./Drug/_02/full_pipeline/gat_preds.npy")
boost_preds = np.load("./Drug/_02/full_pipeline/boost_preds.npy")
pycaret_preds = np.load("./Drug/_02/full_pipeline/pycaret_preds.npy")   # ✅ test 예측
test_id = pd.read_csv("./Drug/test.csv")["ID"]

# 🔍 최적 α 찾기 (GAT vs Boost)
best_score = -np.inf
best_alpha = 0.5

for alpha in np.linspace(0, 1, 21):
    blended_oof = alpha * gat_oof + (1 - alpha) * boost_oof
    score = print_scores(y, blended_oof, label=f"α(GAT)={alpha:.2f}, β(Boost)={1-alpha:.2f}")
    if score > best_score:
        best_score = score
        best_alpha = alpha

# ✅ 최종 blending + pycaret test 가중치 추가 (예: GAT:Boost:PyCaret = 2:3:5)
a, b = best_alpha, 1 - best_alpha
inter_blend = a * gat_preds + b * boost_preds
final_preds = 0.4 * inter_blend + 0.6 * pycaret_preds  # ✅ test용

# 최종 oof는 GAT + Boost + PyCaret
final_oof = 0.4 * (a * gat_oof + b * boost_oof) + 0.6 * pycaret_oof  # ✅ train용
print_scores(y, final_oof, label=f"Final Ensemble (GAT+Boost+PyCaret), α={a:.2f}")

# 저장
submission = pd.DataFrame({
    "ID": test_id,
    "Inhibition": final_preds
})
now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
filename = f"submission_{r}_final_ensemble_with_pycaret({now}).csv"
submission.to_csv(f"./Drug/_02/full_pipeline/{filename}", index=False)
print(f"✅ 최종 앙상블 저장 완료 → {filename}")
