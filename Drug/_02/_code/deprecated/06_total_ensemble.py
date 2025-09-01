import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import datetime

# 점수 함수
def print_scores(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    pearson = pearsonr(y_true, y_pred)[0]
    score = 0.5 * (1 - min(nrmse, 1)) + 0.5 * np.clip(pearson, 0, 1)
    print(f"\n📊 {label}")
    print(f"RMSE     : {rmse:.5f}")
    print(f"NRMSE    : {nrmse:.5f}")
    print(f"Pearson  : {pearson:.5f}")
    print(f"Score📈  : {score:.5f}")
    return score

# 데이터 불러오기
r = 394
y = pd.read_csv("./Drug/train.csv")["Inhibition"].values
gnn_oof = np.load("./Drug/_02/full_pipeline/gat_oof.npy")
boost_oof = np.load("./Drug/_02/full_pipeline/dnn_residual_oof.npy")
pycaret_oof = np.load("./Drug/_02/full_pipeline/pycaret_oof.npy")

gnn_preds = np.load("./Drug/_02/full_pipeline/gat_preds.npy")
boost_preds = np.load("./Drug/_02/full_pipeline/dnn_residual_oof.npy")
pycaret_preds = np.load("./Drug/_02/full_pipeline/pycaret_preds.npy")

test_id = pd.read_csv("./Drug/test.csv")["ID"]

# ✅ 길이 맞추기 (가장 짧은 길이에 맞춤)
min_len = min(len(y), len(gnn_oof), len(boost_oof), len(pycaret_oof))
y = y[:min_len]
gnn_oof = gnn_oof[:min_len]
boost_oof = boost_oof[:min_len]
pycaret_oof = pycaret_oof[:min_len]

# ✅ 예측값도 길이 맞추기
min_test_len = min(len(gnn_preds), len(boost_preds), len(pycaret_preds), len(test_id))
gnn_preds = gnn_preds[:min_test_len]
boost_preds = boost_preds[:min_test_len]
pycaret_preds = pycaret_preds[:min_test_len]
test_id = test_id[:min_test_len]

# Step 1: GNN + Boost 최적 앙상블
best_score = -np.inf
best_alpha = 0.5

for alpha in np.linspace(0, 1, 21):
    blend_oof = alpha * gnn_oof + (1 - alpha) * boost_oof
    score = print_scores(y, blend_oof, label=f"α(GNN)={alpha:.2f}, β(Boost)={1-alpha:.2f}")
    if score > best_score:
        best_score = score
        best_alpha = alpha

# Step 2: GNN+Boost 결과와 PyCaret 합치기 (0.4 vs 0.6)
a, b = best_alpha, 1 - best_alpha
inter_blend_preds = a * gnn_preds + b * boost_preds
final_preds = 0.4 * inter_blend_preds + 0.6 * pycaret_preds

final_oof = 0.4 * (a * gnn_oof + b * boost_oof) + 0.6 * pycaret_oof

# 최종 점수 출력
final_score = print_scores(y, final_oof, label=f"🎯 Final Ensemble (GNN+Boost+PyCaret), α={a:.2f}")

# 저장
submission = pd.DataFrame({
    "ID": test_id,
    "Inhibition": final_preds
})
now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
filename = f"submission_{r}_final_ensemble({now}).csv"
submission.to_csv(f"./Drug/_02/full_pipeline/{filename}", index=False)

print(f"\n✅ 최종 앙상블 결과 저장 완료 → {filename}")
print(f"🧪 랜덤 시드 : {r}")
