import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import datetime

r = 73

# 점수 출력 함수
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
    return score

# 데이터 불러오기
y = pd.read_csv("./Drug/train.csv")["Inhibition"].values
dmpnn_oof = np.load("./Drug/_02/2_npy/pre_dmpnn_oof.npy")
boost_oof = np.load("./Drug/_02/2_npy/boost_oof.npy")
dmpnn_preds = np.load("./Drug/_02/2_npy/pre_dmpnn_preds.npy")
boost_preds = np.load("./Drug/_02/2_npy/boost_preds.npy")
test_id = pd.read_csv("./Drug/test.csv")["ID"]

# 앙상블 최적 alpha 찾기
alphas = np.linspace(0, 1, 21)
best_score = -np.inf
best_alpha = 0.5

for alpha in alphas:
    final_oof = alpha * dmpnn_oof + (1 - alpha) * boost_oof
    score = print_scores(y, final_oof, label=f"α={alpha:.2f}")
    if score > best_score:
        best_score = score
        best_alpha = alpha

# 최종 앙상블 예측
final_preds = best_alpha * dmpnn_preds + (1 - best_alpha) * boost_preds
final_oof = best_alpha * dmpnn_oof + (1 - best_alpha) * boost_oof

# 점수 출력
print_scores(y, final_oof, label=f"Final Ensemble α={best_alpha:.2f}")

# 저장
now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
filename = f"submission_{r}_final_ensemble({now}).csv"
save_path = f"./Drug/_02/3_submission/{filename}"

submission = pd.DataFrame({
    "ID": test_id,
    "Inhibition": final_preds
})
submission.to_csv(save_path, index=False)
print(f"✅ 최종 앙상블 파일 저장 완료 → {filename}")


# submission_73_final_ensemble(20250704_2145).csv
# 📊 Final Ensemble α=0.10 결과
# RMSE     : 23.55323
# NRMSE    : 0.23700
# Pearson  : 0.45665
# Score📈  : 0.60982
# r = 73