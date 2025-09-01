import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import itertools

# 🔹 파일 경로
gat_oof_path = "./Drug/_02/full_pipeline/gat_oof.npy"
gin_oof_path = "./Drug/_02/full_pipeline/gin_oof.npy"
boost_oof_path = "./Drug/_02/full_pipeline/boost_oof.npy"
train_csv_path = "./Drug/train.csv"

# 🔹 로드
y_true = pd.read_csv(train_csv_path)["Inhibition"].values
gat_oof = np.load(gat_oof_path)
gin_oof = np.load(gin_oof_path)
boost_oof = np.load(boost_oof_path)

# 🔹 점수 계산 함수
def compute_score(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    pearson = pearsonr(y_true, y_pred)[0]
    score = 0.5 * (1 - min(nrmse, 1)) + 0.5 * np.clip(pearson, 0, 1)
    return rmse, nrmse, pearson, score

# 🔹 α(GAT), β(GIN), γ(Boosting) 조합 생성 (합 = 1.0)
alphas = np.linspace(0, 1, 11)
best_score = -np.inf
best_weights = (0, 0, 1)

print("🔍 Searching best α(GAT), β(GIN), γ(Boosting) weights...")
for a in alphas:
    for b in alphas:
        c = 1.0 - a - b
        if c < 0 or c > 1: continue  # 합이 1이 넘어가면 skip
        blended = a * gat_oof + b * gin_oof + c * boost_oof
        rmse, nrmse, pearson, score = compute_score(y_true, blended)
        if score > best_score:
            best_score = score
            best_weights = (a, b, c)
        print(f"α(GAT)={a:.2f}, β(GIN)={b:.2f}, γ(Boost)={c:.2f} | Score📈={score:.5f} | RMSE={rmse:.5f} | Pearson={pearson:.5f}")

# 🔹 최종 결과 출력
a, b, c = best_weights
print("\n✅ Best Weights Found")
print(f"→ α(GAT)={a:.2f}, β(GIN)={b:.2f}, γ(Boost)={c:.2f}")
final_blend = a * gat_oof + b * gin_oof + c * boost_oof
rmse, nrmse, pearson, score = compute_score(y_true, final_blend)
print(f"📊 Score📈={score:.5f}, RMSE={rmse:.5f}, Pearson={pearson:.5f}")
