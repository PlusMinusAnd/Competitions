import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# 파일 경로
gin_oof_path = "./Drug/_02/full_pipeline/gin_oof.npy"
gat_oof_path = "./Drug/_02/full_pipeline/gat_oof.npy"
train_csv_path = "./Drug/train.csv"

# 데이터 로드
y_true = pd.read_csv(train_csv_path)["Inhibition"].values
gin_oof = np.load(gin_oof_path)
gat_oof = np.load(gat_oof_path)

# 성능 측정 함수
def compute_score(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    pearson = pearsonr(y_true, y_pred)[0]
    score = 0.5 * (1 - min(nrmse, 1)) + 0.5 * np.clip(pearson, 0, 1)
    return rmse, nrmse, pearson, score

# α 조합 탐색
alphas = np.linspace(0, 1, 21)
results = []

for alpha in alphas:
    blended = alpha * gat_oof + (1 - alpha) * gin_oof
    rmse, nrmse, pearson, score = compute_score(y_true, blended)
    results.append((alpha, rmse, nrmse, pearson, score))
    print(f"α={alpha:.2f} | RMSE={rmse:.5f} | NRMSE={nrmse:.5f} | Pearson={pearson:.5f} | Score📈={score:.5f}")

# 최적 조합 출력
results = sorted(results, key=lambda x: -x[4])  # score 기준 정렬
best_alpha, best_rmse, best_nrmse, best_pearson, best_score = results[0]
print(f"\n✅ 최적 α: {best_alpha:.2f}")
print(f"📊 Best Score📈: {best_score:.5f} | RMSE: {best_rmse:.5f} | NRMSE: {best_nrmse:.5f} | Pearson: {best_pearson:.5f}")

# 시각화
alphas_plot = [r[0] for r in results]
scores_plot = [r[4] for r in results]

plt.figure(figsize=(8,5))
plt.plot(alphas_plot, scores_plot, marker='o')
plt.title("GAT-GIN 앙상블 Score📈 vs α")
plt.xlabel("α (GAT 비중)")
plt.ylabel("Score📈")
plt.grid(True)
plt.tight_layout()
plt.show()
