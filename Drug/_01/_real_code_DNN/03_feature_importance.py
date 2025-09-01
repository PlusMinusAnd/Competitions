import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# 경로 설정
train_path = './Drug/_engineered_data_DNN/train_filled_dnn.csv'
test_path = './Drug/_engineered_data_DNN/test_with_rdkit_features.csv'
save_path = './Drug/_engineered_data_DNN/'

# 데이터 로드
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# 수치형 피처만 추출 (ID, SMILES 제외)
X_train = train.select_dtypes(include=[float, int]).drop(columns=['Inhibition'])
y_train = train['Inhibition']

# 🔹 1. Feature Importance (RandomForest)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

importances = pd.Series(model.feature_importances_, index=X_train.columns)
importances_sorted = importances.sort_values(ascending=False)

# 저장: Feature Importance Bar Plot
plt.figure(figsize=(10, 12))
importances_sorted.plot(kind='barh')
plt.title("Feature Importance (RandomForest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{save_path}feature_importance_full.png")
plt.close()

# 🔹 2. Correlation Heatmap (Train + Test)
all_features = pd.concat([
    train.select_dtypes(include=[float, int]).drop(columns=['Inhibition']),
    test.select_dtypes(include=[float, int])
], axis=0, ignore_index=True)

corr = all_features.corr()

plt.figure(figsize=(18, 16))
sns.heatmap(corr, cmap='coolwarm', square=True, xticklabels=False, yticklabels=False)
plt.title("Feature Correlation Heatmap (Train + Test)")
plt.tight_layout()
plt.savefig(f"{save_path}feature_correlation_heatmap.png")
plt.close()
