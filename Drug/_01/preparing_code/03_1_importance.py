'''#######CYP3A4 억제와 기능기의 상관관계 분석 코드##########
# ======================= 라이브러리 =======================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ======================= 데이터 로드 =======================
data = pd.read_csv('./_data/dacon/drug/data/train_with_functional_group_counts.csv')

# ======================= 기능기 컬럼만 선택 =======================
feature_cols = [col for col in data.columns if col not in ['ID', 'Canonical_Smiles', 'Inhibition']]

# ======================= 상관 분석 =======================
corr = data[feature_cols + ['Inhibition']].corr()['Inhibition'].drop('Inhibition').sort_values(ascending=False)

# ======================= 결과 출력 =======================
print('CYP3A4 억제와 기능기 상관관계 (상위)')
print(corr)

# ======================= 시각화 =======================
plt.figure(figsize=(8, len(corr) * 0.5))
sns.barplot(x=corr.values, y=corr.index, palette='coolwarm')
plt.title('Correlation between Functional Groups and Inhibition (CYP3A4)')
plt.xlabel('Correlation')
plt.ylabel('Functional Group')
plt.tight_layout()
plt.show()




#############억제율 상위 20%와 하위 20% 비교###############

# 상위 20%, 하위 20% 그룹 나누기
top_20 = data.sort_values(by='Inhibition', ascending=False).head(int(len(data) * 0.2))
bottom_20 = data.sort_values(by='Inhibition', ascending=True).head(int(len(data) * 0.2))

# 각 그룹별 기능기 평균 개수 비교
top_mean = top_20[feature_cols].mean()
bottom_mean = bottom_20[feature_cols].mean()

# 차이 계산
diff = (top_mean - bottom_mean).sort_values(ascending=False)

# 출력
print('상위 20% - 하위 20% 기능기 차이')
print(diff)

# 시각화
plt.figure(figsize=(8, len(diff) * 0.5))
sns.barplot(x=diff.values, y=diff.index, palette='viridis')
plt.title('Functional Group Difference (Top 20% - Bottom 20%)')
plt.xlabel('Mean Count Difference')
plt.ylabel('Functional Group')
plt.tight_layout()
plt.show()


from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# 데이터 로드
df = pd.read_csv('./Drug/data/features_rdkit.csv')

# 독립 변수(X)와 종속 변수(y) 분리
X = df.drop(['index', 'SMILES', 'Inhibition'], axis=1)
y = df['Inhibition']

# 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 피처 임포턴스 추출
importances = model.feature_importances_
feature_names = X.columns

# 데이터프레임으로 정리
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(importance_df)

# CSV로 저장
importance_df.to_csv('./Drug/feature_importance_matrix.csv', index=False)'''


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
df = pd.read_csv('./Drug/data/features_rdkit.csv')

# 필요 없는 컬럼 제거
features = df.drop(['index', 'SMILES', 'Inhibition'], axis=1)

# 상관 행렬 계산 (피어슨 상관계수)
corr_matrix = features.corr(method='pearson')

# 상관 행렬 출력
print(corr_matrix)

# 히트맵으로 시각화
plt.figure(figsize=(16, 14))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt=".2f", square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.7})
plt.title('Feature Correlation Heatmap', fontsize=18)
plt.tight_layout()

# 이미지로 저장
plt.savefig('./Drug/feature_correlation_heatmap.png', dpi=300)
plt.close()
