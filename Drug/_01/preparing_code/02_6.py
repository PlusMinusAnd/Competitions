'''from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np

# 데이터 로드 (예시)
df = pd.read_csv('./Drug/data/features_rdkit.csv')

# 결측치 개수 확인
missing_values = df.isnull().sum()

# 결측치가 있는 컬럼만 필터링
missing_values = missing_values[missing_values > 0]

print("결측치가 있는 컬럼과 개수:")
print(missing_values)'''

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# 시각화
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))  # 상위 20개 피처
plt.title('Top 20 Feature Importances')
plt.tight_layout()

# 그래프 저장
plt.savefig('./Drug/feature_importance.png', dpi=300)

plt.close()  # 메모리 절약을 위해 창 닫기
