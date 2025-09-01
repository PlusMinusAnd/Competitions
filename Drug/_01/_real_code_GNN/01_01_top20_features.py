from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 🔥 데이터 로드
train = pd.read_csv('./Drug/train.csv')
test = pd.read_csv('./Drug/test.csv')

train['Inhibition'] = train['Inhibition']
test['Inhibition'] = np.nan  # 테스트는 Inhibition 없음

# 🔥 SMILES 리스트
train_smiles = train['Canonical_Smiles'].tolist()
test_smiles = test['Canonical_Smiles'].tolist()

# 전체 SMILES
all_smiles = train_smiles + test_smiles


# ✅ RDKit descriptor 계산 함수
descriptor_names = [desc[0] for desc in Descriptors.descList]
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

def smiles_to_descriptor(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * len(descriptor_names)
    return calculator.CalcDescriptors(mol)


# ✅ 모든 descriptor 계산
descriptor_data = [smiles_to_descriptor(s) for s in all_smiles]
descriptor_df = pd.DataFrame(descriptor_data, columns=descriptor_names)

# ✅ index 및 SMILES 정보 추가
descriptor_df['index'] = list(range(len(all_smiles)))
descriptor_df['SMILES'] = all_smiles

# ✅ 타겟 (Inhibition) 추가
all_inhibition = pd.concat([train['Inhibition'], test['Inhibition']], ignore_index=True)
descriptor_df['Inhibition'] = all_inhibition


# ✅ 🔥 Feature Importance 기반 상위 20개 선택
# → NaN 있는 descriptor 제거
feature_df = descriptor_df.drop(columns=['index', 'SMILES', 'Inhibition'])
feature_df = feature_df.dropna(axis=1)

# train 데이터만으로 importance 계산
train_features = feature_df.iloc[:len(train)]
train_target = descriptor_df['Inhibition'].iloc[:len(train)]

# 모델 학습
model = RandomForestRegressor(n_estimators=500, random_state=42)
model.fit(train_features, train_target)

# Feature importance 계산
importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': train_features.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# 상위 20개 선택
top20_features = importance_df['Feature'].head(20).tolist()

print(f"상위 20개 피처:\n{top20_features}")


# ✅ 🔗 상관관계 기반 중복 피처 제거
corr_matrix = feature_df[top20_features].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

threshold = 0.85
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

print(f"상관관계 기준 제거할 피처:\n{to_drop}")

# 최종 피처 목록
final_features = [f for f in top20_features if f not in to_drop]
print(f"최종 선택된 피처:\n{final_features}")


# ✅ 최종 데이터프레임 구성
final_df = descriptor_df[['index', 'SMILES'] + final_features + ['Inhibition']]

# 🔥 Train/Test 분리 저장
final_train = final_df.iloc[:len(train)]
final_test = final_df.iloc[len(train):]

os.makedirs('./Drug/final_data', exist_ok=True)

final_train.to_csv('./Drug/_engineered_data/train_final.csv', index=False)
final_test.to_csv('./Drug/_engineered_data/test_final.csv', index=False)

print("✔️ 저장 완료: ./Drug/_engineered_data/")
