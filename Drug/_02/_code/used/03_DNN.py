##### DNN 데이터 제작 #####


import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import random

def set_seed(seed=73):
    np.random.seed(seed)
    random.seed(seed)

set_seed()

# 파일 경로
train_path = "./Drug/train.csv"
test_path = "./Drug/test.csv"

# 데이터 로드
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# SMILES to Mol
def smiles_to_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            Chem.SanitizeMol(mol)
        return mol
    except:
        return None

train_df['mol'] = train_df['Canonical_Smiles'].apply(smiles_to_mol)
test_df['mol'] = test_df['Canonical_Smiles'].apply(smiles_to_mol)

# Descriptor 이름 정의
descriptor_names = [desc[0] for desc in Descriptors._descList]
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

# Mol → descriptor
def compute_descriptors(mol):
    if mol is None:
        return [np.nan] * len(descriptor_names)
    return list(calculator.CalcDescriptors(mol))

# 계산
train_desc = np.array([compute_descriptors(mol) for mol in train_df['mol']])
test_desc = np.array([compute_descriptors(mol) for mol in test_df['mol']])

# DataFrame 변환
train_desc_df = pd.DataFrame(train_desc, columns=descriptor_names)
test_desc_df = pd.DataFrame(test_desc, columns=descriptor_names)
train_desc_df['Inhibition'] = train_df['Inhibition'].values

# 1. 상수열 제거
stds = train_desc_df.drop(columns="Inhibition").std()
non_constant_cols = stds[stds > 0].index.tolist()
train_desc_df = train_desc_df[non_constant_cols + ['Inhibition']]
test_desc_df = test_desc_df[non_constant_cols]

# 2. KNN Imputer
imputer = KNNImputer(n_neighbors=5)
X_full = pd.concat([train_desc_df.drop(columns="Inhibition"), test_desc_df])
X_imputed = imputer.fit_transform(X_full)
X_train = X_imputed[:len(train_desc_df)]
X_test = X_imputed[len(train_desc_df):]

# 3. 상관관계 높은 피처 제거
X_df = pd.DataFrame(X_train, columns=non_constant_cols)
corr_matrix = X_df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X_df = X_df.drop(columns=to_drop)
X_test_df = pd.DataFrame(X_test, columns=non_constant_cols).drop(columns=to_drop)

# 4. 클리핑 (이상값 완화)
lower = X_df.quantile(0.01)
upper = X_df.quantile(0.99)
X_df = X_df.clip(lower=lower, upper=upper, axis=1)
X_test_df = X_test_df.clip(lower=lower, upper=upper, axis=1)

# 5. 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_df)
X_test_scaled = scaler.transform(X_test_df)

# 6. 최종 DataFrame 구성 및 저장
final_train = pd.DataFrame(X_train_scaled, columns=X_df.columns)
final_test = pd.DataFrame(X_test_scaled, columns=X_df.columns)
final_train['Inhibition'] = train_df['Inhibition'].values

final_train.to_csv("./Drug/_02/0_dataset/train_descriptor.csv", index=False)
final_test.to_csv("./Drug/_02/0_dataset/test_descriptor.csv", index=False)

print("✅ RDKit descriptor 전처리 및 저장 완료")
