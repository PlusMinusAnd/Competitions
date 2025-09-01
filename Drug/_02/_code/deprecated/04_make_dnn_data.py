# 04_make_dnn_data.py
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# 경로
train_path = "./Drug/train.csv"
test_path = "./Drug/test.csv"
save_dir = "./Drug/_02/full_pipeline/0_dataset"
os.makedirs(save_dir, exist_ok=True)

# 데이터 로딩
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
y = train['Inhibition'].values

# SMILES to Mol
train['mol'] = train['Canonical_Smiles'].apply(lambda x: Chem.MolFromSmiles(x))
test['mol'] = test['Canonical_Smiles'].apply(lambda x: Chem.MolFromSmiles(x))

# RDKit descriptor 계산
desc_names = [d[0] for d in Descriptors._descList]
desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

def get_desc(mol):
    if mol is None:
        return [np.nan] * len(desc_names)
    return list(desc_calc.CalcDescriptors(mol))

X_train = np.array([get_desc(m) for m in train['mol']])
X_test = np.array([get_desc(m) for m in test['mol']])

X_train_df = pd.DataFrame(X_train, columns=desc_names)
X_test_df = pd.DataFrame(X_test, columns=desc_names)

X_train_df['Inhibition'] = y

# 결측치 제거
X_train_df = X_train_df.dropna()
X_test_df = X_test_df.dropna()
y = X_train_df['Inhibition'].values
X_train_df = X_train_df.drop(columns=['Inhibition'])

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_df)
X_test_scaled = scaler.transform(X_test_df)

# 저장
train_final = pd.DataFrame(X_train_scaled, columns=X_train_df.columns)
train_final['Inhibition'] = y
train_final.to_csv(f"{save_dir}/train_descriptor.csv", index=False)

pd.DataFrame(X_test_scaled, columns=X_test_df.columns).to_csv(f"{save_dir}/test_descriptor.csv", index=False)
print("✅ DNN descriptor 데이터 저장 완료")
