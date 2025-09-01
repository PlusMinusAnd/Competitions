##### DNN 데이터 제작 #####

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

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

# Inhibition 결합
train_desc_df['Inhibition'] = train_df['Inhibition'].values

# 결측치 처리 (평균으로 대체)
train_desc_df = train_desc_df.fillna(train_desc_df.mean())
test_desc_df = test_desc_df.fillna(train_desc_df.mean())  # 학습 평균 기준으로 처리

# 저장
train_desc_df.to_csv("./Drug/_02/0_dataset/train_descriptor.csv", index=False)
test_desc_df.to_csv("./Drug/_02/0_dataset/test_descriptor.csv", index=False)

print("✅ RDKit descriptor 생성 및 결측치 처리 완료")


