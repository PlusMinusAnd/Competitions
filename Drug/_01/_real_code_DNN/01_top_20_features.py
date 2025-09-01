from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np

# 파일 경로
train_path = './Drug/train.csv'  
test_path = './Drug/test.csv'    

# CSV 파일 로드
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# RDKit descriptor 목록
descriptor_names = [desc[0] for desc in Descriptors._descList]
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

# SMILES → descriptor 변환 함수
def smiles_to_features(smiles_list):
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            desc = calculator.CalcDescriptors(mol)
        else:
            desc = [np.nan] * len(descriptor_names)
        features.append(desc)
    return pd.DataFrame(features, columns=descriptor_names)

# Feature 생성
train_features = smiles_to_features(train_df.iloc[:, 1])  # 2열: SMILES
test_features = smiles_to_features(test_df.iloc[:, 1])

# 데이터 결합
train_final = pd.concat([train_df.iloc[:, [0, 1]], train_features, train_df['Inhibition']], axis=1)
test_final = pd.concat([test_df.iloc[:, [0, 1]], test_features], axis=1)

# 저장
train_final.to_csv('./Drug/_engineered_data_DNN/train_with_rdkit_features.csv', index=False)
test_final.to_csv('./Drug/_engineered_data_DNN/test_with_rdkit_features.csv', index=False)
