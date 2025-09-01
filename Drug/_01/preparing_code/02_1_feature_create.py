import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

# SMILES → 파생 피처 추출 함수
def extract_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * 10  # 잘못된 SMILES에 대한 예외 처리
    return [
        Descriptors.MolWt(mol),                        # 분자량
        Descriptors.MolLogP(mol),                      # 소수성 logP
        Descriptors.NumHDonors(mol),                   # 수소 공여자 수
        Descriptors.NumHAcceptors(mol),                # 수소 수용자 수
        Descriptors.TPSA(mol),                         # 극성 표면적
        Descriptors.NumRotatableBonds(mol),            # 회전 가능한 결합 수
        Descriptors.NumAromaticRings(mol),             # 방향족 고리 수
        Descriptors.NumAliphaticRings(mol),            # 지방족 고리 수
        Descriptors.FractionCSP3(mol),                 # sp3 탄소 비율
        sum(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024))  # 서브구조 수
    ]

# CSV 파일 불러오기
df = pd.read_csv('./_data/dacon/drug/train.csv')
df_test = pd.read_csv('./_data/dacon/drug/test.csv')


# Canonical_Smiles 컬럼에서 파생 피처 추출
smiles_list = df.iloc[:, 1]  # 또는 df["Canonical_Smiles"]
smiles_list_test = df_test.iloc[:, 1]  # 또는 df["Canonical_Smiles"]
features = [extract_features(s) for s in smiles_list]
features_test = [extract_features(s) for s in smiles_list_test]

# 데이터프레임으로 변환
feature_df = pd.DataFrame(features, columns=[
    'MolWt', 'LogP', 'HDonors', 'HAcceptors', 'TPSA',
    'RotBonds', 'AromaticRings', 'AliphaticRings', 'CSP3', 'MorganSubstructs'
])
feature_df_test = pd.DataFrame(features_test, columns=[
    'MolWt', 'LogP', 'HDonors', 'HAcceptors', 'TPSA',
    'RotBonds', 'AromaticRings', 'AliphaticRings', 'CSP3', 'MorganSubstructs'
])

# 원본 데이터와 합치기 (선택사항)
# result_df = pd.concat([df, feature_df], axis=1)

# 저장
feature_df.to_csv('./_data/dacon/drug/data/derived_features.csv', index=False)
feature_df_test.to_csv('./_data/dacon/drug/data/derived_features_test.csv', index=False)

print("✅ 파생 피처 저장 완료: ./_data/dacon/drug/derived_features.csv")
