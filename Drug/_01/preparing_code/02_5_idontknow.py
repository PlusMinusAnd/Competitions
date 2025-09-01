from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np

# 1. SMILES → Descriptor 변환 함수
def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    descriptor_names = [desc[0] for desc in Descriptors.descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    descriptors = calculator.CalcDescriptors(mol)
    return pd.Series(descriptors, index=descriptor_names)

# 2. 데이터 로드
train_csv = pd.read_csv('./Drug/train.csv', index_col=0)
smiles_list = train_csv['Canonical_Smiles'].tolist()
inhibition = train_csv['Inhibition'].tolist()

# 3. Descriptor 변환
descriptor_list = []
for smiles in smiles_list:
    descriptor = smiles_to_descriptors(smiles)
    descriptor_list.append(descriptor)

descriptor_df = pd.DataFrame(descriptor_list)

# 4. 최종 데이터프레임 구성
# index + SMILES + descriptors + inhibition
final_df = descriptor_df.copy()
final_df.insert(0, 'SMILES', smiles_list)      # SMILES 컬럼 추가
final_df['Inhibition'] = inhibition            # Inhibition 컬럼 추가

# 인덱스 정리 (원래 인덱스 번호로)
final_df.index.name = 'index'
final_df.reset_index(inplace=True)

# 5. 저장
final_df.to_csv('./Drug/data/features_rdkit.csv', index=False)

print(final_df.head())
