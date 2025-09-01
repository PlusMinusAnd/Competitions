from rdkit import Chem
import numpy as np
import pandas as pd
import os

# 🔥 CSV 파일 경로
train_csv_path = './Drug/train.csv'
test_csv_path = './Drug/test.csv'

# 🔥 데이터 로드
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# 🔥 저장 폴더 생성
base_path = './Drug/_npy_data/'
os.makedirs(os.path.join(base_path, 'adjacency_matrices'), exist_ok=True)
os.makedirs(os.path.join(base_path, 'bond_strength_matrices'), exist_ok=True)
os.makedirs(os.path.join(base_path, 'combined_matrices'), exist_ok=True)

# 🔥 함수: SMILES → 행렬 저장
def process_and_save(smiles_list, prefix):
    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"❌ Invalid SMILES at index {idx}: {smiles}")
            continue
        
        N = mol.GetNumAtoms()

        # ✅ 결합 여부 (인접 행렬)
        adjacency = Chem.GetAdjacencyMatrix(mol)

        # ✅ 결합 강도 행렬
        bond_strength = np.zeros((N, N))
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = bond.GetBondTypeAsDouble()  # 단일=1, 이중=2, 삼중=3, 방향족=1.5
            bond_strength[i, j] = bond_type
            bond_strength[j, i] = bond_type  # 대칭 행렬

        # ✅ 결합 여부 + 강도 (N, N, 2) 형태
        combined_matrix = np.stack([adjacency, bond_strength], axis=2)

        # 🔥 파일명 지정
        name = f"{prefix}_{idx}"

        # 🔥 npy 파일로 저장
        np.save(os.path.join(base_path, 'adjacency_matrices', f'{name}.npy'), adjacency)
        np.save(os.path.join(base_path, 'bond_strength_matrices', f'{name}.npy'), bond_strength)
        np.save(os.path.join(base_path, 'combined_matrices', f'{name}.npy'), combined_matrix)

        print(f"✔️ Saved {name}")

# 🔥 Train 데이터 처리
process_and_save(train_df['Canonical_Smiles'], 'train')

# 🔥 Test 데이터 처리
process_and_save(test_df['Canonical_Smiles'], 'test')

print("✔️ 모든 파일 저장 완료!")
