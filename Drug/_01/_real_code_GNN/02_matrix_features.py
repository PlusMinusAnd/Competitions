from rdkit import Chem
import pandas as pd
import numpy as np
import os


# 🔥 데이터 로드
train = pd.read_csv('./Drug/train.csv')
test = pd.read_csv('./Drug/test.csv')

train_smiles = train['Canonical_Smiles'].tolist()
test_smiles = test['Canonical_Smiles'].tolist()

# 전체 smiles
all_smiles = train_smiles + test_smiles


# ✅ 모든 분자의 atom 수 계산
def get_atom_num(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return mol.GetNumAtoms()

atom_nums = [get_atom_num(s) for s in all_smiles]
max_atom_num = max(atom_nums)

print(f"Max atom number: {max_atom_num}")



# ✅ 분자 → (adjacency, bond strength) → (max_atom, max_atom, 2) 패딩 포함
def mol_to_padded_matrix(smiles, max_size):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((max_size, max_size, 2))
    
    N = mol.GetNumAtoms()

    # 인접 행렬
    adjacency = Chem.GetAdjacencyMatrix(mol)

    # 결합 강도 행렬
    bond_strength = np.zeros((N, N))
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()  # 단일=1, 이중=2, 삼중=3, 방향족=1.5
        bond_strength[i, j] = bond_type
        bond_strength[j, i] = bond_type

    # (N, N, 2)로 쌓기
    combined = np.stack([adjacency, bond_strength], axis=2)

    # 중앙 패딩
    padded = np.zeros((max_size, max_size, 2))

    start = (max_size - N) // 2
    end = start + N

    padded[start:end, start:end, :] = combined

    return padded



# ✅ Train 변환
train_matrices = np.array([mol_to_padded_matrix(s, max_atom_num) for s in train_smiles])

# ✅ Test 변환
test_matrices = np.array([mol_to_padded_matrix(s, max_atom_num) for s in test_smiles])

print(f"Train shape: {train_matrices.shape}")
print(f"Test shape: {test_matrices.shape}")



# ✅ 저장

np.save('./Drug/_npy_data/train_graph.npy', train_matrices)
np.save('./Drug/_npy_data/test_graph.npy', test_matrices)

print("✔️ 저장 완료: ./Drug/_npy_data/train_graph.npy, test_graph.npy")
