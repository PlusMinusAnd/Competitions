##### GNN 데이터 제작 (고도화 버전) #####
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
import dgl
from rdkit import Chem
from rdkit.Chem import Descriptors, rdchem
from rdkit.Chem import GetPeriodicTable
import random, os
import joblib

# 시드 고정
r = 73
random.seed(r)
np.random.seed(r)
torch.manual_seed(r)

# 경로 설정
train_path = "./Drug/train.csv"
test_path = "./Drug/test.csv"
save_dir = "./Drug/_02/0_dataset"
os.makedirs(save_dir, exist_ok=True)

# 공유 반지름 (Å) + 전기음성도 (Pauling scale)
COVALENT_RADII = {
    1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39
}
ELECTRONEGATIVITY = {
    1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66
}


##### 1. 데이터 로딩 #####
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

##### 2. Mol 변환 함수 (예외 처리 포함) #####
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

# 실패 로그 저장
train_df[train_df['mol'].isnull()].to_csv(f"{save_dir}/invalid_train_mol.csv", index=False)
test_df[test_df['mol'].isnull()].to_csv(f"{save_dir}/invalid_test_mol.csv", index=False)

##### 3. 피처 정의 #####
ptable = GetPeriodicTable()

# 공유 반지름 매핑
COVALENT_RADII = {
    1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39
}

# one-hot 유연화
def one_hot_encoding(value, choices):
    return [int(value == c) for c in choices]

# 원자 피처 (안정형)
def atom_features(atom):
    atomic_num_choices = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
    hybridization_choices = [
        rdchem.HybridizationType.SP,
        rdchem.HybridizationType.SP2,
        rdchem.HybridizationType.SP3,
        rdchem.HybridizationType.SP3D,
        rdchem.HybridizationType.SP3D2,
    ]

    atomic_num = atom.GetAtomicNum()
    electronegativity = ELECTRONEGATIVITY.get(atomic_num, 0.0)
    covalent_radius = COVALENT_RADII.get(atomic_num, 1.0)

    features = []
    features += one_hot_encoding(atomic_num, atomic_num_choices)
    features.append(atom.GetDegree())
    features.append(atom.GetFormalCharge())
    features.append(atom.GetTotalNumHs())
    features += one_hot_encoding(atom.GetHybridization(), hybridization_choices)
    features.append(int(atom.GetIsAromatic()))
    features.append(atom.GetImplicitValence())
    features.append(atom.GetTotalValence()) 
    features.append(atom.GetNumRadicalElectrons())
    features.append(int(atom.IsInRing()))
    features.append(covalent_radius)
    features.append(electronegativity)

    return torch.tensor(features, dtype=torch.float)


# 결합 피처
def bond_features(bond):
    stereo_choices = list(rdchem.BondStereo.values)
    stereo_encoded = one_hot_encoding(bond.GetStereo(), stereo_choices)
    return torch.tensor([
        bond.GetBondTypeAsDouble(),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
        *stereo_encoded
    ], dtype=torch.float)

##### 4. Mol → DGLGraph 변환 #####
def mol_to_dgl_graph(mol):
    num_atoms = mol.GetNumAtoms()
    atom_feats = [atom_features(mol.GetAtomWithIdx(i)) for i in range(num_atoms)]
    
    srcs, dsts, bond_feats = [], [], []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        srcs += [u, v]
        dsts += [v, u]
        b_feat = bond_features(bond)
        bond_feats += [b_feat, b_feat]

    g = dgl.graph((srcs, dsts), num_nodes=num_atoms)
    g.ndata['h'] = torch.stack(atom_feats)
    g.edata['e'] = torch.stack(bond_feats)
    return g

##### 5. 순차 변환 및 저장 #####
train_graphs = [mol_to_dgl_graph(mol) for mol in train_df['mol'] if mol is not None]
test_graphs = [mol_to_dgl_graph(mol) for mol in test_df['mol'] if mol is not None]

joblib.dump(train_graphs, f"{save_dir}/train_graphs.pkl")
joblib.dump(test_graphs, f"{save_dir}/test_graphs.pkl")

##### 6. 분자 전체 특성 저장 (선택) #####
train_df['MolWt'] = train_df['mol'].apply(lambda m: Descriptors.MolWt(m) if m else np.nan)
train_df['logP'] = train_df['mol'].apply(lambda m: Descriptors.MolLogP(m) if m else np.nan)
train_df[['Canonical_Smiles', 'MolWt', 'logP']].to_csv(f"{save_dir}/train_mol_features.csv", index=False)

print("✅ GNN 그래프 변환 및 저장 완료!")
