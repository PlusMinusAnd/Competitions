##### 01_feature_gen.py #####
"""
RDKit descriptor + Fingerprint + Graph + PyCaret용 데이터 생성 파이프라인
"""
import os
import numpy as np
import pandas as pd
import random
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import DataStructs
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib

### 고정 랜덤 시드 설정 ###
r = 394
random.seed(r)
np.random.seed(r)

### 경로 설정 ###
train_path = './Drug/train.csv'
test_path = './Drug/test.csv'
save_dir = './Drug/_02/full_pipeline/0_dataset'
os.makedirs(save_dir, exist_ok=True)

### 데이터 로드 ###
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

### SMILES -> Mol ###
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

### descriptor 이름 ###
descriptor_names = [desc[0] for desc in Descriptors._descList]
descriptor_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

### RDKit descriptor 계산 함수 ###
def calc_rdkit_descriptors(mol):
    if mol is None:
        return [np.nan] * len(descriptor_names)
    return list(descriptor_calc.CalcDescriptors(mol))

### Fingerprint 계산 함수 ###
def mol_to_fp(mol, radius=2, nBits=1024):
    if mol is None:
        return np.zeros(nBits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    arr = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def mol_to_maccs(mol):
    if mol is None:
        return np.zeros(167)
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

### 전체 계산 ###
print("📦 RDKit descriptor + Fingerprint 계산 중...")
train_desc = np.array([calc_rdkit_descriptors(mol) for mol in train_df['mol']])
test_desc  = np.array([calc_rdkit_descriptors(mol) for mol in test_df['mol']])
train_fp   = np.array([mol_to_fp(mol) for mol in train_df['mol']])
test_fp    = np.array([mol_to_fp(mol) for mol in test_df['mol']])
train_maccs= np.array([mol_to_maccs(mol) for mol in train_df['mol']])
test_maccs = np.array([mol_to_maccs(mol) for mol in test_df['mol']])

# 결합
X_train_full = np.concatenate([train_desc, train_fp, train_maccs], axis=1)
X_test_full  = np.concatenate([test_desc, test_fp, test_maccs], axis=1)

# 결측치 제거
print("🧹 결측치 제거 (dropna)...")
mask = ~np.isnan(X_train_full).any(axis=1)
X_train_full = X_train_full[mask]
y_train = train_df.loc[mask, 'Inhibition'].values

# 저장
np.save(f"{save_dir}/X_train_full.npy", X_train_full)
np.save(f"{save_dir}/X_test_full.npy", X_test_full)
np.save(f"{save_dir}/y_train.npy", y_train)

### 수치형/범주형 분리 위한 OneHotEncoder는 DNN 단계에서 적용 ###

print("✅ 수치형 + fingerprint + descriptor 데이터 저장 완료!")

### 다음 단계: 02_train_dnn.py 로 이동 ###

# ────────────────────────────────
# Graph 생성 (RDKit Mol → DGL)
# ────────────────────────────────
from rdkit.Chem import rdmolops
from dgl import graph as dgl_graph, add_self_loop
from rdkit import Chem
import torch
import joblib
import dgl
import os

def mol_to_graph(mol):
    num_atoms = mol.GetNumAtoms()
    atom_feats = []
    for atom in mol.GetAtoms():
        feat = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetTotalNumHs(),
            atom.GetImplicitValence(),
            int(atom.GetIsAromatic())
        ]
        atom_feats.append(torch.tensor(feat, dtype=torch.float))

    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append((i, j))
        edges.append((j, i))

    g = dgl.graph(edges, num_nodes=num_atoms)
    g = add_self_loop(g)
    g.ndata['h'] = torch.stack(atom_feats)
    return g

# SMILES → Mol
train_df['mol'] = train_df['Canonical_Smiles'].apply(Chem.MolFromSmiles)
test_df['mol'] = test_df['Canonical_Smiles'].apply(Chem.MolFromSmiles)

train_graphs = [mol_to_graph(mol) for mol in train_df['mol'] if mol is not None]
test_graphs = [mol_to_graph(mol) for mol in test_df['mol'] if mol is not None]

os.makedirs("./Drug/_02/full_pipeline/0_dataset", exist_ok=True)
joblib.dump(train_graphs, "./Drug/_02/full_pipeline/0_dataset/train_graphs.pkl")
joblib.dump(test_graphs, "./Drug/_02/full_pipeline/0_dataset/test_graphs.pkl")
print("✅ GNN 그래프 데이터 저장 완료")
