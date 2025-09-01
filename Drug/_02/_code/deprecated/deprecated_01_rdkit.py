##### GNN 데이터 제작 #####
import warnings
warnings.filterwarnings('ignore')

###### 1. 데이터 호출 ######
import pandas as pd

# 파일 경로
train_path = "./Drug/train.csv"
test_path = "./Drug/test.csv"

# 데이터 불러오기
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 열 이름과 앞부분 미리보기
train_columns = train_df.columns.tolist()
test_columns = test_df.columns.tolist()

train_head = train_df.head()
test_head = test_df.head()

(train_columns, test_columns, train_head, test_head)

###### 2. 분자 데이터 graph 변환 ######

from rdkit import Chem
from rdkit.Chem import rdmolops

# SMILES → RDKit Mol 객체 변환 함수
def smiles_to_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            Chem.SanitizeMol(mol)
        return mol
    except:
        return None

# train과 test의 SMILES → Mol 변환
train_df['mol'] = train_df['Canonical_Smiles'].apply(smiles_to_mol)
test_df['mol'] = test_df['Canonical_Smiles'].apply(smiles_to_mol)

# 유효한 Mol의 개수 확인
valid_train_mol_count = train_df['mol'].notnull().sum()
valid_test_mol_count = test_df['mol'].notnull().sum()

valid_train_mol_count, valid_test_mol_count

##### 3. 그래프 데이터로 변환 후 저장 #####

from dgl import DGLGraph
import dgl
import torch
import numpy as np

# 난수 시드 고정
import random
random.seed(73)
np.random.seed(73)
torch.manual_seed(73)

# 원자 특성 추출 함수 (간단 버전)
# 1. 향상된 atom_features + 2. one-hot 인코딩 버전 적용
def one_hot_encoding(value, choices):
    return [int(value == c) for c in choices]

def atom_features(atom):
    atomic_num_choices = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
    hybridization_choices = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]

    features = []
    features += one_hot_encoding(atom.GetAtomicNum(), atomic_num_choices)
    features.append(atom.GetDegree())
    features.append(atom.GetFormalCharge())
    features.append(atom.GetTotalNumHs())
    features += one_hot_encoding(atom.GetHybridization(), hybridization_choices)
    features.append(int(atom.GetIsAromatic()))
    features.append(atom.GetImplicitValence())
    features.append(atom.GetTotalValence()) 
    features.append(atom.GetNumRadicalElectrons())
    features.append(int(atom.IsInRing()))
    return torch.tensor(features, dtype=torch.float)



# 결합 특성 추출 함수
def bond_features(bond):
    return torch.tensor([
        bond.GetBondTypeAsDouble(),      # 결합 차수
        int(bond.GetIsConjugated()),     # 공명 여부
        int(bond.IsInRing())             # 고리 구조 여부
    ], dtype=torch.float)

# Mol → DGLGraph 변환 함수
def mol_to_dgl_graph(mol):
    g = dgl.DGLGraph()
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    atom_feats = [atom_features(mol.GetAtomWithIdx(i)) for i in range(num_atoms)]
    g.ndata['h'] = torch.stack(atom_feats)

    srcs, dsts, bond_feats = [], [], []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # 방향성 메시지를 위해 양방향 추가
        srcs += [u, v]
        dsts += [v, u]
        bond_feat = bond_features(bond)
        bond_feats += [bond_feat, bond_feat]

    g.add_edges(srcs, dsts)
    g.edata['e'] = torch.stack(bond_feats)
    return g

# DGL 그래프 변환 및 저장
train_graphs = [mol_to_dgl_graph(mol) for mol in train_df['mol']]
test_graphs = [mol_to_dgl_graph(mol) for mol in test_df['mol']]

# DGLGraph는 직접 저장 불가 → 리스트 저장
import joblib
joblib.dump(train_graphs, "./Drug/_02/0_dataset/train_graphs.pkl")
joblib.dump(test_graphs, "./Drug/_02/0_dataset/test_graphs.pkl")

print('1. 완료')