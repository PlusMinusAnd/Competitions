# 기본
import os
import random
import datetime
import joblib
import warnings

# 연산
import numpy as np
import pandas as pd

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# DGL
import dgl
from dgl.nn.pytorch import GATConv

# RDKit
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem, rdchem
from rdkit.Chem import GetPeriodicTable
from rdkit.ML.Descriptors import MoleculeDescriptors

# Sklearn
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV

# 통계
from scipy.stats import pearsonr

# 모델
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings('ignore')

r = random.randint(1, 19)

# 결과 프린트
def print_scores(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    pearson = pearsonr(y_true, y_pred)[0]
    score = 0.5 * (1 - min(nrmse, 1)) + 0.5 * np.clip(pearson, 0, 1)
    print(f"📊 {label} 결과")
    print(f"RMSE     : {rmse:.5f}")
    print(f"NRMSE    : {nrmse:.5f}")
    print(f"Pearson  : {pearson:.5f}")
    print(f"Score    : {score:.5f}")
    print(f"Random   : {r}")
    return score


#================================================================
#========================= GNN 전처리 ===========================
#================================================================


# 파일 로드
train = pd.read_csv("./Drug/train.csv")
test = pd.read_csv("./Drug/test.csv")
DNN = pd.read_csv("./Drug/_04/_datasets/DNN.csv")

##### 2. Mol 변환 함수 (예외 처리 포함) #####
def smiles_to_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            Chem.SanitizeMol(mol)
        return mol
    except:
        return None

train['mol'] = train['Canonical_Smiles'].apply(smiles_to_mol)
test['mol'] = test['Canonical_Smiles'].apply(smiles_to_mol)

##### 3. 피처 정의 #####
ptable = GetPeriodicTable()

# one-hot 유연화
def one_hot_encoding(value, choices):
    return [int(value == c) for c in choices]

# 공유 반지름 (Å) + 전기음성도 (Pauling scale)
COVALENT_RADII = {
    1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39
}
ELECTRONEGATIVITY = {
    1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66
}

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

    # 그래프 생성 (원래 엣지 기준)
    g = dgl.graph((srcs, dsts), num_nodes=num_atoms)

    # ✅ self-loop 추가
    g = dgl.add_self_loop(g)

    # ✅ self-loop에 대응하는 dummy bond feature 추가
    num_edges = g.num_edges()
    num_bond_feats = len(bond_feats)

    if num_edges > num_bond_feats:
        # 하나의 dummy bond feature는 기존 것과 같은 shape을 가져야 함
        dummy_bond = torch.zeros_like(bond_feats[0])
        for _ in range(num_edges - num_bond_feats):
            bond_feats.append(dummy_bond)

    # 노드 및 엣지 피처 저장
    g.ndata['h'] = torch.stack(atom_feats)
    g.edata['e'] = torch.stack(bond_feats)

    return g

##### 5. 순차 변환 및 저장 #####
train_graphs = [mol_to_dgl_graph(mol) for mol in train['mol'] if mol is not None]
test_graphs = [mol_to_dgl_graph(mol) for mol in test['mol'] if mol is not None]

# print(train_graphs)
# print(test_graphs)
# print(type(train_graphs))
# print(type(test_graphs))
# exit()
print("✅ GNN 전처리 완료!")

#================================================================
#====================== GNN 모델 구성 ===========================
#================================================================

class GraphDataset(Dataset):
    def __init__(self, graphs, labels=None):
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return g, label
        else:
            return g

from dgl.nn import Set2Set, GATv2Conv

class GATv2Set2Set(nn.Module):
    def __init__(self, in_node_feats, hidden_size=128, num_heads=4, dropout=0.2):
        super().__init__()
        self.gat1 = GATv2Conv(in_node_feats, hidden_size, num_heads)
        self.gat2 = GATv2Conv(hidden_size * num_heads, hidden_size, 1)
        self.set2set = Set2Set(hidden_size, n_iters=6, n_layers=1)
        self.readout = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, g):
        h = g.ndata['h']
        h = self.gat1(g, h).flatten(1)
        h = F.elu(h)
        h = self.gat2(g, h).mean(1)
        g.ndata['h'] = h
        hg = self.set2set(g, h)
        return self.readout(hg).squeeze(-1)


def collate_graphs(batch):
    graphs, labels = zip(*batch)
    return dgl.batch(graphs), torch.stack(labels)

def evaluate(model, dataloader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in dataloader:
            g = batch.to(device)
            pred = model(g)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds)

def main():
    y = train["Inhibition"].values
    print("🔹 GNN 모델별 5-Fold 학습 시작")

    # ✅ CPU 고정
    device = torch.device("cpu")

    n_splits = 5
    oof = np.zeros(len(train_graphs))
    test_preds_each_fold = []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=r)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_graphs)):
        print(f"🔁 Fold {fold + 1}")

        tr_graphs = [train_graphs[i] for i in tr_idx]
        val_graphs = [train_graphs[i] for i in val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        train_dataset = GraphDataset(tr_graphs, y_tr)
        val_dataset = GraphDataset(val_graphs, y_val)
        test_dataset = GraphDataset(test_graphs)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_graphs)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_graphs)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=lambda x: dgl.batch(x))

        in_node_feats = train_graphs[0].ndata['h'].shape[1]

        model = GATv2Set2Set(in_node_feats=in_node_feats, hidden_size=128, num_heads=4, dropout=0.2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        best_rmse = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(1, 101):
            model.train()
            for batch in train_loader:
                g, yb = batch
                g = g.to(device)
                yb = yb.to(device)
                pred = model(g)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            val_preds = []
            with torch.no_grad():
                for batch in val_loader:
                    g, _ = batch
                    g = g.to(device)
                    pred = model(g).squeeze(-1)
                    val_preds.append(np.atleast_1d(pred.cpu().numpy()))
            val_preds = np.concatenate(val_preds) 
            val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

            print(f"Epoch {epoch:03d} | Val RMSE: {val_rmse:.4f}")
            print_scores(y_val, val_preds, "KFold_GAT_Set2set")

            if val_rmse < best_rmse:
                best_rmse = val_rmse
                patience_counter = 0
                os.makedirs('./Drug/_02/full_pipeline/fold/', exist_ok=True)
                torch.save(model.state_dict(), f"./Drug/_02/full_pipeline/fold/gat_fold{fold}.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("⏹ Early stopping triggered")
                    break

        model.load_state_dict(torch.load(f"./Drug/_02/full_pipeline/fold/gat_fold{fold}.pt"))
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                g, _ = batch
                g = g.to(device)
                pred = model(g)
                val_preds.append(pred.cpu().numpy())
        oof[val_idx] = np.concatenate(val_preds)

        test_pred = evaluate(model, test_loader, device)
        test_preds_each_fold.append(test_pred)

    test_preds = np.mean(test_preds_each_fold, axis=0)
    
    print(test_preds.shape)
    
    save_dir = "./Drug/_04/_datasets/"
    os.makedirs(save_dir, exist_ok=True)
    GNN = pd.DataFrame(test_preds, columns=['GNN'])
    DNN_GNN = pd.concat([DNN, GNN], axis=1)
    DNN_GNN.to_csv(save_dir + 'DNN_GNN.csv', index=False)
    
if __name__ == "__main__":
    main()

print("✅ GATv2 + Set2set 모델 실행 완료!")

#================================================================
#===================== 최종 가중 평균 ===========================
#================================================================

dnn_gnn = pd.read_csv('./Drug/_04/_datasets/DNN_GNN.csv')
submit = pd.read_csv('./Drug/sample_submission.csv')

alpha = 0.5  # 또는 다른 비율로 조정 가능
inhibition = alpha * dnn_gnn["DNN"] + (1 - alpha) * dnn_gnn["GNN"]
submit['Inhibition']=inhibition

sub_path = './Drug/_04/_submission/'
os.makedirs(sub_path, exist_ok=True)
submit.to_csv(sub_path + 'submission.csv', index=False)
print("✅ 최종 Submission 저장 완료!")
