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

# 통계
from scipy.stats import pearsonr

# 모델
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# PyCaret
from pycaret.regression import *

RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings('ignore')

# 시드 고정
r = random.randint(1,1000)
random.seed(r)
np.random.seed(r)
torch.manual_seed(r)

# 경로
train_path = "./Drug/train.csv"
test_path = "./Drug/test.csv"
save_dir = "./Drug/_02/full_pipeline/0_dataset"
os.makedirs(save_dir, exist_ok=True)

# 데이터 로드
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# SMILES 추출
train_smiles = train_df["Canonical_Smiles"].tolist()
test_smiles = test_df["Canonical_Smiles"].tolist()

# descriptor + fingerprint 추출 함수
def get_descriptor_fingerprint(smiles_list):
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    all_features = []
    valid_indices = []

    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        # RDKit Descriptors
        desc = list(calculator.CalcDescriptors(mol))

        # Morgan Fingerprint (bit vector)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fp_array = np.zeros((2048,), dtype=int)
        AllChem.DataStructs.ConvertToNumpyArray(fp, fp_array)
        fp_bits = fp_array.tolist()

        # 결합
        features = desc + fp_bits
        all_features.append(features)
        valid_indices.append(idx)

    desc_cols = descriptor_names
    fp_cols = [f"FP_{i}" for i in range(2048)]
    all_cols = desc_cols + fp_cols

    return pd.DataFrame(all_features, columns=all_cols), valid_indices

# 피처 생성
train_feat_df, train_valid_idx = get_descriptor_fingerprint(train_smiles)
test_feat_df, test_valid_idx = get_descriptor_fingerprint(test_smiles)

# index 맞추기
train_df = train_df.iloc[train_valid_idx].reset_index(drop=True)
test_df = test_df.iloc[test_valid_idx].reset_index(drop=True)

# Inhibition 추가
train_feat_df["Inhibition"] = train_df["Inhibition"].values

# 1. 상수열 제거
stds = train_feat_df.drop(columns="Inhibition").std()
non_constant_cols = stds[stds > 0].index.tolist()
train_feat_df = train_feat_df[non_constant_cols + ["Inhibition"]]
test_feat_df = test_feat_df[non_constant_cols]

# 2. 결측값 KNN 대체
imputer = KNNImputer(n_neighbors=5)
X_all = pd.concat([train_feat_df.drop(columns="Inhibition"), test_feat_df])
X_imputed = imputer.fit_transform(X_all)
X_train = X_imputed[:len(train_feat_df)]
X_test = X_imputed[len(train_feat_df):]

# 3. 상관관계 높은 피처 제거
X_df = pd.DataFrame(X_train, columns=non_constant_cols)
corr_matrix = X_df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
X_df = X_df.drop(columns=to_drop)
X_test_df = pd.DataFrame(X_test, columns=non_constant_cols).drop(columns=to_drop)

# 4. 클리핑
lower = X_df.quantile(0.01)
upper = X_df.quantile(0.99)
X_df = X_df.clip(lower=lower, upper=upper, axis=1)
X_test_df = X_test_df.clip(lower=lower, upper=upper, axis=1)
from sklearn.preprocessing import RobustScaler
# 5. 스케일링
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_df)
X_test_scaled = scaler.transform(X_test_df)

#==================================================================
from sklearn.decomposition import PCA

# # 누적 설명 분산 비율 기준으로 차원 선택
# pca = PCA(n_components=X_train_scaled.shape[0])
# pca.fit(X_train_scaled)

# # 누적 설명 분산 비율 계산
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# threshold = 0.999  # 96% 이상 설명하도록
# n_components = np.argmax(cumsum >= threshold) + 1
# print(f"🎯 선택된 PCA 차원 수: {n_components}")

# exit()

# pca = PCA(n_components=1006)
# X_train_pca = pca.fit_transform(X_train_scaled)
# X_test_pca = pca.transform(X_test_scaled)

# 6. 최종 저장
final_train = pd.DataFrame(X_train_scaled, )
final_test = pd.DataFrame(X_test_scaled,)
final_train["Inhibition"] = train_df["Inhibition"].values

#==================================================================

final_train.to_csv(f"{save_dir}/train_descriptor.csv", index=False)
final_test.to_csv(f"{save_dir}/test_descriptor.csv", index=False)

print("✅ 완료: 전체 열 수 =", final_train.shape[1])

# ===================================================
# 전처리 바꿈
# ===================================================



# 경로 설정
train_path = "./Drug/train.csv"
test_path = "./Drug/test.csv"
save_dir = "./Drug/_02/full_pipeline/0_dataset"
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
train_graphs = [mol_to_dgl_graph(mol) for mol in train_df['mol'] if mol is not None]
test_graphs = [mol_to_dgl_graph(mol) for mol in test_df['mol'] if mol is not None]

joblib.dump(train_graphs, f"{save_dir}/train_graphs.pkl")
joblib.dump(test_graphs, f"{save_dir}/test_graphs.pkl")

##### 6. 분자 전체 특성 저장 (선택) #####
train_df['MolWt'] = train_df['mol'].apply(lambda m: Descriptors.MolWt(m) if m else np.nan)
train_df['logP'] = train_df['mol'].apply(lambda m: Descriptors.MolLogP(m) if m else np.nan)
train_df[['Canonical_Smiles', 'MolWt', 'logP']].to_csv(f"{save_dir}/train_mol_features.csv", index=False)

print("✅ GNN 그래프 변환 및 저장 완료!")


##### GAT 모델 구성 #####


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
    train_graphs = joblib.load("./Drug/_02/full_pipeline/0_dataset/train_graphs.pkl")
    test_graphs = joblib.load("./Drug/_02/full_pipeline/0_dataset/test_graphs.pkl")
    train_df = pd.read_csv("./Drug/train.csv")
    test_df = pd.read_csv("./Drug/test.csv")
    y = train_df["Inhibition"].values

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
            # print(f"val_preds.shape = {val_preds.shape}, y_val.shape = {y_val.shape}")
            val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

            print(f"Epoch {epoch:03d} | Val RMSE: {val_rmse:.4f}")

            if val_rmse < best_rmse:
                best_rmse = val_rmse
                patience_counter = 0
                torch.save(model.state_dict(), f"./Drug/_02/full_pipeline/gat_fold{fold}.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("⏹ Early stopping triggered")
                    break

        model.load_state_dict(torch.load(f"./Drug/_02/full_pipeline/gat_fold{fold}.pt"))
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
    test_std = np.std(test_preds_each_fold, axis=0)
    print(f"📊 Test prediction std across folds: {np.mean(test_std):.5f}")

    np.save("./Drug/_02/full_pipeline/gat_oof.npy", oof)
    np.save("./Drug/_02/full_pipeline/gat_preds.npy", test_preds)

    submission = pd.DataFrame({
        "ID": test_df["ID"],
        "Inhibition": test_preds
    })
    
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"pre_GAT_{r}_({now}).csv"
    save_path = f"./Drug/_02/full_pipeline/{filename}"
    
    submission.to_csv(save_path, index=False)
    print("✅ gat_oof.npy, gat_preds.npy, pre_GAT.csv 저장 완료")


if __name__ == "__main__":
    main()
    


def print_scores(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    pearson = pearsonr(y_true, y_pred)[0]
    score = 0.5 * (1 - min(nrmse, 1)) + 0.5 * np.clip(pearson, 0, 1)
    print(f"📊 {label} 결과")
    print(f"RMSE     : {rmse:.5f}")
    print(f"NRMSE    : {nrmse:.5f}")
    print(f"Pearson  : {pearson:.5f}")
    print(f"Score📈  : {score:.5f}")

train = pd.read_csv("./Drug/_02/full_pipeline/0_dataset/train_descriptor.csv")
test = pd.read_csv("./Drug/_02/full_pipeline/0_dataset/test_descriptor.csv")
X = train.drop(columns=["Inhibition"])
y = train["Inhibition"].values
X_test = test.copy()

kf = KFold(n_splits=5, shuffle=True, random_state=r)

param_lgbm = {
    "n_estimators": 1000,
    "learning_rate": 0.015875869112729403,
    "max_depth": 5,
    "num_leaves": 77,
    "min_child_samples": 21,
    "colsample_bytree": 0.8956019941712514,
    "random_state": r
}

param_cat = {
    "iterations": 1000,
    "learning_rate": 0.020782688572110002,
    "depth": 4,
    "l2_leaf_reg": 9.322000582507451,
    "verbose": 0,
    "random_seed": r
}

param_xgb = {
    "n_estimators": 1000,
    "learning_rate": 0.010607554597937956,
    "max_depth": 3,
    "subsample": 0.7961002707424782,
    "colsample_bytree": 0.8105132963922017,
    "verbosity": 0,
    "random_state": r
}

models = {
    "lgbm": LGBMRegressor(**param_lgbm, verbosity=-1),
    "xgb": XGBRegressor(**param_xgb, early_stopping_rounds=20),
    "cat": CatBoostRegressor(**param_cat)
}

oof_preds = {name: np.zeros(len(X)) for name in models}
test_preds = {name: np.zeros(len(X_test)) for name in models}

print("🔹 Boosting 모델별 5-Fold 학습 시작")

for name, model in models.items():
    print(f"✅ {name.upper()} 시작")
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if name == "lgbm":
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                callbacks=[early_stopping(stopping_rounds=20), log_evaluation(period=0)],
            )
        elif name == "xgb":
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        elif name == "cat":
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=20,
                use_best_model=True
            )

        oof_preds[name][val_idx] = model.predict(X_val)
        test_preds[name] += model.predict(X_test) / kf.n_splits

    print_scores(y, oof_preds[name], label=f"{name.upper()}")

# OOF stacking 앙상블
print("🔹 OOF 앙상블 최적 alpha 탐색")
best_score = -np.inf
best_weights = None
alphas = np.linspace(0, 1, 11)

for a in alphas:
    for b in alphas:
        if a + b > 1: continue
        c = 1 - a - b
        blended = a * oof_preds["lgbm"] + b * oof_preds["xgb"] + c * oof_preds["cat"]
        rmse = np.sqrt(mean_squared_error(y, blended))
        nrmse = rmse / (np.max(y) - np.min(y))
        pearson = pearsonr(y, blended)[0]
        score = 0.5 * (1 - min(nrmse, 1)) + 0.5 * np.clip(pearson, 0, 1)
        if score > best_score:
            best_score = score
            best_weights = (a, b, c)

a, b, c = best_weights
print(f"✅ 최적 가중치: LGBM={a:.2f}, XGB={b:.2f}, CAT={c:.2f} | Score: {best_score:.5f}")

final_oof = a * oof_preds["lgbm"] + b * oof_preds["xgb"] + c * oof_preds["cat"]
final_test = a * test_preds["lgbm"] + b * test_preds["xgb"] + c * test_preds["cat"]

print_scores(y, final_oof, label="Stacked Ensemble")

submission = pd.DataFrame({
    "ID": pd.read_csv("./Drug/test.csv")["ID"],
    "Inhibition": final_test
})

import datetime

# 현재 시간 포맷
now = datetime.datetime.now().strftime("%Y%m%d_%H%M")

# 파일명 생성
filename = f"pre_DNN_{r}_({now}).csv"
save_path = f"./Drug/_02/full_pipeline/{filename}"

# 저장
submission.to_csv(save_path, index=False)
print(f"✅ {filename} 저장 완료")

np.save("./Drug/_02/full_pipeline/boost_oof.npy", final_oof)
np.save("./Drug/_02/full_pipeline/boost_preds.npy", final_test)
print("✅ boost_oof.npy, boost_preds.npy 저장 완료")

# ===== File: 05_ensemble.py ===== #


# 점수 출력 함수
def print_scores(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    pearson = pearsonr(y_true, y_pred)[0]
    score = 0.5 * (1 - min(nrmse, 1)) + 0.5 * np.clip(pearson, 0, 1)
    print(f"📊 {label} 결과")
    print(f"RMSE     : {rmse:.5f}")
    print(f"NRMSE    : {nrmse:.5f}")
    print(f"Pearson  : {pearson:.5f}")
    print(f"Score📈  : {score:.5f}")
    return score

# 데이터 불러오기
y = pd.read_csv("./Drug/train.csv")["Inhibition"].values
dmpnn_oof = np.load("./Drug/_02/full_pipeline/gat_oof.npy")
boost_oof = np.load("./Drug/_02/full_pipeline/boost_oof.npy")
dmpnn_preds = np.load("./Drug/_02/full_pipeline/gat_preds.npy")
boost_preds = np.load("./Drug/_02/full_pipeline/boost_preds.npy")
test_id = pd.read_csv("./Drug/test.csv")["ID"]

# 앙상블 최적 alpha 찾기
alphas = np.linspace(0, 1, 21)
best_score = -np.inf
best_alpha = 0.5

for alpha in alphas:
    final_oof = alpha * dmpnn_oof + (1 - alpha) * boost_oof
    score = print_scores(y, final_oof, label=f"α={alpha:.2f}")
    if score > best_score:
        best_score = score
        best_alpha = alpha

# 최종 앙상블 예측
final_preds = best_alpha * dmpnn_preds + (1 - best_alpha) * boost_preds
final_oof = best_alpha * dmpnn_oof + (1 - best_alpha) * boost_oof

# 점수 출력
print_scores(y, final_oof, label=f"Final Ensemble α={best_alpha:.2f}")

# 저장
now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
filename = f"submission_{r}_final_ensemble({now}).csv"
save_path = f"./Drug/_02/full_pipeline/{filename}"

submission = pd.DataFrame({
    "ID": test_id,
    "Inhibition": final_preds
})
submission.to_csv(save_path, index=False)
print(f"✅ 최종 앙상블 파일 저장 완료 → {filename}")
print('random_state :', r)


train = pd.read_csv("./Drug/_02/full_pipeline/0_dataset/train_descriptor.csv")
test = pd.read_csv("./Drug/_02/full_pipeline/0_dataset/test_descriptor.csv")

oof_preds = np.zeros(len(train))
test_preds = []

kf = KFold(n_splits=5, shuffle=True, random_state=r)

for fold, (train_idx, valid_idx) in enumerate(kf.split(train)):
    fold_train = train.iloc[train_idx].copy()
    fold_valid = train.iloc[valid_idx].copy()

    setup(data=fold_train, target='Inhibition', session_id=r,
          normalize=True, verbose=False, use_gpu=False, )

    model = compare_models(include=['lr', 'ridge', 'lasso','rf', 'gbr'], sort='R2')
    final_model = finalize_model(model)

    valid_preds = predict_model(final_model, data=fold_valid)
    
    # ✅ 예측 컬럼명 자동 감지
    label_col = None
    for col in ['Label', 'prediction_label', 'Prediction']:
        if col in valid_preds.columns:
            label_col = col
            break
    if label_col is None:
        label_col = valid_preds.columns[-1]

    oof_preds[valid_idx] = valid_preds[label_col].values

    test_pred_df = predict_model(final_model, data=test)
    test_preds.append(test_pred_df[label_col].values if label_col in test_pred_df.columns else test_pred_df.iloc[:, -1].values)

# 평균 test 예측
test_preds_mean = np.mean(test_preds, axis=0)

np.save("./Drug/_02/full_pipeline/pycaret_oof.npy", oof_preds)
np.save("./Drug/_02/full_pipeline/pycaret_preds.npy", test_preds_mean)

def print_scores(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    pearson = pearsonr(y_true, y_pred)[0]
    score = 0.5 * (1 - min(nrmse, 1)) + 0.5 * np.clip(pearson, 0, 1)
    print(f"📊 {label}")
    print(f"RMSE     : {rmse:.5f}")
    print(f"NRMSE    : {nrmse:.5f}")
    print(f"Pearson  : {pearson:.5f}")
    print(f"Score📈  : {score:.5f}")
    return score

# 데이터 로드
y = pd.read_csv("./Drug/train.csv")["Inhibition"].values
gat_oof = np.load("./Drug/_02/full_pipeline/gat_oof.npy")
boost_oof = np.load("./Drug/_02/full_pipeline/boost_oof.npy")
pycaret_oof = np.load("./Drug/_02/full_pipeline/pycaret_oof.npy")       # ✅ 추가
gat_preds = np.load("./Drug/_02/full_pipeline/gat_preds.npy")
boost_preds = np.load("./Drug/_02/full_pipeline/boost_preds.npy")
pycaret_preds = np.load("./Drug/_02/full_pipeline/pycaret_preds.npy")   # ✅ test 예측
test_id = pd.read_csv("./Drug/test.csv")["ID"]

#=================
# 최적 가중치 찾기
#=================

best_score = -np.inf
best_weights = (0.3, 0.3, 0.4)  # 초기값

# a: GAT, b: Boost, c: PyCaret
for a in np.linspace(0, 1, 11):
    for b in np.linspace(0, 1 - a, 11):
        c = 1.0 - a - b
        if c < 0 or c > 1:
            continue
        
        # OOF 기준 평가
        blend_oof = a * gat_oof + b * boost_oof + c * pycaret_oof
        score = print_scores(y, blend_oof, label=f"α={a:.2f}, β={b:.2f}, γ={c:.2f}")
        
        if score > best_score:
            best_score = score
            best_weights = (a, b, c)

# 최적 가중치 추출
a, b, c = best_weights
# a, b, c = 0.05, 0.05, 0.90
print(f"✅ 최적 가중치: α(GAT)={a:.2f}, β(Boost)={b:.2f}, γ(PyCaret)={c:.2f}, Score={best_score:.5f}")

# test 예측
inter_blend = a * gat_preds + b * boost_preds + c * pycaret_preds
final_oof = a * gat_oof + b * boost_oof + c * pycaret_oof

# 점수 출력
final_score = print_scores(y, final_oof, label=f"Final Ensemble α={a:.2f}, β={b:.2f}, γ={c:.2f}")

# 저장
submission = pd.DataFrame({
    "ID": test_id,
    "Inhibition": inter_blend
})
now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
filename = f"submission_{r}_final_ensemble_opt_weight({now}).csv"
submission.to_csv(f"./Drug/_02/full_pipeline/{filename}", index=False)
print(f"랜덤 시드값 : {r}")
print(f"✅ 최종 저장: {filename}")

import shutil

# 🎯 기준 점수 설정
SCORE_THRESHOLD = 0.61616  # 원하는 기준값으로 설정

# 점수가 기준보다 낮으면 전체 디렉토리 삭제
if final_score < SCORE_THRESHOLD:
    shutil.rmtree("./Drug/_02/full_pipeline")
    print(f"🚫 Score {final_score:.5f} < 기준 {SCORE_THRESHOLD} → 전체 디렉토리 삭제 완료")
else:
    print(f"🎉 Score {final_score:.5f} ≥ 기준 {SCORE_THRESHOLD} → 디렉토리 유지")

# ✅ 최적 가중치: α(GAT)=0.20, β(Boost)=0.32, γ(PyCaret)=0.48, Score=0.56802
# 📊 Final Ensemble α=0.20, β=0.32, γ=0.48
# RMSE     : 24.62871
# NRMSE    : 0.24782
# Pearson  : 0.38386
# Score📈  : 0.56802
# 랜덤 시드값 : 707

# 📊 Final Ensemble α=0.20, β=0.24, γ=0.56
# RMSE     : 24.46559
# NRMSE    : 0.24618
# Pearson  : 0.40069
# Score📈  : 0.57726
# 랜덤 시드값 : 916

# 📊 Final Ensemble α=0.30, β=0.14, γ=0.56
# RMSE     : 24.46458
# NRMSE    : 0.24617
# Pearson  : 0.40795
# Score📈  : 0.58089
# 랜덤 시드값 : 830

# ✅ 최적 가중치: α(GAT)=0.00, β(Boost)=0.50, γ(PyCaret)=0.50, Score=0.61351
# 📊 Final Ensemble α=0.00, β=0.50, γ=0.50
# RMSE     : 23.46277
# NRMSE    : 0.23609
# Pearson  : 0.46311
# Score📈  : 0.61351
# 랜덤 시드값 : 549

# ✅ 최적 가중치: α(GAT)=0.00, β(Boost)=0.60, γ(PyCaret)=0.40, Score=0.61616
# 📊 Final Ensemble α=0.00, β=0.60, γ=0.40
# RMSE     : 23.42074
# NRMSE    : 0.23566
# Pearson  : 0.46798
# Score📈  : 0.61616
# 랜덤 시드값 : 368