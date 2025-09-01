# ========================
# 임포트 및 랜덤 시드 고정
# ========================
# ========================
# 임포트 및 랜덤 시드 고정
# ========================
import pandas as pd
import numpy as np
import os
import json
import random
import seaborn as sns
import matplotlib.pyplot as plt
import dgl
import pickle
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from dgl.nn import GATv2Conv, Set2Set
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import early_stopping, log_evaluation
from sklearn.neighbors import NearestNeighbors
from sklearn.base import clone
from torch.utils.data import DataLoader, TensorDataset, Dataset
from scipy.special import logit, expit

DEVICE = torch.device("cpu")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from scipy.stats import pearsonr

def get_final_score(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    norm_rmse = rmse / (np.max(y_true) - np.min(y_true))
    A = norm_rmse
    B, _ = pearsonr(y_true, y_pred)
    return A, B, 0.5 * (1 - min(A, 1)) + 0.5 * B

data_path = './Drug/'
save_path = './Drug/_06/01_submission/'
os.makedirs(save_path, exist_ok=True)
seed_file = "./Drug/_06/01_submission/01_00.json"

# 파일이 없으면 처음 생성
if not os.path.exists(seed_file):
    seed_state = {"seed": 42}
else:
    with open(seed_file, "r") as f:
        seed_state = json.load(f)

# 현재 seed 값 사용
SEED = 1 #seed_state["seed"]
print(f"[Current Run SEED]: {SEED}")

# 다음 실행을 위해 seed 값 1 증가
seed_state["seed"] += 1
with open(seed_file, "w") as f:
    json.dump(seed_state, f)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

#######################
""" # (기본 데이터 로드)
train = pd.read_csv(data_path + 'train.csv', index_col=0)
test = pd.read_csv(data_path + 'test.csv', index_col=0)
 """
submission = pd.read_csv(data_path + 'sample_submission.csv')

# print(train.columns)
# # Index(['Canonical_Smiles', 'Inhibition'], dtype='object')
# print(test.columns)
# # Index(['Canonical_Smiles'], dtype='object')
# print(submission.columns)
# # Index(['ID', 'Inhibition'], dtype='object')

""" # (GNN 데이터 제작)
train_smiles = train["Canonical_Smiles"].tolist()
test_smiles = test["Canonical_Smiles"].tolist()

# ----------------------------
# SMILES -> DGL 그래프 함수
# ----------------------------
def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_feats = []
    for atom in mol.GetAtoms():
        atom_feats.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic())
        ])
    
    g = dgl.graph(([], []))
    g.add_nodes(len(atom_feats))
    g.ndata['h'] = torch.tensor(atom_feats, dtype=torch.float32)

    src, dst = [], []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src += [u, v]
        dst += [v, u]

    g.add_edges(src, dst)
    return g

def smiles_to_graphs(smiles_list):
    graphs = []
    valid_idx = []
    for idx, smi in enumerate(tqdm(smiles_list)):
        g = mol_to_graph(smi)
        if g is not None:
            graphs.append(g)
            valid_idx.append(idx)
    return graphs, valid_idx

# ----------------------------
# 변환 실행
# ----------------------------
gnn_train_graphs, train_valid_idx = smiles_to_graphs(train_smiles)
gnn_test_graphs, test_valid_idx = smiles_to_graphs(test_smiles)

gnn_train_labels = train.iloc[train_valid_idx]['Inhibition'].reset_index(drop=True).values
gnn_test_ids = test.iloc[test_valid_idx].reset_index(drop=True).index

# ----------------------------
# 저장
# ----------------------------
with open(save_path + 'gnn_train.pkl', 'wb') as f:
    pickle.dump((gnn_train_graphs, gnn_train_labels), f)

with open(save_path + 'gnn_test.pkl', 'wb') as f:
    pickle.dump((gnn_test_graphs, gnn_test_ids), f)

print(f"✅ 저장 완료: train {len(gnn_train_graphs)}개, test {len(gnn_test_graphs)}개")
 """

""" # (DNN 데이터 제작)
import pandas as pd
from tqdm import tqdm


# SMILES 추출
train_smiles = train["Canonical_Smiles"].tolist()
test_smiles = test["Canonical_Smiles"].tolist()

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
        Crippen.MolLogP(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
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
train_feat_df, train_valid_idx = get_descriptor_fingerprint(train_smiles)
test_feat_df, test_valid_idx = get_descriptor_fingerprint(test_smiles)

# 유효한 row 기준으로 SMILES 추출
train_df = train.iloc[train_valid_idx].reset_index(drop=True)
test_df = test.iloc[test_valid_idx].reset_index(drop=True)

# 타겟 추가
train_feat_df["Inhibition"] = train_df["Inhibition"].values

# 상수 피처 제거
stds = train_feat_df.drop(columns="Inhibition").std()
non_constant_cols = stds[stds > 0].index.tolist()
train_feat_df = train_feat_df[non_constant_cols + ["Inhibition"]]
test_feat_df = test_feat_df[non_constant_cols]

# ✅ column-wise 병합
train_final = pd.concat([train_df[["Canonical_Smiles"]], train_feat_df], axis=1)
test_final = pd.concat([test_df[["Canonical_Smiles"]], test_feat_df], axis=1)

# # 저장
# train_final.to_csv(save_path + 'tr.csv', index=False)
# test_final.to_csv(save_path + 'te.csv', index=False)

# train = pd.read_csv(save_path +'tr.csv')
# test = pd.read_csv(save_path +'te.csv')

train_df  = train_final.drop(['Canonical_Smiles'], axis=1).copy()
test_df  = test_final.drop(['Canonical_Smiles'], axis=1).copy()

def get_multicollinearity_drop_cols(df, threshold=0.9):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    return [column for column in upper.columns if any(upper[column] > threshold)]


def split_fingerprint_columns(df):
    fp_cols = [col for col in df.columns if col.startswith("FP_")]
    non_fp_cols = [col for col in df.columns if not col.startswith("FP_")]

    df_fp = df[fp_cols].copy()
    df_desc = df[non_fp_cols].copy()

    print(f"✅ Fingerprint 컬럼 수: {len(fp_cols)}개")
    print(f"✅ Descriptor 컬럼 수: {len(non_fp_cols)}개")
    return df_desc, df_fp

# target 컬럼 분리
target_col = 'Inhibition'
y = train[target_col]
train_desc, train_fp = split_fingerprint_columns(train_df.drop(columns=target_col))

# test도 동일하게
test_desc, test_fp = split_fingerprint_columns(test_df)

# 제거 대상 컬럼
drop_cols = get_multicollinearity_drop_cols(train_desc, threshold=0.9)

# 2. 제거 적용 (일관되게)
train_desc_clean = train_desc.drop(columns=drop_cols)
test_desc_clean = test_desc.drop(columns=drop_cols)

print(f"✅ 제거된 피처 수 (공통): {len(drop_cols)}")

# 필요시 다시 결합
train_final_desc = pd.concat([train_desc_clean, y], axis=1)
train_final_fp = pd.concat([train_fp, y], axis=1)

# print(train_final_desc.shape)
# print(train_final_fp.shape)
# print(test_desc_clean.shape)
# print(test_fp.shape)

train_de_final = pd.concat([train[["Canonical_Smiles"]], train_final_desc], axis=1)
test_de_final = pd.concat([test[["Canonical_Smiles"]], test_desc_clean], axis=1)
train_fp_final = pd.concat([train[["Canonical_Smiles"]], train_final_fp], axis=1)
test_fp_final = pd.concat([test[["Canonical_Smiles"]], test_fp], axis=1)

# 저장
train_de_final.to_csv(save_path + 'de_train.csv', index=False)
test_de_final.to_csv(save_path + 'de_test.csv', index=False)
train_fp_final.to_csv(save_path + 'fp_train.csv', index=False)
test_fp_final.to_csv(save_path + 'fp_test.csv', index=False)
"""

""" # (DNN 모델)
#=======================
# 0. 데이터 로드
#=======================

de_train = pd.read_csv(save_path + 'de_train.csv')
de_test  = pd.read_csv(save_path + 'de_test.csv')
fp_train = pd.read_csv(save_path + 'fp_train.csv')
fp_test  = pd.read_csv(save_path + 'fp_test.csv')

#=======================
# 1. Descriptors
#=======================
# 데이터 준비
Xd = de_train.drop(['Canonical_Smiles', 'Inhibition'], axis=1).copy()
Yd = de_train['Inhibition'].copy()
testd = de_test.drop(['Canonical_Smiles'], axis=1).copy()

Xd = Xd.fillna(Xd.median())
testd = testd.fillna(Xd.median())

# K-Fold 설정
N_SPLIT = 5
kf = KFold(n_splits=N_SPLIT, shuffle=True, random_state=42)

# OOF 및 테스트 예측값 저장 배열
oof_d_lgb = np.zeros(len(Xd))
oof_d_xgb = np.zeros(len(Xd))
oof_d_cb = np.zeros(len(Xd))
test_d_lgb = np.zeros(len(testd))
test_d_xgb = np.zeros(len(testd))
test_d_cb = np.zeros(len(testd))

# 학습 루프
print("--- Descriptor Model Training ---")
for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(Xd)), total=N_SPLIT):
    X_train, X_val = Xd.iloc[train_idx], Xd.iloc[val_idx]
    y_train, y_val = Yd.iloc[train_idx], Yd.iloc[val_idx]

    # LGBM
    lgb = LGBMRegressor(n_estimators=1000, learning_rate=0.01, random_state=fold, verbosity=-1, early_stopping_rounds=50)
    lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    oof_d_lgb[val_idx] = lgb.predict(X_val)
    test_d_lgb += lgb.predict(testd) / N_SPLIT

    # XGBoost
    xgb = XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=6, tree_method='hist', random_state=fold, verbosity=0, early_stopping_rounds=50)
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    oof_d_xgb[val_idx] = xgb.predict(X_val)
    test_d_xgb += xgb.predict(testd) / N_SPLIT

    # CatBoost
    cb = CatBoostRegressor(iterations=1000, learning_rate=0.01, depth=6, random_seed=fold, verbose=False, loss_function='RMSE')
    cb.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    oof_d_cb[val_idx] = cb.predict(X_val)
    test_d_cb += cb.predict(testd) / N_SPLIT

# Descriptor 모델 앙상블 결과
oof_d_avg = (oof_d_lgb + oof_d_xgb + oof_d_cb) / 3
test_d_avg = (test_d_lgb + test_d_xgb + test_d_cb) / 3

#---------------------------------------
# FP 모델 학습 (결과2)
#---------------------------------------

# 데이터 준비
Xf = fp_train.drop(['Canonical_Smiles'], axis=1).copy()
Yf = de_train['Inhibition'].copy() # Inhibition 타겟은 동일

# --- 에러 해결 코드 추가: fp_test의 컬럼을 fp_train과 일치시키기 ---
testf = fp_test.drop(['Canonical_Smiles'], axis=1).copy()
train_cols = Xf.columns
test_cols = testf.columns

# fp_train에만 있는 컬럼을 fp_test에 추가하고 값은 0으로 채움
missing_cols_in_test = set(train_cols) - set(test_cols)
for c in missing_cols_in_test:
    testf[c] = 0

# fp_test에만 있는 컬럼을 삭제
extra_cols_in_test = set(test_cols) - set(train_cols)
testf = testf.drop(columns=list(extra_cols_in_test))

# 최종적으로 fp_train과 동일한 컬럼 순서로 정렬
testf = testf[train_cols]
# -------------------------------------------------------------

# K-Fold 설정 (동일한 seed 사용)
kf = KFold(n_splits=N_SPLIT, shuffle=True, random_state=42)

# OOF 및 테스트 예측값 저장 배열
oof_f_lgb = np.zeros(len(Xf))
oof_f_xgb = np.zeros(len(Xf))
oof_f_cb = np.zeros(len(Xf))
test_f_lgb = np.zeros(len(testf))
test_f_xgb = np.zeros(len(testf))
test_f_cb = np.zeros(len(testf))

# 학습 루프
print("\n--- Fingerprint Model Training ---")
for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(Xf)), total=N_SPLIT):
    X_train, X_val = Xf.iloc[train_idx], Xf.iloc[val_idx]
    y_train, y_val = Yf.iloc[train_idx], Yf.iloc[val_idx]

    # LGBM
    lgb = LGBMRegressor(n_estimators=1000, learning_rate=0.01, random_state=fold, verbosity=-1, early_stopping_rounds=50)
    # LighGBM의 feature 수 체크를 비활성화 (오류 방지)
    # predict_disable_shape_check=True 옵션은 권장되지 않으므로, 데이터 정제 방식을 사용
    lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    oof_f_lgb[val_idx] = lgb.predict(X_val)
    test_f_lgb += lgb.predict(testf) / N_SPLIT

    # XGBoost
    xgb = XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=6, tree_method='hist', random_state=fold, verbosity=0, early_stopping_rounds=50)
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    oof_f_xgb[val_idx] = xgb.predict(X_val)
    test_f_xgb += xgb.predict(testf) / N_SPLIT

    # CatBoost
    cb = CatBoostRegressor(iterations=1000, learning_rate=0.01, depth=6, random_seed=fold, verbose=False, loss_function='RMSE')
    cb.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    oof_f_cb[val_idx] = cb.predict(X_val)
    test_f_cb += cb.predict(testf) / N_SPLIT

# FP 모델 앙상블 결과
oof_f_avg = (oof_f_lgb + oof_f_xgb + oof_f_cb) / 3
test_f_avg = (test_f_lgb + test_f_xgb + test_f_cb) / 3

#---------------------------------------
# 최종 앙상블 및 Residual 보정
#---------------------------------------

# Descriptor와 FP 모델의 OOF 및 Test 예측값 병합
oof_avg = (oof_d_avg + oof_f_avg) / 2
test_avg = (test_d_avg + test_f_avg) / 2

# 전체 데이터셋 결합
X_full = pd.concat([Xd, Xf], axis=1)
test_full = pd.concat([testd, testf], axis=1)

# StandardScaler 적용
scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X_full)
test_full_scaled = scaler.transform(test_full)

# Residual 계산
Y_full = de_train['Inhibition'].copy()
residual = Y_full - oof_avg

# Residual 모델 학습 (RidgeCV)
res_model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
res_model.fit(X_full_scaled, residual)

# 예측 (잔차 보정값)
res_pred_oof = res_model.predict(X_full_scaled)
res_pred_test = res_model.predict(test_full_scaled)

# 최종 예측 = 앙상블 예측값 + Residual 보정값
final_oof = oof_avg + res_pred_oof
final_test = test_avg + res_pred_test

#---------------------------------------
# 평가 및 저장
#---------------------------------------

def get_final_score(y_true, y_pred):
    nrmse = np.sqrt(np.mean((y_true - y_pred)**2)) / (y_true.max() - y_true.min())
    pearsonr = np.corrcoef(y_true, y_pred)[0, 1]
    score = 0.5 * pearsonr + 0.5 * (1 - nrmse)
    return nrmse, pearsonr, score

A, B, score = get_final_score(Y_full, final_oof)
print(f"\n🎯 Residual NRMSE: {A:.5f}")
print(f"📈 Residual Pearsonr: {B:.5f}")
print(f"⭐ Residual Final SCORE: {score:.5f}")

# 예측 결과 저장
submission['Inhibition'] = np.clip(final_test, 0, 100)
submission.to_csv(save_path + 'residual_ensemble_fp_de_fixed.csv', index=False)
 """

""" # (GNN 모델)
with open(save_path +'gnn_train.pkl', 'rb') as f:
    gnn_train_graphs, gnn_train_labels = pickle.load(f)

with open(save_path +'/gnn_test.pkl', 'rb') as f:
    gnn_test_graphs, gnn_test_ids = pickle.load(f)
first_graph_ndata_keys = gnn_train_graphs[0].ndata.keys()
# print(f"Keys in the first graph's ndata: {first_graph_ndata_keys}")
# exit()
# ========================
# GNN 모델 정의 (GATv2Conv + Set2Set)
# ========================
class GATv2Set2Set(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(GATv2Set2Set, self).__init__()
        self.conv1 = GATv2Conv(in_dim, hidden_dim, num_heads=num_heads, activation=F.elu)
        self.conv2 = GATv2Conv(hidden_dim * num_heads, hidden_dim, num_heads=num_heads, activation=F.elu)
        self.readout = Set2Set(hidden_dim * num_heads, n_iters=6, n_layers=3)
        self.predict = nn.Sequential(
            nn.Linear(2 * hidden_dim * num_heads, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, g, features):
        g = g.to(DEVICE)
        features = features.to(DEVICE)

        h = self.conv1(g, features).flatten(1)
        h = self.conv2(g, h).flatten(1)
        
        with g.local_scope():
            g.ndata['h'] = h
            hg = self.readout(g, h)
            return self.predict(hg)
# ========================
# 데이터셋 및 DataLoader
# ========================
class GraphDataset(Dataset):
    def __init__(self, graphs, labels=None):
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.graphs[idx], torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            return self.graphs[idx], None

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    
    # 각 그래프에 self-loop 추가
    graphs = [dgl.add_self_loop(g) for g in graphs]
    
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.tensor([label for label in labels if label is not None], dtype=torch.float32)
    return batched_graph, batched_labels.unsqueeze(1)

gnn_train_graphs, gnn_val_graphs, gnn_train_labels, gnn_val_labels = train_test_split(
    gnn_train_graphs, gnn_train_labels, test_size=0.2, random_state=SEED
)

# 학습 데이터셋 및 DataLoader
train_dataset = GraphDataset(gnn_train_graphs, gnn_train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate)

# 검증 데이터셋 및 DataLoader
val_dataset = GraphDataset(gnn_val_graphs, gnn_val_labels)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate)

# 테스트 데이터셋 및 DataLoader
test_dataset = GraphDataset(gnn_test_graphs)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate)
\
# ========================
# 모델 학습 및 추론 (Early Stopping 추가)
# ========================
in_dim = gnn_train_graphs[0].ndata['h'].shape[1]
hidden_dim = 64
out_dim = 1
num_heads = 4

model = GATv2Set2Set(in_dim, hidden_dim, out_dim, num_heads).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Early Stopping 파라미터
patience = 20  # 성능 개선이 없을 때 몇 번의 에포크를 더 기다릴지
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

epochs = 100
for epoch in range(epochs):
    # 훈련 단계
    model.train()
    train_loss = 0
    for batched_graph, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]"):
        labels = labels.to(DEVICE)
        node_features = batched_graph.ndata['h'].float()
        optimizer.zero_grad()
        output = model(batched_graph, node_features)
        
        # MSE Loss 사용
        loss = loss_fn(output, labels)
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    avg_train_loss = train_loss / len(train_dataloader)
    
    # 검증 단계
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batched_graph, labels in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [VALID]"):
            labels = labels.to(DEVICE)
            node_features = batched_graph.ndata['h'].float()
            output = model(batched_graph, node_features)
            
            # MSE Loss 사용
            loss = loss_fn(output, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_dataloader)
    
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")

    # Early Stopping 로직
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = model.state_dict() # 최적 모델 가중치 저장
        print(f"Validation loss improved. Saving model weights.")
    else:
        patience_counter += 1
        print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
    
    if patience_counter >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        if best_model_state:
            model.load_state_dict(best_model_state) # 최적 모델 가중치 복원
        break

# 추론
model.eval()
gnn_preds = []
with torch.no_grad():
    for batched_graph, _ in tqdm(test_dataloader, desc="[INFERENCE]"):
        node_features = batched_graph.ndata['h'].float()
        output = model(batched_graph, node_features)
        gnn_preds.extend(output.cpu().numpy().flatten())

# 결과 저장
submission = pd.DataFrame({'id': gnn_test_ids, 'smiles': [None] * len(gnn_test_ids), 'Predicted': gnn_preds})
submission.drop('smiles', axis=1, inplace=True)
submission.to_csv(save_path + 'gnn_gatv2set2set_submission.csv', index=False)

print("GNN GATv2Set2Set 모델 학습 및 추론 완료. 결과가 저장되었습니다.")
 """

""" # (GNN + DNN)
# 데이터 로드
try:
    de_train = pd.read_csv(save_path + 'de_train.csv')
    de_test = pd.read_csv(save_path + 'de_test.csv')
    # fp_train, fp_test는 제외
    with open(save_path + 'gnn_train.pkl', 'rb') as f:
        gnn_train_graphs, gnn_train_labels = pickle.load(f)
    with open(save_path + 'gnn_test.pkl', 'rb') as f:
        gnn_test_graphs, gnn_test_ids = pickle.load(f)
except FileNotFoundError:
    print("Error: Ensure data files are in the correct path.")
    exit()

# 성능 계산 함수
def get_final_score(y_true, y_pred):
    nrmse = np.sqrt(np.mean((y_true - y_pred)**2)) / (y_true.max() - y_true.min())
    pearsonr = np.corrcoef(y_true, y_pred)[0, 1]
    score = 0.5 * pearsonr + 0.5 * (1 - nrmse)
    return nrmse, pearsonr, score

#---------------------------------------
# 1. Descriptor 모델 학습
#---------------------------------------

# Descriptor 데이터 준비
Xd = de_train.drop(['Canonical_Smiles', 'Inhibition'], axis=1).copy()
Yd = de_train['Inhibition'].copy()
testd = de_test.drop(['Canonical_Smiles'], axis=1).copy()
Xd = Xd.fillna(Xd.median())
testd = testd.fillna(Xd.median())

# K-Fold 설정
N_SPLIT = 5
kf = KFold(n_splits=N_SPLIT, shuffle=True, random_state=SEED)

oof_d_lgb, oof_d_xgb, oof_d_cb = np.zeros(len(Xd)), np.zeros(len(Xd)), np.zeros(len(Xd))
test_d_lgb, test_d_xgb, test_d_cb = np.zeros(len(testd)), np.zeros(len(testd)), np.zeros(len(testd))

print("--- Descriptor Model Training ---")
for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(Xd)), total=N_SPLIT):
    X_train, X_val = Xd.iloc[train_idx], Xd.iloc[val_idx]
    y_train, y_val = Yd.iloc[train_idx], Yd.iloc[val_idx]

    lgb = LGBMRegressor(n_estimators=1000, learning_rate=0.01, random_state=SEED, verbosity=-1, early_stopping_rounds=50)
    lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    oof_d_lgb[val_idx] = lgb.predict(X_val)
    test_d_lgb += lgb.predict(testd) / N_SPLIT

    xgb = XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=6, tree_method='hist', random_state=SEED, verbosity=0, early_stopping_rounds=50)
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    oof_d_xgb[val_idx] = xgb.predict(X_val)
    test_d_xgb += xgb.predict(testd) / N_SPLIT

    cb = CatBoostRegressor(iterations=1000, learning_rate=0.01, depth=6, random_seed=SEED, verbose=False, loss_function='RMSE')
    cb.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    oof_d_cb[val_idx] = cb.predict(X_val)
    test_d_cb += cb.predict(testd) / N_SPLIT

# Descriptor 모델 앙상블 결과
oof_d_avg = (oof_d_lgb + oof_d_xgb + oof_d_cb) / 3
test_d_avg = (test_d_lgb + test_d_xgb + test_d_cb) / 3

# 성능 출력
Y_full = de_train['Inhibition'].copy()
nrmse_d, pearsonr_d, score_d = get_final_score(Y_full, oof_d_avg)
print("\n--- Descriptor Model Performance ---")
print(f"🎯 NRMSE: {nrmse_d:.5f}")
print(f"📈 Pearsonr: {pearsonr_d:.5f}")
print(f"⭐ SCORE: {score_d:.5f}")

#---------------------------------------
# 2. GNN 모델 학습 및 K-Fold 예측
#---------------------------------------

class GATv2Set2Set(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(GATv2Set2Set, self).__init__()
        self.conv1 = GATv2Conv(in_dim, hidden_dim, num_heads=num_heads, activation=F.elu)
        self.conv2 = GATv2Conv(hidden_dim * num_heads, hidden_dim, num_heads=num_heads, activation=F.elu)
        self.readout = Set2Set(hidden_dim * num_heads, n_iters=6, n_layers=3)
        self.predict = nn.Sequential(
            nn.Linear(2 * hidden_dim * num_heads, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, g, features):
        g = g.to(DEVICE)
        features = features.to(DEVICE)
        h = self.conv1(g, features).flatten(1)
        h = self.conv2(g, h).flatten(1)
        with g.local_scope():
            g.ndata['h'] = h
            hg = self.readout(g, h)
            return self.predict(hg)

class GraphDataset(Dataset):
    def __init__(self, graphs, labels=None):
        self.graphs = graphs
        self.labels = labels
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.graphs[idx], torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            return self.graphs[idx], None

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    graphs = [dgl.add_self_loop(g) for g in graphs]
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.tensor([label for label in labels if label is not None], dtype=torch.float32)
    return batched_graph, batched_labels.unsqueeze(1)

gnn_oof_preds = np.zeros(len(gnn_train_graphs))
gnn_test_preds = np.zeros(len(gnn_test_graphs))
kf_gnn = KFold(n_splits=N_SPLIT, shuffle=True, random_state=SEED)

print("\n--- GNN Model K-Fold Training ---")
for fold, (train_idx, val_idx) in tqdm(enumerate(kf_gnn.split(gnn_train_graphs)), total=N_SPLIT):
    train_graphs_fold = [gnn_train_graphs[i] for i in train_idx]
    train_labels_fold = [gnn_train_labels[i] for i in train_idx]
    val_graphs_fold = [gnn_train_graphs[i] for i in val_idx]
    val_labels_fold = [gnn_train_labels[i] for i in val_idx]

    train_dataset = GraphDataset(train_graphs_fold, train_labels_fold)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate)
    val_dataset = GraphDataset(val_graphs_fold, val_labels_fold)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate)
    test_dataset = GraphDataset(gnn_test_graphs)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate)

    in_dim = gnn_train_graphs[0].ndata['h'].shape[1]
    model = GATv2Set2Set(in_dim, hidden_dim=64, out_dim=1, num_heads=4).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    best_model_state = None

    for epoch in range(100):
        model.train()
        train_loss = 0
        for batched_graph, labels in train_dataloader:
            labels = labels.to(DEVICE)
            node_features = batched_graph.ndata['h'].float()
            optimizer.zero_grad()
            output = model(batched_graph, node_features)
            loss = loss_fn(output, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batched_graph, labels in val_dataloader:
                labels = labels.to(DEVICE)
                node_features = batched_graph.ndata['h'].float()
                output = model(batched_graph, node_features)
                loss = loss_fn(output, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    if best_model_state:
        model.load_state_dict(best_model_state)

    model.eval()
    fold_val_preds = []
    with torch.no_grad():
        for batched_graph, _ in val_dataloader:
            node_features = batched_graph.ndata['h'].float()
            output = model(batched_graph, node_features)
            fold_val_preds.extend(output.cpu().numpy().flatten())
    gnn_oof_preds[val_idx] = np.array(fold_val_preds)

    fold_test_preds = []
    with torch.no_grad():
        for batched_graph, _ in test_dataloader:
            node_features = batched_graph.ndata['h'].float()
            output = model(batched_graph, node_features)
            fold_test_preds.extend(output.cpu().numpy().flatten())
    gnn_test_preds += np.array(fold_test_preds) / N_SPLIT

nrmse_gnn, pearsonr_gnn, score_gnn = get_final_score(Y_full, gnn_oof_preds)
print("\n--- GNN Model Performance ---")
print(f"🎯 NRMSE: {nrmse_gnn:.5f}")
print(f"📈 Pearsonr: {pearsonr_gnn:.5f}")
print(f"⭐ SCORE: {score_gnn:.5f}")

#---------------------------------------
# 3. 최종 Residual 보정 앙상블 (Fingerprint 제외)
#---------------------------------------

# 2가지 모델의 OOF 및 테스트 예측값 결합
oof_base_ensemble = (oof_d_avg + gnn_oof_preds) / 2
test_base_ensemble = (test_d_avg + gnn_test_preds) / 2

nrmse_base, pearsonr_base, score_base = get_final_score(Y_full, oof_base_ensemble)
print("\n--- Base Ensemble Performance (2 Models) ---")
print(f"🎯 NRMSE: {nrmse_base:.5f}")
print(f"📈 Pearsonr: {pearsonr_base:.5f}")
print(f"⭐ SCORE: {score_base:.5f}")

# 잔차 계산
residual = Y_full - oof_base_ensemble

# 잔차 보정 모델의 입력 데이터 생성: DE, GNN 예측값 사용
X_full = pd.concat([Xd, pd.DataFrame(gnn_oof_preds, columns=['gnn_pred'])], axis=1)
test_full = pd.concat([testd, pd.DataFrame(gnn_test_preds, columns=['gnn_pred'])], axis=1)

scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X_full)
test_full_scaled = scaler.transform(test_full)

res_model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
res_model.fit(X_full_scaled, residual)

# 최종 예측
res_pred_oof = res_model.predict(X_full_scaled)
res_pred_test = res_model.predict(test_full_scaled)
final_oof = oof_base_ensemble + res_pred_oof
final_test = test_base_ensemble + res_pred_test

#---------------------------------------
# 4. 최종 평가 및 결과 저장
#---------------------------------------

nrmse_final, pearsonr_final, score_final = get_final_score(Y_full, final_oof)
print(f"\n=======================================")
print(f"--- 최종 Residual Ensemble Performance (Fingerprint 제외) ---")
print(f"🎯 NRMSE: {nrmse_final:.5f}")
print(f"📈 Pearsonr: {pearsonr_final:.5f}")
print(f"⭐ SCORE: {score_final:.5f}")
print(f"=======================================")

submission = pd.DataFrame({'id': gnn_test_ids})
submission['Inhibition'] = np.clip(final_test, 0, 100)
submission.to_csv(save_path + 'final_residual_ensemble_no_fp.csv', index=False)

print("\n모든 모델 학습 및 최종 앙상블 완료. Fingerprint를 제외한 결과 파일이 저장되었습니다.")
"""
