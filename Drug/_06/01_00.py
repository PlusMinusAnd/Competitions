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
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import logit, expit

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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

submission = pd.read_csv(data_path + 'sample_submission.csv')

# print(train.columns)
# # Index(['Canonical_Smiles', 'Inhibition'], dtype='object')
# print(test.columns)
# # Index(['Canonical_Smiles'], dtype='object')
# print(submission.columns)
# # Index(['ID', 'Inhibition'], dtype='object')


import pandas as pd
from tqdm import tqdm

"""

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors

train = pd.read_csv(data_path + 'train.csv', index_col=0)
test = pd.read_csv(data_path + 'test.csv', index_col=0)
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

# 저장
train_final.to_csv(save_path + 'tr.csv', index=False)
test_final.to_csv(save_path + 'te.csv', index=False)
"""
"""
train = pd.read_csv(save_path +'tr.csv')
test = pd.read_csv(save_path +'te.csv')

train_df  = train.drop(['Canonical_Smiles'], axis=1)
test_df  = test.drop(['Canonical_Smiles'], axis=1)

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

de_train = de_train.drop(['Canonical_Smiles'], axis=1)
de_test = de_test.drop(['Canonical_Smiles'], axis=1)

Xd = de_train.drop(['Inhibition'], axis=1).copy()
Yd = de_train['Inhibition'].copy()
testd = de_test.copy()

Xd = Xd.fillna(Xd.median())
testd = testd.fillna(Xd.median()) 

y_scaled = Yd / 100.0
y_logit = logit(y_scaled.clip(1e-5, 1 - 1e-5))  # logit 변환

# print(y_logit)

xd_train, xd_test, yd_train, yd_test = train_test_split(
    Xd, y_logit, random_state=SEED, train_size=0.8
)

ss = StandardScaler()
xd_train = ss.fit_transform(xd_train)
xd_test = ss.transform(xd_test)
testd = ss.transform(testd)

# 분포 시각화를 위해 Yd를 복원


USE = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE else 'cpu')

xd_train = torch.tensor(xd_train, dtype=torch.float32).to(DEVICE)
xd_test = torch.tensor(xd_test, dtype=torch.float32).to(DEVICE)
yd_train = torch.tensor(yd_train.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)
yd_test = torch.tensor(yd_test.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)

# print(xd_train.size())  torch.Size([1344, 159])
# print(xd_test.size())   torch.Size([337, 159])
# print(yd_train.size())  torch.Size([1344, 1])
# print(yd_test.size())   

train_set = TensorDataset(xd_train, yd_train)
test_set = TensorDataset(xd_test, yd_test)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)


class DNN(nn.Module) :
    def __init__(self, feature):
        super().__init__()
        self.hl1 = nn.Sequential(
            nn.Linear(feature, 512),
            nn.ReLU(),
            nn.LazyBatchNorm1d(),
            nn.Dropout(0.3)
        )
        self.hl2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.LazyBatchNorm1d(),
            nn.Dropout(0.4)
        )
        self.hl3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LazyBatchNorm1d(),
            nn.Dropout(0.3)
        )
        self.hl4 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LazyBatchNorm1d(),
            nn.Dropout(0.3)
        )
        self.hl5 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.out = nn.Linear(64, 1)
        
    def forward(self, x) :
        x = self.hl1(x)
        x = self.hl2(x)
        x = self.hl3(x)
        x = self.hl4(x)
        x = self.hl5(x)
        x = self.out(x)
        return x

class Trainer :
    def __init__(self, model, criterion, optimizer, train_loader, test_loader):
        self.model = model.to(DEVICE)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.best_model = None
        
    def train(self) :
        self.model.train()
        epoch_loss = 0
        
        for x, y in self.train_loader :
            self.optimizer.zero_grad()
            hypo = self.model(x)
            loss = self.criterion(hypo, y)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(self.train_loader)
    
    def evaluate(self) :
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for x, y in self.test_loader :
                hypo = self.model(x)
                loss = self.criterion(hypo, y)
                
                epoch_loss += loss.item()
        return epoch_loss / len(self.test_loader)
    
    def fit(self, epochs=100, patience=10) :
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs+1) :
            loss = self.train()
            val_loss = self.evaluate()
            print(f"[Epoch {epoch}] Loss {loss:.6f} | Val_loss {val_loss:.6f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                self.best_model = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                    break

        # 모델 복원
        if self.best_model:
            self.model.load_state_dict(self.best_model)

model = DNN(159)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

trainer = Trainer(model, criterion, optimizer, train_loader, test_loader)
trainer.fit(10000)
loss = trainer.evaluate()

print(f"Loss {loss}")





