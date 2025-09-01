# 02_model_fingerprint/train_fingerprint_dnn.py

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import sys
from rdkit import RDLogger
import warnings

RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings('ignore')

# utils 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_utils')))
from rdkit_features import get_morgan_fingerprint
from set_seed import set_seed

# 시드 고정
seed = set_seed()
print('고정된 SEED :', seed)

device = torch.device("cpu")

# 경로
train_path = './Drug/_05/data/train.csv'
test_path  = './Drug/_05/data/test.csv'
save_dir   = './Drug/_05/mk_data'
os.makedirs(save_dir, exist_ok=True)

# 데이터 로딩
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_smiles = train_df['Canonical_Smiles'].tolist()
test_smiles = test_df['Canonical_Smiles'].tolist()
y = train_df['Inhibition'].values
train_ids = train_df['ID']
test_ids = test_df['ID']

# 피처 생성
x = get_morgan_fingerprint(train_smiles)
x_test = get_morgan_fingerprint(test_smiles)

# Dataset 정의
class FingerprintDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx]

# MLP 모델 정의
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

# 학습 파라미터
epochs = 20
batch_size = 64
lr = 1e-3
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

# 결과 저장용
oof_preds = np.zeros(len(train_df))
test_preds = []

# KFold 학습
for fold, (train_idx, val_idx) in enumerate(kf.split(x)):
    model = MLP(input_dim=x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_dataset = FingerprintDataset(x[train_idx], y[train_idx])
    val_dataset = FingerprintDataset(x[val_idx], y[val_idx])
    test_dataset = FingerprintDataset(x_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # OOF 예측
    model.eval()
    val_preds = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            val_preds.extend(pred)
    oof_preds[val_idx] = val_preds

    # Test 예측
    test_fold_preds = []
    with torch.no_grad():
        for xb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            test_fold_preds.extend(pred)
    test_preds.append(test_fold_preds)

# 평균 test 예측
final_test_preds = np.mean(test_preds, axis=0)

# 저장

pd.DataFrame({'ID': train_ids, 'Fingerprint_OOF': oof_preds}).to_csv(f'{save_dir}/fingerprint_oof.csv', index=False)
pd.DataFrame({'ID': test_ids, 'Fingerprint_Pred': final_test_preds}).to_csv(f'{save_dir}/fingerprint_preds.csv', index=False)

print(f"✅ 저장 완료:")
print(f" - OOF : {save_dir}/fingerprint_oof.csv")
print(f" - TEST: {save_dir}/fingerprint_preds.csv")
