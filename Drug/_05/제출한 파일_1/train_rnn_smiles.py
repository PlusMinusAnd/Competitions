import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os
import sys
from tqdm import tqdm
from scipy.stats import pearsonr

# ✅ 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_utils')))
from smiles_tokenizer import build_vocab, encode_smiles
from set_seed import set_seed

# 시드 고정
seed = set_seed()
print('고정된 SEED :', seed)

# ✅ 기본 설정
device = torch.device("cpu")
max_len = 128
batch_size = 32
epochs = 5
lr = 1e-3
save_dir = './Drug/_05/mk_data'
os.makedirs(save_dir, exist_ok=True)

# ✅ 데이터 불러오기
train_df = pd.read_csv('./Drug/_05/data/train.csv')
test_df = pd.read_csv('./Drug/_05/data/test.csv')

train_smiles = train_df['Canonical_Smiles'].tolist()
test_smiles = test_df['Canonical_Smiles'].tolist()
y = train_df['Inhibition'].values
train_ids = train_df['ID']
test_ids = test_df['ID']

# ✅ 문자 vocab 생성 및 인코딩
stoi, itos = build_vocab(train_smiles + test_smiles)
x = np.array(encode_smiles(train_smiles, stoi, max_len))      # 수정
x_test = np.array(encode_smiles(test_smiles, stoi, max_len))  # 수정

# ✅ Dataset 정의
class SMILESDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x[idx], self.y[idx]
        return self.x[idx]

# ✅ 모델 정의 (Simple RNN)
class SMILESRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.rnn(x)
        return self.fc(h[-1]).squeeze(1)

# ✅ 평가 함수
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
    return score

# ✅ KFold 학습
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_smiles))
test_preds = []

for fold, (train_idx, val_idx) in enumerate(kf.split(x)):
    print(f"\n🚀 Fold {fold}")
    model = SMILESRNN(vocab_size=len(stoi)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_ds = SMILESDataset(x[train_idx], y[train_idx])
    val_ds = SMILESDataset(x[val_idx], y[val_idx])
    test_ds = SMILESDataset(x_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # === 학습 ===
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # === OOF 예측
    model.eval()
    val_pred = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            pred = model(xb)
            val_pred.extend(pred.cpu().numpy())
    oof_preds[val_idx] = val_pred

    # === Test 예측
    fold_test_preds = []
    with torch.no_grad():
        for xb in test_loader:
            xb = xb.to(device)
            pred = model(xb)
            fold_test_preds.extend(pred.cpu().numpy())
    test_preds.append(fold_test_preds)

# ✅ 저장

final_test_preds = np.mean(test_preds, axis=0)

pd.DataFrame({'ID': train_ids, 'RNN_OOF': oof_preds}).to_csv(f'{save_dir}/rnn_oof.csv', index=False)
pd.DataFrame({'ID': test_ids, 'RNN_Pred': final_test_preds}).to_csv(f'{save_dir}/rnn_preds.csv', index=False)
print(f"✅ 저장 완료: {save_dir}/rnn_oof.csv, {save_dir}/rnn_preds.csv")
