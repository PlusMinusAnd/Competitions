import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATv2Conv
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
import os
import sys

# utils 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_utils')))
from graph_utils import smiles_to_dgl
from set_seed import set_seed

# 시드 고정
seed = set_seed()
print('고정된 SEED :', seed)

device = torch.device("cpu")

# === 데이터 로딩 ===
train_df = pd.read_csv('./Drug/_05/data/train.csv')
test_df = pd.read_csv('./Drug/_05/data/test.csv')
train_smiles = train_df['Canonical_Smiles'].tolist()
test_smiles = test_df['Canonical_Smiles'].tolist()
y = train_df['Inhibition'].values
train_ids = train_df['ID']
test_ids = test_df['ID']

# === 그래프 변환 ===
train_graphs = [smiles_to_dgl(s) for s in tqdm(train_smiles)]
test_graphs = [smiles_to_dgl(s) for s in tqdm(test_smiles)]

# None 제거
train_filtered = [(g, y[i], train_ids[i]) for i, g in enumerate(train_graphs) if g is not None]
test_filtered = [(g, test_ids[i]) for i, g in enumerate(test_graphs) if g is not None]

train_graphs, y, train_ids = zip(*train_filtered)
test_graphs, test_ids = zip(*test_filtered)
y = np.array(y)

# Self-loop
train_graphs = [dgl.add_self_loop(g) for g in train_graphs]
test_graphs = [dgl.add_self_loop(g) for g in test_graphs]

# === Dataset 정의 ===
class MoleculeDataset(Dataset):
    def __init__(self, graphs, labels=None):
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.graphs[idx], torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            return self.graphs[idx]

def collate_graphs(batch):
    if isinstance(batch[0], tuple):
        graphs, labels = zip(*batch)
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.stack(labels)
    else:
        return dgl.batch(batch)

# === 모델 정의 ===
class GATv2Net(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_heads=4):
        super().__init__()
        self.gat1 = GATv2Conv(in_dim, hidden_dim, num_heads, allow_zero_in_degree=True)
        self.gat2 = GATv2Conv(hidden_dim * num_heads, hidden_dim, 1, allow_zero_in_degree=True)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, g):
        h = g.ndata['h']
        h = self.gat1(g, h)
        h = h.view(h.size(0), -1)
        h = self.gat2(g, h)
        h = h.mean(1)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return self.readout(hg).squeeze()

# === 학습 파라미터 ===
epochs = 20
lr = 1e-3
batch_size = 32
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

oof_preds = np.zeros(len(train_graphs))
test_preds = []
save_dir = './Drug/_05/mk_data'
os.makedirs(save_dir, exist_ok=True)

for fold, (train_idx, val_idx) in enumerate(kf.split(train_graphs)):
    model = GATv2Net(in_dim=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_dataset = MoleculeDataset([train_graphs[i] for i in train_idx], y[train_idx])
    val_dataset = MoleculeDataset([train_graphs[i] for i in val_idx], y[val_idx])
    test_dataset = MoleculeDataset(test_graphs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_graphs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)

    for epoch in range(epochs):
        model.train()
        for g_batch, y_batch in train_loader:
            g_batch = g_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(g_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # === OOF 예측 ===
    model.eval()
    val_preds = []
    with torch.no_grad():
        for g_batch, _ in val_loader:
            g_batch = g_batch.to(device)
            pred = model(g_batch).cpu().numpy()
            val_preds.extend(pred)
    oof_preds[val_idx] = val_preds

    # === Test 예측 ===
    fold_test_preds = []
    with torch.no_grad():
        for g_batch in test_loader:
            g_batch = g_batch.to(device)
            pred = model(g_batch).cpu().numpy()
            fold_test_preds.extend(pred)
    test_preds.append(fold_test_preds)

# 평균
final_test_preds = np.mean(test_preds, axis=0)

# 저장
pd.DataFrame({'ID': train_ids, 'GNN_OOF': oof_preds}).to_csv(f'{save_dir}/gnn_oof.csv', index=False)
pd.DataFrame({'ID': test_ids, 'GNN_Pred': final_test_preds}).to_csv(f'{save_dir}/gnn_preds.csv', index=False)
print(f"✅ 저장 완료: {save_dir}/gnn_oof.csv, {save_dir}/gnn_preds.csv")
