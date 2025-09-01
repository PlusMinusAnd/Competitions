# 03_train_gnn.py
# ✅ Set2Set 기반 GAT GNN 모델 학습

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GATConv, Set2Set
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import joblib
import datetime

r = 394
np.random.seed(r)
torch.manual_seed(r)

# 🔹 평가 함수

def evaluate_score(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    pearson = pearsonr(y_true, y_pred)[0]
    score = 0.5 * (1 - min(nrmse, 1)) + 0.5 * np.clip(pearson, 0, 1)
    return rmse, nrmse, pearson, score

# 🔹 Dataset 정의

class GraphDataset(Dataset):
    def __init__(self, graphs, labels=None):
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        if self.labels is not None:
            return g, torch.tensor(self.labels[idx], dtype=torch.float32)
        return g


# 🔹 GAT + Set2Set 모델

class GATSet2Set(nn.Module):
    def __init__(self, in_feats, hidden_size=128, num_heads=4):
        super().__init__()
        self.gat1 = GATConv(in_feats, hidden_size, num_heads)
        self.gat2 = GATConv(hidden_size * num_heads, hidden_size, 1)
        self.readout = Set2Set(hidden_size, n_iters=6, n_layers=1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, g):
        h = g.ndata['h']
        h = self.gat1(g, h).flatten(1)
        h = F.elu(h)
        h = self.gat2(g, h).squeeze(1)
        g.ndata['h'] = h
        hg = self.readout(g, h)
        return self.mlp(hg).squeeze(-1)


# 🔹 학습 루프

def train():
    train_graphs = joblib.load("./Drug/_02/full_pipeline/0_dataset/train_graphs.pkl")
    test_graphs = joblib.load("./Drug/_02/full_pipeline/0_dataset/test_graphs.pkl")
    y = pd.read_csv("./Drug/train.csv")["Inhibition"].values

    oof = np.zeros(len(train_graphs))
    test_preds = []

    device = torch.device("cpu")
    kf = KFold(n_splits=5, shuffle=True, random_state=r)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_graphs)):
        tr_g = [train_graphs[i] for i in train_idx]
        val_g = [train_graphs[i] for i in val_idx]
        y_tr = y[train_idx]
        y_val = y[val_idx]

        model = GATSet2Set(train_graphs[0].ndata['h'].shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        train_loader = DataLoader(GraphDataset(tr_g, y_tr), batch_size=64, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(GraphDataset(val_g, y_val), batch_size=64, shuffle=False, collate_fn=collate)
        test_loader = DataLoader(GraphDataset(test_graphs), batch_size=64, shuffle=False, collate_fn=lambda x: dgl.batch(x))

        best_rmse = float('inf')
        patience, counter = 10, 0

        for epoch in range(100):
            model.train()
            for bg, yb in train_loader:
                bg, yb = bg.to(device), yb.to(device)
                pred = model(bg)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            val_preds = []
            with torch.no_grad():
                for bg, _ in val_loader:
                    bg = bg.to(device)
                    val_preds.append(model(bg).cpu().numpy())

            val_preds = np.concatenate(val_preds)
            rmse, *_ = evaluate_score(y_val, val_preds)
            print(f"Fold {fold+1} Epoch {epoch+1} | RMSE: {rmse:.4f}")

            if rmse < best_rmse:
                best_rmse = rmse
                counter = 0
                torch.save(model.state_dict(), f"./Drug/_02/full_pipeline/gat_fold{fold}.pt")
            else:
                counter += 1
                if counter >= patience:
                    print("Early Stopping")
                    break

        # Predict
        model.load_state_dict(torch.load(f"./Drug/_02/full_pipeline/gat_fold{fold}.pt"))
        model.eval()

        val_preds = []
        with torch.no_grad():
            for bg, _ in val_loader:
                bg = bg.to(device)
                val_preds.append(model(bg).cpu().numpy())
        oof[val_idx] = np.concatenate(val_preds)

        test_fold_preds = []
        with torch.no_grad():
            for bg in test_loader:
                bg = bg.to(device)
                test_fold_preds.append(model(bg).cpu().numpy())
        test_preds.append(np.concatenate(test_fold_preds))

    # 앙상블 평균
    test_final = np.mean(test_preds, axis=0)

    # 저장
    np.save("./Drug/_02/full_pipeline/gat_oof.npy", oof)
    np.save("./Drug/_02/full_pipeline/gat_preds.npy", test_final)
    print("✅ GNN 저장 완료")

    rmse, nrmse, pearson, score = evaluate_score(y, oof)
    print(f"📊 GNN 최종 점수")
    print(f"RMSE     : {rmse:.5f}")
    print(f"NRMSE    : {nrmse:.5f}")
    print(f"Pearson  : {pearson:.5f}")
    print(f"Score📈  : {score:.5f}")


def collate(batch):
    graphs, labels = zip(*batch)
    return dgl.batch(graphs), torch.stack(labels)


if __name__ == "__main__":
    train()