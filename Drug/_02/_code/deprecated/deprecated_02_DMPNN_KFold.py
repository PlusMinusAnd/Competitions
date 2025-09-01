
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from torch.utils.data import Dataset, DataLoader
import joblib
import numpy as np
import pandas as pd
import random
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import os

# 시드 고정
def set_seed(seed=73):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed()

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

class DMPNN(nn.Module):
    def __init__(self, in_node_feats=6, in_edge_feats=3, hidden_size=128):
        super(DMPNN, self).__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_edge_feats + in_node_feats * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def message_func(self, edges):
        return {'m': edges.data['e']}

    def forward(self, g):
        with g.local_scope():
            h = g.ndata['h']
            e = g.edata['e']
            g.apply_edges(lambda edges: {'e': self.edge_mlp(
                torch.cat([edges.src['h'], edges.dst['h'], edges.data['e']], dim=1)
            )})
            g.update_all(self.message_func, dgl.function.mean('m', 'neigh'))
            hg = dgl.mean_nodes(g, 'neigh')
            return self.readout(hg).squeeze()

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
    # 경로 설정
    train_graphs = joblib.load("./Drug/_02/0_dataset/train_graphs.pkl")
    test_graphs = joblib.load("./Drug/_02/0_dataset/test_graphs.pkl")
    train_df = pd.read_csv("./Drug/train.csv")
    test_df = pd.read_csv("./Drug/test.csv")
    y = train_df["Inhibition"].values

    # 기본 설정
    device = torch.device("cpu")
    n_splits = 5
    oof = np.zeros(len(train_graphs))
    test_preds = np.zeros(len(test_graphs))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=73)

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

        model = DMPNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        for epoch in range(1, 21):
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

        # validation 예측
        val_graph_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_graphs)
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch in val_graph_loader:
                g, _ = batch
                g = g.to(device)
                pred = model(g)
                val_preds.append(pred.cpu().numpy())
        oof[val_idx] = np.concatenate(val_preds)

        # test 예측
        model.eval()
        test_preds += evaluate(model, test_loader, device) / n_splits

    # 저장
    np.save("./Drug/_02/2_npy/preprocessing_dmpnn_oof.npy", oof)
    np.save("./Drug/_02/2_npy/preprocessing_dmpnn_preds.npy", test_preds)

    submission = pd.DataFrame({
        "ID": test_df["ID"],
        "Inhibition": test_preds
    })
    submission.to_csv("./Drug/_02/1_dmpnn/preprocessing_dmpnn.csv", index=False)
    print("✅ dmpnn_oof.npy, dmpnn_preds.npy, preprocessing_dmpnn.csv 저장 완료")

if __name__ == "__main__":
    main()
