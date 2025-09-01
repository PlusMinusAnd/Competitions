##### DMPNN 모델 구성 #####

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.dataloading import GraphDataLoader
from torch.utils.data import Dataset, DataLoader
import joblib
import numpy as np
import pandas as pd
import random
from sklearn.metrics import mean_squared_error
import os

# 시드 고정
def set_seed(seed=73):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed()

# -------------------------------
# 데이터셋 정의
# -------------------------------
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

# -------------------------------
# D-MPNN 모델 정의
# -------------------------------
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

# -------------------------------
# 학습 루프
# -------------------------------
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        g, y = batch
        g = g.to(device)
        y = y.to(device)
        pred = model(g)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(dataloader.dataset)

def evaluate_model(model, dataloader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, tuple):
                g, y = batch
                y = y.to(device)
            else:
                g = batch
                y = None
            g = g.to(device)
            pred = model(g)
            preds.append(pred.cpu().numpy())
            if y is not None:
                trues.append(y.cpu().numpy())

    preds = np.concatenate(preds)
    if trues:
        trues = np.concatenate(trues)
        rmse = np.sqrt(mean_squared_error(trues, preds))
        return rmse, preds
    return None, preds

# 그래프 + 레이블 배치용 collate 함수 정의
def collate_graphs(batch):
    graphs, labels = zip(*batch)  # 튜플을 분리
    batched_graph = dgl.batch(graphs)
    labels = torch.stack(labels)  # 텐서로 변환
    return batched_graph, labels


# -------------------------------
# 메인 실행
# -------------------------------
def main():
    # 경로 설정
    train_graph_path = "./Drug/_02/0_dataset/train_graphs.pkl"
    test_graph_path = "./Drug/_02/0_dataset/test_graphs.pkl"
    train_csv_path = "./Drug/train.csv"
    test_csv_path = "./Drug/test.csv"

    # 데이터 불러오기
    train_graphs = joblib.load(train_graph_path)
    test_graphs = joblib.load(test_graph_path)
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    y = train_df['Inhibition'].values

    # Dataset & DataLoader
    train_dataset = GraphDataset(train_graphs, y)
    test_dataset = GraphDataset(test_graphs)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_graphs)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=dgl.batch)

    # 모델 초기화
    device = torch.device("cpu")
    model = DMPNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 학습
    for epoch in range(1, 51):
        loss = train_model(model, train_loader, optimizer, criterion, device)
        rmse, _ = evaluate_model(model, train_loader, device)
        print(f"Epoch {epoch:03d} | Train Loss: {loss:.4f} | RMSE: {rmse:.4f}")

    # 추론
    _, test_preds = evaluate_model(model, test_loader, device)
    submission = pd.DataFrame({'ID': test_df['ID'], 'Inhibition': test_preds})
    submission.to_csv("./Drug/_02/1_dmpnn/submission_dmpnn.csv", index=False)
    print("✅ submission_dmpnn.csv 저장 완료")

if __name__ == "__main__":
    main()
