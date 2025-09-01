import os
import random
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
import joblib


# ✅ 시드 고정
r = 73
random.seed(r)
np.random.seed(r)
torch.manual_seed(r)
if torch.cuda.is_available():
    torch.cuda.manual_seed(r)
    torch.cuda.manual_seed_all(r)
dgl.random.seed(r)


# =============================
# ✅ Graph Dataset 클래스
# =============================
class GraphDataset(Dataset):
    def __init__(self, graph_path, csv_path, is_train=True):
        self.graphs, _ = dgl.load_graphs(graph_path)
        csv = pd.read_csv(csv_path)

        self.graphs = [dgl.add_self_loop(g) for g in self.graphs]

        if is_train:
            self.labels = torch.tensor(csv['Inhibition'].values, dtype=torch.float32)
        else:
            self.labels = None

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.graphs[idx], self.labels[idx]
        else:
            return self.graphs[idx]


# =============================
# ✅ collate 함수
# =============================
def collate(batch):
    if isinstance(batch[0], tuple):
        graphs, labels = map(list, zip(*batch))
        batched_graph = dgl.batch(graphs)
        labels = torch.tensor(labels, dtype=torch.float32)
        return batched_graph, labels
    else:
        batched_graph = dgl.batch(batch)
        return batched_graph


# =============================
# ✅ GCN 모델
# =============================
class GCNRegressor(nn.Module):
    def __init__(self, in_feats, hidden_feats, dropout=0.3):
        super(GCNRegressor, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_feats, hidden_feats, allow_zero_in_degree=True)
        self.conv3 = GraphConv(hidden_feats, hidden_feats, allow_zero_in_degree=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_feats, 1)

    def forward(self, g):
        h = g.ndata['h']
        h = F.relu(self.conv1(g, h))
        h = self.dropout(h)
        h = F.relu(self.conv2(g, h))
        h = self.dropout(h)
        h = F.relu(self.conv3(g, h))
        g.ndata['h_final'] = h
        hg = dgl.mean_nodes(g, 'h_final')
        return self.fc(hg).squeeze()


# =============================
# ✅ 데이터 로드
# =============================
train_graph_dataset = GraphDataset('./Drug/_engineered_data/train_graph.bin', './Drug/_engineered_data/filled_train_final.csv', is_train=True)
test_graph_dataset = GraphDataset('./Drug/_engineered_data/test_graph.bin', './Drug/_engineered_data/test_final.csv', is_train=False)

train_tabular = pd.read_csv('./Drug/_engineered_data/filled_train_final.csv')
test_tabular = pd.read_csv('./Drug/_engineered_data/test_final.csv')

device = 'cpu'
in_feats = train_graph_dataset[0][0].ndata['h'].shape[1]


# =============================
# ✅ 폴더 생성
# =============================
os.makedirs('./Drug/_submission_files', exist_ok=True)
os.makedirs('./Drug/_models', exist_ok=True)


# =============================
# ✅ GNN 모델 학습 및 예측
# =============================
kf = KFold(n_splits=5, shuffle=True, random_state=r)
fold = 1
train_losses = []
val_losses = []
gnn_fold_preds = []

for train_index, val_index in kf.split(train_graph_dataset):
    print(f"\n🚀 GNN Fold {fold} 시작")

    train_subset = torch.utils.data.Subset(train_graph_dataset, train_index)
    val_subset = torch.utils.data.Subset(train_graph_dataset, val_index)

    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_graph_dataset, batch_size=16, shuffle=False, collate_fn=collate)

    model = GCNRegressor(in_feats, 128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    # ✅ 학습
    for epoch in range(50):
        model.train()
        total_loss = 0

        for batch, labels in train_loader:
            batch = batch.to(device)
            labels = labels.to(device)

            preds = model(batch)
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")

    train_losses.append(avg_loss)

    # ✅ Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch, labels in val_loader:
            batch = batch.to(device)
            labels = labels.to(device)
            preds = model(batch)
            loss = loss_fn(preds, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"✅ GNN Fold {fold} | Validation Loss: {avg_val_loss:.4f}")

    # ✅ 테스트 예측
    fold_preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds = model(batch)
            fold_preds.append(preds.cpu())

    fold_preds = torch.cat(fold_preds)
    gnn_fold_preds.append(fold_preds)

    # ✅ 모델 저장
    torch.save(model.state_dict(), f'./Drug/_models/gnn_model_fold{fold}.pt')

    fold += 1


gnn_preds = torch.stack(gnn_fold_preds).mean(dim=0)


# =============================
# ✅ DNN 모델 학습 및 예측
# =============================
drop_cols_train = [col for col in ['index', 'SMILES'] if col in train_tabular.columns]
drop_cols_test = [col for col in ['index', 'SMILES', 'Inhibition'] if col in test_tabular.columns]

X_train = train_tabular.drop(columns=drop_cols_train + ['Inhibition'])
y_train = train_tabular['Inhibition']
X_test = test_tabular.drop(columns=drop_cols_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

dnn_model = RandomForestRegressor(n_estimators=500, random_state=r)
dnn_model.fit(X_train_scaled, y_train)

dnn_preds = dnn_model.predict(X_test_scaled)
dnn_preds = torch.tensor(dnn_preds)

# ✅ 모델 저장
joblib.dump(dnn_model, './Drug/_models/dnn_model.pkl')
joblib.dump(scaler, './Drug/_models/dnn_scaler.pkl')


# =============================
# ✅ 가중 앙상블
# =============================
alpha = 0.7
beta = 0.3

ensemble_preds = (alpha * gnn_preds.numpy()) + (beta * dnn_preds.numpy())


# =============================
# ✅ 성능 출력
# =============================
print("\n============================")
print("✅ GNN Fold별 Train Loss:", train_losses)
print("✅ GNN Fold별 Val Loss:", val_losses)
print(f"✅ GNN 평균 Val Loss: {np.mean(val_losses):.4f}")
print("============================")

# (참고) GNN과 DNN은 테스트 데이터에서 ground truth가 없으므로 여기서는 MSE 비교가 아닌
# 모델 출력 간 차이로만 확인 가능

print("\n✅ 각 모델 테스트 예측 비교:")
print(f"GNN 예측 평균: {gnn_preds.mean().item():.4f}")
print(f"DNN 예측 평균: {dnn_preds.mean().item():.4f}")
print(f"앙상블 예측 평균: {ensemble_preds.mean():.4f}")
print("============================")
print(f"RandomState = {r}")

# =============================
# ✅ 제출 파일 저장
# =============================
now = datetime.now().strftime('%Y%m%d_%H%M')
submission_path = f'./Drug/_submission_files/submission_ensemble_{now}.csv'

sub_csv = pd.read_csv('./Drug/sample_submission.csv')
sub_csv['Inhibition'] = ensemble_preds
sub_csv.to_csv(submission_path, index=False)

print(f"\n✔️ 앙상블 제출 파일 저장 완료 → {submission_path}")
