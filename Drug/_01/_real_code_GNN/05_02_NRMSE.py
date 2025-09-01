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
from sklearn.metrics import mean_squared_error
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
# ✅ Dataset 클래스
# =============================
class GraphDataset(Dataset):
    def __init__(self, graph_path, csv_path):
        self.graphs, _ = dgl.load_graphs(graph_path)
        csv = pd.read_csv(csv_path)

        self.graphs = [dgl.add_self_loop(g) for g in self.graphs]
        self.labels = torch.tensor(csv['Inhibition'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]


# =============================
# ✅ collate 함수
# =============================
def collate(batch):
    graphs, labels = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels, dtype=torch.float32)
    return batched_graph, labels


# =============================
# ✅ GCN 모델
# =============================
class GCNRegressor(nn.Module):
    def __init__(self, in_feats, hidden_feats=128, dropout=0.3):
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
        return self.fc(hg)


# =============================
# ✅ NRMSE 함수
# =============================
def compute_nrmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse / (y_true.max() - y_true.min())


# =============================
# ✅ 데이터 로드
# =============================
train_graph_dataset = GraphDataset('./Drug/_engineered_data/train_graph.bin', './Drug/_engineered_data/filled_train_final.csv')
test_graph_dataset = GraphDataset('./Drug/_engineered_data/test_graph.bin', './Drug/_engineered_data/test_final.csv')

train_tabular = pd.read_csv('./Drug/_engineered_data/filled_train_final.csv')
test_tabular = pd.read_csv('./Drug/_engineered_data/test_final.csv')

X = train_tabular.drop(columns=['index', 'SMILES', 'Inhibition'], errors='ignore')
y = train_tabular['Inhibition'].values

y_max = y.max()
y_min = y.min()

X_test = test_tabular.drop(columns=['index', 'SMILES', 'Inhibition'], errors='ignore')

# ✅ 저장 폴더
os.makedirs('./Drug/_submission_files', exist_ok=True)
os.makedirs('./Drug/_models', exist_ok=True)


# =============================
# ✅ KFold 시작
# =============================
kf = KFold(n_splits=5, shuffle=True, random_state=r)
fold = 1

gnn_nrmse_list = []
dnn_nrmse_list = []
ensemble_nrmse_list = []

gnn_test_fold_preds = []
dnn_test_fold_preds = []

for train_idx, val_idx in kf.split(X):
    print(f"\n🚀 Fold {fold} 시작")

    # ✅ Graph Dataset Split
    train_graph = torch.utils.data.Subset(train_graph_dataset, train_idx)
    val_graph = torch.utils.data.Subset(train_graph_dataset, val_idx)

    train_loader = DataLoader(train_graph, batch_size=16, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_graph, batch_size=16, shuffle=False, collate_fn=collate)

    # ✅ Tabular Dataset Split
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # ✅ GNN 모델
    in_feats = train_graph_dataset[0][0].ndata['h'].shape[1]
    gnn_model = GCNRegressor(in_feats).to('cpu')

    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(50):
        gnn_model.train()
        total_loss = 0
        for batch, labels in train_loader:
            batch = batch.to('cpu')
            labels = labels.unsqueeze(1)

            preds = gnn_model(batch)
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | GNN Train Loss: {avg_loss:.4f}")

    # ✅ GNN 검증
    gnn_model.eval()
    gnn_preds_list = []
    val_labels_list = []

    with torch.no_grad():
        for batch, labels in val_loader:
            batch = batch.to('cpu')
            labels = labels.unsqueeze(1)

            preds = gnn_model(batch)

            gnn_preds_list.append(preds.numpy().flatten())
            val_labels_list.append(labels.numpy().flatten())

    gnn_preds = np.concatenate(gnn_preds_list)
    val_labels = np.concatenate(val_labels_list)

    gnn_nrmse = compute_nrmse(val_labels, gnn_preds)
    gnn_nrmse_list.append(gnn_nrmse)

    print(f"✅ GNN Fold {fold} NRMSE: {gnn_nrmse:.4f}")

    # ✅ DNN 모델
    dnn_model = RandomForestRegressor(n_estimators=500, random_state=r)
    dnn_model.fit(X_train_scaled, y_train)

    dnn_preds = dnn_model.predict(X_val_scaled)

    dnn_nrmse = compute_nrmse(val_labels, dnn_preds)
    dnn_nrmse_list.append(dnn_nrmse)

    print(f"✅ DNN Fold {fold} NRMSE: {dnn_nrmse:.4f}")

    # ✅ 앙상블
    alpha = 0.7
    beta = 0.3
    ensemble_preds = alpha * gnn_preds + beta * dnn_preds

    ensemble_nrmse = compute_nrmse(val_labels, ensemble_preds)
    ensemble_nrmse_list.append(ensemble_nrmse)

    print(f"✅ Ensemble Fold {fold} NRMSE: {ensemble_nrmse:.4f}")

    # ✅ 테스트 데이터 예측
    test_loader = DataLoader(test_graph_dataset, batch_size=16, shuffle=False, collate_fn=collate)

    gnn_model.eval()
    gnn_test_preds = []
    with torch.no_grad():
        for batch, _ in test_loader:
            batch = batch.to('cpu')
            preds = gnn_model(batch)
            gnn_test_preds.append(preds.numpy().flatten())

    gnn_test_preds = np.concatenate(gnn_test_preds)
    gnn_test_fold_preds.append(gnn_test_preds)

    X_test_scaled = scaler.transform(X_test)
    dnn_test_preds = dnn_model.predict(X_test_scaled)
    dnn_test_fold_preds.append(dnn_test_preds)

    # ✅ 모델 저장
    torch.save(gnn_model.state_dict(), f'./Drug/_models/gnn_model_fold{fold}.pt')
    joblib.dump(dnn_model, f'./Drug/_models/dnn_model_fold{fold}.pkl')
    joblib.dump(scaler, f'./Drug/_models/scaler_fold{fold}.pkl')

    fold += 1


# =============================
# ✅ 테스트 예측 앙상블
# =============================
gnn_test_final = np.mean(gnn_test_fold_preds, axis=0)
dnn_test_final = np.mean(dnn_test_fold_preds, axis=0)

ensemble_test_preds = (alpha * gnn_test_final) + (beta * dnn_test_final)


# =============================
# ✅ 결과 출력 및 저장
# =============================
print("\n============================")
print(f"✅ GNN NRMSE: {gnn_nrmse_list}")
print(f"✅ DNN NRMSE: {dnn_nrmse_list}")
print(f"✅ Ensemble NRMSE: {ensemble_nrmse_list}")
print(f"✅ 평균 GNN NRMSE: {np.mean(gnn_nrmse_list):.4f}")
print(f"✅ 평균 DNN NRMSE: {np.mean(dnn_nrmse_list):.4f}")
print(f"✅ 평균 Ensemble NRMSE: {np.mean(ensemble_nrmse_list):.4f}")
print(f"✅ Random State: {r}")
print("============================")


# ✅ CSV 저장
now = datetime.now().strftime('%Y%m%d_%H%M')
result_path = f'./Drug/_submission_files/ensemble_result_{now}.csv'

result_df = pd.DataFrame({
    'Fold': list(range(1, 6)),
    'GNN_NRMSE': gnn_nrmse_list,
    'DNN_NRMSE': dnn_nrmse_list,
    'Ensemble_NRMSE': ensemble_nrmse_list
})
result_df.loc['Mean'] = ['Mean', np.mean(gnn_nrmse_list), np.mean(dnn_nrmse_list), np.mean(ensemble_nrmse_list)]

result_df.to_csv(result_path, index=False)
print(f"\n✔️ 결과 저장 완료 → {result_path}")


# ✅ Submission 저장
submission_path = f'./Drug/_submission_files/submission_ensemble_{now}.csv'
sub_csv = pd.read_csv('./Drug/sample_submission.csv')
sub_csv['Inhibition'] = ensemble_test_preds
sub_csv.to_csv(submission_path, index=False)

print(f"\n✔️ 앙상블 제출 파일 저장 완료 → {submission_path}")
