import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from datetime import datetime
import os


# ✅ Dataset 클래스
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


# ✅ collate 함수
def collate(batch):
    if isinstance(batch[0], tuple):
        graphs, labels = map(list, zip(*batch))
        batched_graph = dgl.batch(graphs)
        labels = torch.tensor(labels, dtype=torch.float32)
        return batched_graph, labels
    else:
        batched_graph = dgl.batch(batch)
        return batched_graph


# ✅ GNN 모델
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


# ✅ 데이터셋 로드
train_dataset = GraphDataset('./Drug/_engineered_data/train_graph.bin', './Drug/train.csv', is_train=True)
test_dataset = GraphDataset('./Drug/_engineered_data/test_graph.bin', './Drug/test.csv', is_train=False)

device = 'cpu'
in_feats = train_dataset[0][0].ndata['h'].shape[1]

# ✅ Dataloader
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate)

# ✅ KFold 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold = 1
val_losses = []
all_fold_preds = []

for train_index, val_index in kf.split(train_dataset):
    print(f"\n🚀 Fold {fold} 시작")

    train_subset = torch.utils.data.Subset(train_dataset, train_index)
    val_subset = torch.utils.data.Subset(train_dataset, val_index)

    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False, collate_fn=collate)

    # ✅ 모델 초기화
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

    # ✅ 검증
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

    print(f"✅ Fold {fold} | Validation Loss: {avg_val_loss:.4f}")

    # ✅ 테스트셋 예측
    fold_preds = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds = model(batch)
            fold_preds.append(preds.cpu())

    fold_preds = torch.cat(fold_preds)
    all_fold_preds.append(fold_preds)

    fold += 1


# ✅ 모든 Fold 예측 → 평균 앙상블
ensemble_preds = torch.stack(all_fold_preds).mean(dim=0)

# ✅ 제출 파일 저장
os.makedirs('./Drug/submission_files', exist_ok=True)

now = datetime.now().strftime('%Y%m%d_%H%M')
submission_path = f'./Drug/submission_files/submission_{now}.csv'

sub_csv = pd.read_csv('./Drug/sample_submission.csv')
sub_csv['Inhibition'] = pd.DataFrame(ensemble_preds.numpy())
sub_csv.to_csv(submission_path, index=False)

print(f"\n✔️ 앙상블 예측 저장 완료 → {submission_path}")


# ✅ KFold 결과 저장
os.makedirs('./Drug/kfold_result_files', exist_ok=True)

result_path = f'./Drug/kfold_result_files/kfold_result_{now}.csv'

result_df = pd.DataFrame({
    'Fold': list(range(1, len(val_losses) + 1)),
    'Validation_Loss': val_losses
})
result_df.loc['Mean'] = ['Mean', np.mean(val_losses)]

result_df.to_csv(result_path, index=False)

print(f"✔️ Fold 결과 저장 완료 → {result_path}")
