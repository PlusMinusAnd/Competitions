import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from torch.utils.data import DataLoader, Dataset
import pandas as pd


# ✅ Dataset 클래스
class GraphDataset(Dataset):
    def __init__(self, graph_path, csv_path, is_train=True):
        self.graphs, _ = dgl.load_graphs(graph_path)
        csv = pd.read_csv(csv_path)

        # ✅ 그래프에 self-loop 추가
        self.graphs = [dgl.add_self_loop(g) for g in self.graphs]

        if is_train:
            self.labels = torch.tensor(csv['Inhibition'].values, dtype=torch.float32)
        else:
            self.labels = None

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        if self.labels is not None:
            label = self.labels[idx]
            return g, label
        else:
            return g


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


# ✅ 데이터셋 및 DataLoader
train_dataset = GraphDataset('./Drug/_engineered_data/train_graph.bin', './Drug/train.csv')
test_dataset = GraphDataset('./Drug/_engineered_data/test_graph.bin', './Drug/test.csv', is_train=False)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate)


# ✅ GNN 모델
class GCNRegressor(nn.Module):
    def __init__(self, in_feats, hidden_feats):
        super(GCNRegressor, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_feats, hidden_feats, allow_zero_in_degree=True)
        self.fc = nn.Linear(hidden_feats, 1)

    def forward(self, g):
        h = g.ndata['h']
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        g.ndata['h_final'] = h
        hg = dgl.mean_nodes(g, 'h_final')
        return self.fc(hg).squeeze()


# ✅ 모델 생성
device = 'cpu'

in_feats = train_dataset[0][0].ndata['h'].shape[1]
model = GCNRegressor(in_feats, 64).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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


# ✅ 테스트 (예측)
model.eval()
all_preds = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        preds = model(batch)
        all_preds.append(preds.cpu())

preds = torch.cat(all_preds)

# ✅ 예측 결과 저장

sub_csv = pd.read_csv('./Drug/sample_submission.csv')
sub_csv['Inhibition'] = pd.DataFrame(preds.numpy())
sub_csv.to_csv('./Drug/submision_file.csv', index=False)

print("✔️ 예측 완료 → ./Drug/submision_file.csv 저장")
