import os, random, datetime, joblib, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
import dgl
from rdkit import Chem
from rdkit.Chem import Descriptors, rdchem
from rdkit.Chem import GetPeriodicTable

warnings.filterwarnings("ignore")

r = random.randint(1, 1000)
random.seed(r)
np.random.seed(r)
torch.manual_seed(r)

train_path = "./Drug/train.csv"
test_path = "./Drug/test.csv"
save_dir = "./Drug/_02/full_pipeline/0_dataset"
os.makedirs(save_dir, exist_ok=True)

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

def smiles_to_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            Chem.SanitizeMol(mol)
        return mol
    except:
        return None

train_df['mol'] = train_df['Canonical_Smiles'].apply(smiles_to_mol)
test_df['mol'] = test_df['Canonical_Smiles'].apply(smiles_to_mol)

COVALENT_RADII = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
                  15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39}
ELECTRONEGATIVITY = {1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
                     15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66}

ptable = GetPeriodicTable()

def one_hot_encoding(value, choices):
    return [int(value == c) for c in choices]

def atom_features(atom):
    atomic_num_choices = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
    hybridization_choices = [
        rdchem.HybridizationType.SP,
        rdchem.HybridizationType.SP2,
        rdchem.HybridizationType.SP3,
        rdchem.HybridizationType.SP3D,
        rdchem.HybridizationType.SP3D2,
    ]
    atomic_num = atom.GetAtomicNum()
    electronegativity = ELECTRONEGATIVITY.get(atomic_num, 0.0)
    covalent_radius = COVALENT_RADII.get(atomic_num, 1.0)
    features = []
    features += one_hot_encoding(atomic_num, atomic_num_choices)
    features.append(atom.GetDegree())
    features.append(atom.GetFormalCharge())
    features.append(atom.GetTotalNumHs())
    features += one_hot_encoding(atom.GetHybridization(), hybridization_choices)
    features.append(int(atom.GetIsAromatic()))
    features.append(atom.GetImplicitValence())
    features.append(atom.GetTotalValence())
    features.append(atom.GetNumRadicalElectrons())
    features.append(int(atom.IsInRing()))
    features.append(covalent_radius)
    features.append(electronegativity)
    return torch.tensor(features, dtype=torch.float)

def bond_features(bond):
    stereo_choices = list(rdchem.BondStereo.values)
    stereo_encoded = one_hot_encoding(bond.GetStereo(), stereo_choices)
    return torch.tensor([
        bond.GetBondTypeAsDouble(),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
        *stereo_encoded
    ], dtype=torch.float)

def mol_to_dgl_graph(mol):
    num_atoms = mol.GetNumAtoms()
    atom_feats = [atom_features(mol.GetAtomWithIdx(i)) for i in range(num_atoms)]
    srcs, dsts, bond_feats = [], [], []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        srcs += [u, v]
        dsts += [v, u]
        b_feat = bond_features(bond)
        bond_feats += [b_feat, b_feat]
    g = dgl.graph((srcs, dsts), num_nodes=num_atoms)
    g = dgl.add_self_loop(g)
    num_edges = g.num_edges()
    num_bond_feats = len(bond_feats)
    if num_edges > num_bond_feats:
        dummy_bond = torch.zeros_like(bond_feats[0])
        for _ in range(num_edges - num_bond_feats):
            bond_feats.append(dummy_bond)
    g.ndata['h'] = torch.stack(atom_feats)
    g.edata['e'] = torch.stack(bond_feats)
    return g

train_graphs = [mol_to_dgl_graph(mol) for mol in train_df['mol'] if mol is not None]
test_graphs = [mol_to_dgl_graph(mol) for mol in test_df['mol'] if mol is not None]

joblib.dump(train_graphs, f"{save_dir}/train_graphs.pkl")
joblib.dump(test_graphs, f"{save_dir}/test_graphs.pkl")

train_graphs = joblib.load("./Drug/_02/full_pipeline/0_dataset/train_graphs.pkl")
test_graphs = joblib.load("./Drug/_02/full_pipeline/0_dataset/test_graphs.pkl")
train_df = pd.read_csv("./Drug/train.csv")
test_df = pd.read_csv("./Drug/test.csv")
y = train_df["Inhibition"].values

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

from dgl.nn import Set2Set, GATv2Conv
class GATv2Set2Set(nn.Module):
    def __init__(self, in_node_feats, hidden_size=128, num_heads=4, dropout=0.2):
        super().__init__()
        self.gat1 = GATv2Conv(in_node_feats, hidden_size, num_heads)
        self.gat2 = GATv2Conv(hidden_size * num_heads, hidden_size, 1)
        self.set2set = Set2Set(hidden_size, n_iters=6, n_layers=1)
        self.readout = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, g):
        h = g.ndata['h']
        h = self.gat1(g, h).flatten(1)
        h = F.elu(h)
        h = self.gat2(g, h).mean(1)
        g.ndata['h'] = h
        hg = self.set2set(g, h)
        return self.readout(hg).squeeze(-1)

def collate_graphs(batch):
    if isinstance(batch[0], tuple):
        graphs, labels = zip(*batch)
        return dgl.batch(graphs), torch.stack(labels)
    else:
        return dgl.batch(batch)
def evaluate(model, dataloader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, tuple):
                g, _ = batch
            else:
                g = batch
            g = g.to(device)
            pred = model(g)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds)

# ======================
# ✅ GNN 학습 + 예측
# ======================
device = torch.device("cpu")
n_splits = 5
oof = np.zeros(len(train_graphs))
test_preds_each_fold = []

kf = KFold(n_splits=n_splits, shuffle=True, random_state=r)
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

    in_feats = train_graphs[0].ndata['h'].shape[1]
    model = GATv2Set2Set(in_node_feats=in_feats).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best_rmse = float("inf")
    patience, patience_counter = 10, 0

    for epoch in range(1, 101):
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

        val_preds = evaluate(model, val_loader, device)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        print(f"Epoch {epoch:03d} | Val RMSE: {val_rmse:.4f}")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), f"./Drug/_02/full_pipeline/gat_fold{fold}.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹ Early stopping triggered")
                break

    model.load_state_dict(torch.load(f"./Drug/_02/full_pipeline/gat_fold{fold}.pt"))
    val_preds = evaluate(model, val_loader, device)
    test_pred = evaluate(model, test_loader, device)

    oof[val_idx] = val_preds
    test_preds_each_fold.append(test_pred)

# ======================
# ✅ 저장 및 출력
# ======================
test_preds = np.mean(test_preds_each_fold, axis=0)
np.save("./Drug/_02/full_pipeline/gat_oof.npy", oof)
np.save("./Drug/_02/full_pipeline/gat_preds.npy", test_preds)

submission = pd.DataFrame({
    "ID": test_df["ID"],
    "Inhibition": test_preds
})
now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
filename = f"GAT_{r}_({now}).csv"
submission.to_csv(f"./Drug/_02/full_pipeline/{filename}", index=False)
print(f"✅ 저장 완료 → {filename}")

# ======================
# ✅ 성능 출력
# ======================
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def print_scores(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    pearson = pearsonr(y_true, y_pred)[0]
    score = 0.5 * (1 - min(nrmse, 1)) + 0.5 * np.clip(pearson, 0, 1)
    print(f"📊 {label} 결과")
    print(f"RMSE     : {rmse:.5f}")
    print(f"NRMSE    : {nrmse:.5f}")
    print(f"Pearson  : {pearson:.5f}")
    print(f"Score📈  : {score:.5f}")
    print(f"Random   : {r}")
    return score

print_scores(y, oof, label="GAT 단독 성능")
