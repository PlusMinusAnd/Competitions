import dgl
import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem
import os


# 🔥 그래프 불러오기
graphs, label_dict = dgl.load_graphs('./Drug/_engineered_data/test_graph.bin')

# 🔥 SMILES 불러오기
import pandas as pd
train_csv = pd.read_csv('./Drug/test.csv')
smiles_list = train_csv['Canonical_Smiles'].tolist()


# ✅ 그래프와 SMILES 매핑
g = graphs[0]
smiles = smiles_list[0]

# ✅ RDKit Mol 객체 생성
mol = Chem.MolFromSmiles(smiles)

# ✅ 원자 이름 리스트 생성
atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
print("원자 리스트:", atom_symbols)

# ✅ DGL → NetworkX 변환
nx_g = g.to_networkx().to_undirected()

# ✅ 패딩 제거
connected_nodes = [n for n in nx_g.nodes if nx_g.degree[n] > 0]
nx_sub = nx_g.subgraph(connected_nodes)

# ✅ 노드 라벨: 패딩 제외 후 원자 기호로 매핑
node_labels = {}
for idx, node in enumerate(sorted(connected_nodes)):
    if idx < len(atom_symbols):
        node_labels[node] = atom_symbols[idx]
    else:
        node_labels[node] = "X"  # 혹시 부족하면 임시

# 엣지 두께 (결합 강도)
edge_weights = []
for u, v in nx_sub.edges():
    try:
        eid = g.edge_ids(u, v, return_uv=False)
        bond_strength = g.edata['h'][eid].item() if isinstance(eid, torch.Tensor) else g.edata['h'][eid]
    except:
        bond_strength = 1  # 연결 안 된 경우 대비 안전

    edge_weights.append(bond_strength * 2)  # 두께 조정

# ✅ 시각화 및 저장
plt.figure(figsize=(6, 6))
nx.draw(
    nx_sub,
    labels=node_labels,
    with_labels=True,
    node_color='skyblue',
    node_size=500,
    edge_color='gray',
    width=edge_weights
)

save_path = './Drug/_engineered_data/GNN_Graph/graph_0_atom_label.png'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300)
plt.close()

print(f"✔️ 저장 완료: {save_path}")
