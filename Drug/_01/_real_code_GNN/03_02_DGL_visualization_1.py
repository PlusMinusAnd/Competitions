
import dgl
import networkx as nx
import matplotlib.pyplot as plt
import os

# 🔥 그래프 불러오기
graphs, label_dict = dgl.load_graphs('./Drug/_engineered_data/test_graph.bin')

print(f"그래프 개수: {len(graphs)}")

# 🔥 첫 번째 그래프 선택
g = graphs[0]

# ✅ DGL → NetworkX 변환
nx_g = g.to_networkx().to_undirected()

'''
####### 패딩 포함 ########
# ✅ 시각화 및 저장
plt.figure(figsize=(6, 6))
nx.draw(
    nx_g,
    with_labels=True,
    node_color='skyblue',
    node_size=500,
    edge_color='gray'
)

# ✅ 저장 경로 설정
save_path = './Drug/_engineered_data/graph_1.png'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300)
plt.close()

print(f"✔️ 그래프 시각화 저장 완료: {save_path}")
'''

'''
####### 패딩 노드 숨기기 ########

# 연결된 노드만 시각화
connected_nodes = [n for n in nx_g.nodes if nx_g.degree[n] > 0]
nx_sub = nx_g.subgraph(connected_nodes)

plt.figure(figsize=(5, 5))
nx.draw(nx_sub, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray')
plt.savefig('./Drug/_engineered_data/GNN_Graph/graph_filtered.png', dpi=300)
plt.close()
'''
'''
####### 패딩 노드는 회색으로 표시하기 ########

node_colors = [
    'skyblue' if nx_g.degree[n] > 0 else 'lightgray'
    for n in nx_g.nodes
]

plt.figure(figsize=(5, 5))
nx.draw(
    nx_g,
    with_labels=True,
    node_color=node_colors,
    node_size=500,
    edge_color='gray'
)
plt.savefig('./Drug/_engineered_data/GNN_Graph/graph_with_padding.png', dpi=300)
plt.close()'''



