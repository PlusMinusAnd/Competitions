import dgl
import torch
import numpy as np
import os

# 🔥 Numpy array → DGL 그래프로 변환 함수
def matrix_to_dgl_graph(matrix):
    """
    matrix: (N, N, 2) 형태
    matrix[..., 0] → 결합 여부 (adjacency)
    matrix[..., 1] → 결합 강도 (bond strength)
    """
    N = matrix.shape[0]
    adjacency = matrix[..., 0]
    bond_strength = matrix[..., 1]

    src, dst = np.where(adjacency == 1)

    # DGL 그래프 생성
    g = dgl.graph((src, dst), num_nodes=N)

    # Node feature: node 수 만큼 dummy feature (예: degree)
    degrees = adjacency.sum(axis=1, keepdims=True)
    g.ndata['h'] = torch.tensor(degrees, dtype=torch.float32)

    # Edge feature: bond strength
    edge_features = []
    for s, d in zip(src, dst):
        edge_features.append([bond_strength[s, d]])
    g.edata['h'] = torch.tensor(edge_features, dtype=torch.float32)

    return g


# 🔥 데이터 로드
train = np.load('./Drug/_npy_data/train_graph.npy')
test = np.load('./Drug/_npy_data/test_graph.npy')

# 🔥 전체 train/test 변환
train_graphs = [matrix_to_dgl_graph(m) for m in train]
test_graphs = [matrix_to_dgl_graph(m) for m in test]

# 🔥 저장
dgl.save_graphs('./Drug/_engineered_data/train_graph.bin', train_graphs)
dgl.save_graphs('./Drug/_engineered_data/test_graph.bin', test_graphs)

print("✔️ 저장 완료: ./Drug/_engineered_data/")


