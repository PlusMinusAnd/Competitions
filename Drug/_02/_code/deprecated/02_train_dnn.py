##### 02_train_dnn.py #####
"""
수치형 + fingerprint 데이터를 활용한 DNN 학습 (2종 모델)
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# 고정 시드
r = 394
np.random.seed(r)
torch.manual_seed(r)

# 데이터 로드
X = np.load("./Drug/_02/full_pipeline/0_dataset/X_train_full.npy")
X_test = np.load("./Drug/_02/full_pipeline/0_dataset/X_test_full.npy")
y = np.load("./Drug/_02/full_pipeline/0_dataset/y_train.npy")

# 저장 디렉토리 준비
save_dir = "./Drug/_02/full_pipeline"
os.makedirs(f"{save_dir}/1_model", exist_ok=True)
os.makedirs(f"{save_dir}/2_oof", exist_ok=True)
os.makedirs(f"{save_dir}/3_preds", exist_ok=True)

### 점수 계산 함수 ###
def compute_score(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    pearson = pearsonr(y_true, y_pred)[0]
    clipped_pearson = np.clip(pearson, 0, 1)
    score = 0.5 * (1 - min(nrmse, 1)) + 0.5 * clipped_pearson
    return rmse, nrmse, pearson, score

### DNN 모델 정의 (Residual 포함) ###
class ResidualDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

### 학습 함수 ###
def train_model(model, X, y, model_name="dnn"):
    kf = KFold(n_splits=5, shuffle=True, random_state=r)
    device = torch.device("cpu")
    oof = np.zeros(len(X))
    preds = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model_ = model(input_dim=X.shape[1]).to(device)
        optimizer = torch.optim.Adam(model_.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)

        best_rmse = float('inf')
        best_model_path = f"{save_dir}/1_model/{model_name}_fold{fold}.pt"

        for epoch in range(1, 101):
            model_.train()
            optimizer.zero_grad()
            pred = model_(X_tr_t)
            loss = criterion(pred, y_tr_t)
            loss.backward()
            optimizer.step()

            model_.eval()
            with torch.no_grad():
                val_pred = model_(X_val_t)
                val_rmse = np.sqrt(mean_squared_error(y_val, val_pred.numpy()))
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                torch.save(model_.state_dict(), best_model_path)

        # OOF 예측
        model_.load_state_dict(torch.load(best_model_path))
        model_.eval()
        oof[val_idx] = model_(X_val_t).detach().numpy()
        preds.append(model_(X_test_t).detach().numpy())

    test_preds = np.mean(preds, axis=0)

    # 점수 출력
    rmse, nrmse, pearson, score = compute_score(y, oof)
    print(f"📊 {model_name.upper()} OOF Score")
    print(f"RMSE     : {rmse:.5f}")
    print(f"NRMSE    : {nrmse:.5f}")
    print(f"Pearson  : {pearson:.5f}")
    print(f"Score📈  : {score:.5f}")

    np.save(f"{save_dir}/2_oof/{model_name}_oof.npy", oof)
    np.save(f"{save_dir}/3_preds/{model_name}_preds.npy", test_preds)

    submission = pd.read_csv("./Drug/test.csv")
    submission["Inhibition"] = test_preds
    submission.to_csv(f"{save_dir}/submission_{model_name}_{r}.csv", index=False)
    print(f"✅ {model_name} 결과 저장 완료!")

### 실행 ###
if __name__ == "__main__":
    train_model(ResidualDNN, X, y, model_name="dnn_residual")
