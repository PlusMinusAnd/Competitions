# 05_meta_model/train_meta_model.py

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.optimize import minimize
import sys
import os

# utils 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_utils')))

# 함수 import
from set_seed import set_seed

# 시드 고정
seed = set_seed()
print('고정된 SEED :', seed)


# 사용자 정의 점수 함수
def calc_score(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    pearson = pearsonr(y_true, y_pred)[0]
    score = 0.5 * (1 - min(nrmse, 1)) + 0.5 * np.clip(pearson, 0, 1)
    return score, rmse, nrmse, pearson

# === 데이터 불러오기 ===
base_path = './Drug/_05'
submission_dir = './Drug/_05/mk_data'
os.makedirs(submission_dir, exist_ok=True)

# oof 예측 (train)
oof1 = pd.read_csv(f'{base_path}/mk_data/descriptor_oof.csv')
oof2 = pd.read_csv(f'{base_path}/mk_data/fingerprint_oof.csv')
oof3 = pd.read_csv(f'{base_path}/mk_data/gnn_oof.csv')
oof4 = pd.read_csv(f'{base_path}/mk_data/rnn_oof.csv')  # or rnn_oof.csv

train_df = pd.read_csv(f'{base_path}/data/train.csv')
y_true = train_df['Inhibition'].values
train_ids = train_df['ID'].values

X = pd.merge(oof1, oof2, on='ID')
X = pd.merge(X, oof3, on='ID')
X = pd.merge(X, oof4, on='ID')
X = X.set_index('ID')
X = X.values

# test 예측
pred1 = pd.read_csv(f'{base_path}/mk_data/descriptor_preds.csv')
pred2 = pd.read_csv(f'{base_path}/mk_data/fingerprint_preds.csv')
pred3 = pd.read_csv(f'{base_path}/mk_data/gnn_preds.csv')
pred4 = pd.read_csv(f'{base_path}/mk_data/rnn_preds.csv')

test_df = pd.read_csv(f'{base_path}/data/test.csv')
test_ids = test_df['ID'].values

X_test = pd.merge(pred1, pred2, on='ID')
X_test = pd.merge(X_test, pred3, on='ID')
X_test = pd.merge(X_test, pred4, on='ID')
X_test = X_test.set_index('ID')
X_test = X_test.values

# ========================
# ✅ 1. RidgeCV 기반 앙상블
# ========================
ridge = RidgeCV()
ridge.fit(X, y_true)
ridge_oof = ridge.predict(X)
ridge_test = ridge.predict(X_test)

ridge_score, rmse, nrmse, pearson = calc_score(y_true, ridge_oof)
print("# 📊 RidgeCV 결과")
print(f"# RMSE     : {rmse:.5f}")
print(f"# NRMSE    : {nrmse:.5f}")
print(f"# Pearson  : {pearson:.5f}")
print(f"# Score    : {ridge_score:.5f}")
print(f"# Seed     : {seed}")
        
import datetime
now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
filename = f"submission_seed{seed}_({now}).csv"
submission_ridge = pd.DataFrame({'ID': test_ids, 'Inhibition': ridge_test})
submission_ridge.to_csv(f'{submission_dir}/{filename}', index=False)
print(f"# 파일명: {filename}")

# 🎯 기준 점수 설정
SCORE_THRESHOLD = 0.61349 
import shutil
# 점수가 기준보다 낮으면 전체 디렉토리 삭제
if ridge_score < SCORE_THRESHOLD:
    shutil.rmtree("./Drug/_05/mk_data")
    print(f"🚫 Score {ridge_score:.5f} < 기준 {SCORE_THRESHOLD} → 전체 디렉토리 삭제 완료")
else:
    print(f"🎉 Score {ridge_score:.5f} ≥ 기준 {SCORE_THRESHOLD} → 디렉토리 유지")

# 📊 RidgeCV 결과
# RMSE     : 23.40505
# NRMSE    : 0.23551
# Pearson  : 0.46249
# Score    : 0.61349
# Seed     : 100
# 파일명: submission_seed100_(20250709_2330).csv