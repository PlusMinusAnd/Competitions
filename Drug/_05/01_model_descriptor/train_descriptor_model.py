# 01_model_descriptor/train_descriptor_model.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import joblib
import sys

# utils 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_utils')))
from rdkit_features import get_rdkit_descriptors
from set_seed import set_seed

# 시드 고정
seed = set_seed()
print('고정된 SEED :', seed)

# 경로
train_path = './Drug/_05/data/train.csv'
test_path  = './Drug/_05/data/test.csv'
save_dir   = './Drug/_05/mk_data'
os.makedirs(save_dir, exist_ok=True)

# 데이터 로딩
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

x_smiles = train_df['Canonical_Smiles'].tolist()
test_smiles = test_df['Canonical_Smiles'].tolist()
y = train_df['Inhibition'].values
train_ids = train_df['ID']
test_ids = test_df['ID']

# RDKit Descriptor 피처 생성
x_feat = get_rdkit_descriptors(x_smiles)
test_feat = get_rdkit_descriptors(test_smiles)

# 결측치 제거 및 스케일링
x_feat = x_feat.dropna(axis=1)
test_feat = test_feat[x_feat.columns]
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_feat)
test_scaled = scaler.transform(test_feat)

# for i in range(1, 1000) :


# 모델 정의
base_models = [
    ('xgb', XGBRegressor(n_estimators=200, random_state=seed)),
    ('lgb', LGBMRegressor(n_estimators=200, random_state=seed)),
    ('cat', CatBoostRegressor(verbose=0, random_state=seed))
]
meta_model = RidgeCV()
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model, n_jobs=-1)

# KFold 학습
kf = KFold(n_splits=5, shuffle=True, random_state=seed)
oof_preds = np.zeros(len(train_df))
test_preds = []

for train_idx, val_idx in kf.split(x_scaled):
    x_train, x_val = x_scaled[train_idx], x_scaled[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    stacking_model.fit(x_train, y_train)
    oof_preds[val_idx] = stacking_model.predict(x_val)
    test_preds.append(stacking_model.predict(test_scaled))

# test 평균 예측
final_test_preds = np.mean(test_preds, axis=0)

# === 저장 ===

# OOF 저장
oof_df = pd.DataFrame({
    'ID': train_ids,
    'Descriptor_OOF': oof_preds
})
oof_df.to_csv(f'{save_dir}/descriptor_oof.csv', index=False)

# Test 저장
test_df = pd.DataFrame({
    'ID': test_ids,
    'Descriptor_Pred': final_test_preds
})
test_df.to_csv(f'{save_dir}/descriptor_preds.csv', index=False)

print(f"✅ 저장 완료:")
print(f" - OOF : {save_dir}/descriptor_oof.csv")
print(f" - TEST: {save_dir}/descriptor_preds.csv")
