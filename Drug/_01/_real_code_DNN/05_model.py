import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from scipy.stats import pearsonr

# 데이터 로드
train = pd.read_csv('./Drug/_engineered_data_DNN/train_final.csv')
test = pd.read_csv('./Drug/_engineered_data_DNN/test_final.csv')

# 피처/타겟 분리
X = train.drop(columns=['ID', 'Canonical_Smiles', 'Inhibition'])
y = train['Inhibition']
X_test = test.drop(columns=['ID', 'Canonical_Smiles'])

# KFold 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)

Cat_params= {'iterations': 895, 'depth': 8, 'learning_rate': 0.014234757332134227, 'l2_leaf_reg': 5.009890356480435, 
             'random_strength': 1.7136961646909046, 'bagging_temperature': 0.36935391585542093, 'border_count': 32}
LGBM_params= {'n_estimators': 567, 'max_depth': 4, 'learning_rate': 0.006563689357466647, 
              'num_leaves': 20, 'reg_alpha': 0.024874040978053735, 'reg_lambda': 0.8696915988225338}
XGB_params= {'n_estimators': 306, 'max_depth': 5, 'learning_rate': 0.013325160842544384, 
             'subsample': 0.7900456721436573, 'colsample_bytree': 0.691869775561059, 'reg_alpha': 0.9152904593451836, 'reg_lambda': 0.8530393538422426}

# 모델 정의
cat_model = CatBoostRegressor(verbose=0, random_state=42,
                              **Cat_params)
lgb_model = LGBMRegressor(random_state=42, **LGBM_params)
xgb_model = XGBRegressor(random_state=42, **XGB_params)

# OOF 예측값 저장
oof_preds_cat = np.zeros(len(X))
oof_preds_lgb = np.zeros(len(X))
oof_preds_xgb = np.zeros(len(X))
test_preds_cat = np.zeros(len(X_test))
test_preds_lgb = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))

# KFold 루프
for train_idx, valid_idx in kf.split(X):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    cat_model.fit(X_train, y_train)
    lgb_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    oof_preds_cat[valid_idx] = cat_model.predict(X_valid)
    oof_preds_lgb[valid_idx] = lgb_model.predict(X_valid)
    oof_preds_xgb[valid_idx] = xgb_model.predict(X_valid)

    test_preds_cat += cat_model.predict(X_test) / kf.n_splits
    test_preds_lgb += lgb_model.predict(X_test) / kf.n_splits
    test_preds_xgb += xgb_model.predict(X_test) / kf.n_splits

# 스태킹: OOF → 최종 메타 모델 (RidgeCV 사용)
stack_train = np.vstack([oof_preds_cat, oof_preds_lgb, oof_preds_xgb]).T
stack_test = np.vstack([test_preds_cat, test_preds_lgb, test_preds_xgb]).T

stack_model = RidgeCV()
stack_model.fit(stack_train, y)
final_oof = stack_model.predict(stack_train)
final_preds = stack_model.predict(stack_test)

# 평가 지표 (대회 기준)
mse = mean_squared_error(y, final_oof)
rmse = np.sqrt(mse)
nrmse = rmse / (y.max() - y.min())
pearson = np.clip(pearsonr(y, final_oof)[0], 0, 1)
score = 0.5 * (1 - min(nrmse, 1)) + 0.5 * pearson

# 결과 저장
submission = test[['ID']].copy()
submission['Inhibition'] = final_preds
submission.to_csv('./Drug/_engineered_data_DNN/submission/stacked_submission.csv', index=False)

print('NRMSE :',nrmse) 
print('  PRS :',pearson)
print('SCORE :',score)
