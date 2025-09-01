import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# 🔹 데이터 로드
train = pd.read_csv('./Drug/_engineered_data_DNN/train_final.csv')  # 경로에 따라 수정
X = train.drop(columns=['ID', 'Canonical_Smiles', 'Inhibition'])
y = train['Inhibition']

# 🔹 NRMSE 정의
def nrmse_score(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return rmse / (np.max(y_true) - np.min(y_true))

# 🔹 CV 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 🔹 CatBoost 최적화 함수
def objective_cat(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 300, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 0.1, 2.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'verbose': 0,
        'random_state': 42
    }
    scores = []
    for train_idx, valid_idx in kf.split(X):
        model = CatBoostRegressor(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[valid_idx])
        scores.append(nrmse_score(y.iloc[valid_idx], preds))
    return np.mean(scores)

# 🔹 LightGBM 최적화 함수
def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'random_state': 42
    }
    scores = []
    for train_idx, valid_idx in kf.split(X):
        model = LGBMRegressor(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[valid_idx])
        scores.append(nrmse_score(y.iloc[valid_idx], preds))
    return np.mean(scores)

# 🔹 XGBoost 최적화 함수
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'random_state': 42
    }
    scores = []
    for train_idx, valid_idx in kf.split(X):
        model = XGBRegressor(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[valid_idx])
        scores.append(nrmse_score(y.iloc[valid_idx], preds))
    return np.mean(scores)

# 🔹 최적화 실행 (각 30회, 필요시 증가 가능)
cat_study = optuna.create_study(direction='minimize')
cat_study.optimize(objective_cat, n_trials=30)

lgb_study = optuna.create_study(direction='minimize')
lgb_study.optimize(objective_lgb, n_trials=30)

xgb_study = optuna.create_study(direction='minimize')
xgb_study.optimize(objective_xgb, n_trials=30)

# 🔹 최적 하이퍼파라미터 출력
print("Best CatBoost params:", cat_study.best_params)
print("Best LGBM params:", lgb_study.best_params)
print("Best XGB params:", xgb_study.best_params)


# Best CatBoost params: {'iterations': 895, 'depth': 8, 'learning_rate': 0.014234757332134227, 'l2_leaf_reg': 5.009890356480435, 'random_strength': 1.7136961646909046, 'bagging_temperature': 0.36935391585542093, 'border_count': 32}
# Best LGBM params: {'n_estimators': 567, 'max_depth': 4, 'learning_rate': 0.006563689357466647, 'num_leaves': 20, 'reg_alpha': 0.024874040978053735, 'reg_lambda': 0.8696915988225338}
# Best XGB params: {'n_estimators': 306, 'max_depth': 5, 'learning_rate': 0.013325160842544384, 'subsample': 0.7900456721436573, 'colsample_bytree': 0.691869775561059, 'reg_alpha': 0.9152904593451836, 'reg_lambda': 0.8530393538422426}