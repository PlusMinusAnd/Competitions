# ===================================== #
#         📦 라이브러리 불러오기        #
# ===================================== #
import numpy as np
import pandas as pd
import random
import warnings
import optuna

from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

# Seed 고정
r = random.randint(1,1000)
random.seed(r)
np.random.seed(r)

# 결과 함수
def print_scores(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pr = pearsonr(y_true, y_pred)[0]
    score = 0.5 * (1 - min(rmse / (np.max(y_true) - np.min(y_true)), 1)) + 0.5 * np.clip(pr, 0, 1)
    print(f"\n📊 {label} 결과")
    print(f"RMSE     : {rmse:.5f}")
    print(f"R2       : {r2:.5f}")
    print(f"Pearson  : {pr:.5f}")
    print(f"SCORE    : {score:.5f}")

data_path = './Drug/_03/1_data/'

numeric_train = pd.read_csv(data_path + 'train_numeric.csv')
numeric_test = pd.read_csv(data_path + 'test_numeric.csv')
categorical_train = pd.read_csv(data_path + 'train_category.csv')
categorical_test = pd.read_csv(data_path + 'test_category.csv')

x1 = numeric_train.drop(['Canonical_Smiles', 'Inhibition'], axis=1).values
y1 = numeric_train['Inhibition'].values
test1 = numeric_test.drop(['Canonical_Smiles'], axis=1).values

# print('x1.shape',x1.shape)            x1.shape (1681, 36)
# print('y1.shape',y1.shape)            y1.shape (1681,)
# print('test1.shape',test1.shape)      test1.shape (100, 36)
# print('x2.shape',x2.shape)            x2.shape (1681, 98)
# print('y2.shape',y2.shape)            y2.shape (1681,)
# print('test2.shape',test2.shape)      test2.shape (100, 98)

nu_x1_train, nu_x1_test, nu_y1_train, nu_y1_test = train_test_split(
    x1, y1, random_state=r, train_size=0.8
)

# ✅ Numeric Optuna 목적 함수들
def nu_objective_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": r
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=r)
    nu_rmse_list = []
    for train_idx, val_idx in kf.split(nu_x1_train):
        x1_train, x1_val = nu_x1_train[train_idx], nu_x1_train[val_idx]
        y1_train, y1_val = nu_y1_train[train_idx], nu_y1_train[val_idx]
        model = XGBRegressor(**params, early_stopping_rounds=30)
        model.fit(x1_train, y1_train,
                  eval_set=[(x1_val, y1_val)], verbose=0)
        preds = model.predict(x1_val)
        nu_rmse = np.sqrt(mean_squared_error(y1_val, preds))
        nu_rmse_list.append(nu_rmse)
    return np.mean(nu_rmse_list)

def nu_objective_lgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "verbosity": -1,
        "random_state": r
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=r)
    nu_rmse_list = []
    for train_idx, val_idx in kf.split(nu_x1_train):
        x1_train, x1_val = nu_x1_train[train_idx], nu_x1_train[val_idx]
        y1_train, y1_val = nu_y1_train[train_idx], nu_y1_train[val_idx]
        model = LGBMRegressor(**params, early_stopping_rounds=30, verbose=-1)
        model.fit(x1_train, y1_train,
                  eval_set=[(x1_val, y1_val)],
                  )
        preds = model.predict(x1_val)
        rmse = np.sqrt(mean_squared_error(y1_val, preds))
        nu_rmse_list.append(rmse)
    return np.mean(nu_rmse_list)

def nu_objective_cat(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-4, 10.0, log=True),
        "random_state": r,
        "verbose": 0
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=r)
    nu_rmse_list = []
    for train_idx, val_idx in kf.split(nu_x1_train):
        x1_train, x1_val = nu_x1_train[train_idx], nu_x1_train[val_idx]
        y1_train, y1_val = nu_y1_train[train_idx], nu_y1_train[val_idx]
        model = CatBoostRegressor(**params)
        model.fit(x1_train, y1_train,
                  eval_set=(x1_val, y1_val),
                  early_stopping_rounds=30)
        preds = model.predict(x1_val)
        rmse = np.sqrt(mean_squared_error(y1_val, preds))
        nu_rmse_list.append(rmse)
    return np.mean(nu_rmse_list)

# 🎯 튜닝 실행
print("🔧 Numeric_Tuning XGBoost...")
nu_study_xgb = optuna.create_study(direction="minimize")
nu_study_xgb.optimize(nu_objective_xgb, n_trials=1)

print("🔧 Numeric_Tuning LightGBM...")
nu_study_lgb = optuna.create_study(direction="minimize")
nu_study_lgb.optimize(nu_objective_lgb, n_trials=1)

print("🔧 Numeric_Tuning CatBoost...")
nu_study_cat = optuna.create_study(direction="minimize")
nu_study_cat.optimize(nu_objective_cat, n_trials=1)

# ✅ 최적 모델 학습
nu_best_xgb = XGBRegressor(**nu_study_xgb.best_params)
nu_best_lgb = LGBMRegressor(**nu_study_lgb.best_params)
nu_best_cat = CatBoostRegressor(**nu_study_cat.best_params)

# ====================== NUMERIC ==========================

nu_kf = KFold(n_splits=5, shuffle=True, random_state=r)
nu_oof_preds = []
nu_oof_true = []

for train_idx, val_idx in nu_kf.split(nu_x1_train):
    x1_train, x1_val = nu_x1_train[train_idx], nu_x1_train[val_idx]
    y1_train, y1_val = nu_y1_train[train_idx], nu_y1_train[val_idx]

    nu_best_xgb.fit(x1_train, y1_train)
    nu_best_lgb.fit(x1_train, y1_train)
    nu_best_cat.fit(x1_train, y1_train)

    nu_pred_xgb = nu_best_xgb.predict(x1_val)
    nu_pred_lgb = nu_best_lgb.predict(x1_val)
    nu_pred_cat = nu_best_cat.predict(x1_val)

    nu_stacked_input = np.vstack([nu_pred_xgb, nu_pred_lgb, nu_pred_cat]).T
    nu_ridge = RidgeCV()
    nu_ridge.fit(nu_stacked_input, y1_val)

    nu_final_pred = nu_ridge.predict(nu_stacked_input)
    nu_oof_preds.append(nu_final_pred)
    nu_oof_true.append(y1_val)

# 📊 최종 결과 출력
nu_oof_preds = np.concatenate(nu_oof_preds)
nu_oof_true = np.concatenate(nu_oof_true)

print_scores(nu_oof_true, nu_oof_preds)

# ====================== CATEGORICAL ==========================

x2 = categorical_train.drop(['Canonical_Smiles', 'Inhibition'], axis=1).values
y2 = categorical_train['Inhibition'].values
test2 = categorical_test.drop(['Canonical_Smiles'], axis=1).values

ca_x2_train, ca_x2_test, ca_y2_train, ca_y2_test = train_test_split(
    x2, y2, random_state=r, train_size=0.8
)

# ✅ Categorical Optuna 목적 함수들
def ca_objective_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": r
    }

    ca_kf = KFold(n_splits=5, shuffle=True, random_state=r)
    ca_rmse_list = []
    for train_idx, val_idx in ca_kf.split(ca_x2_train):
        x2_train, x2_val = ca_x2_train[train_idx], ca_x2_train[val_idx]
        y2_train, y2_val = ca_y2_train[train_idx], ca_y2_train[val_idx]
        model = XGBRegressor(**params, early_stopping_rounds=30)
        model.fit(x2_train, y2_train,
                  eval_set=[(x2_val, y2_val)], verbose=0)
        preds = model.predict(ca_x2_test)
        rmse = np.sqrt(mean_squared_error(ca_y2_test, preds))
        ca_rmse_list.append(rmse)
    return np.mean(ca_rmse_list)

def ca_objective_lgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "verbosity": -1,
        "random_state": r
    }

    ca_kf = KFold(n_splits=5, shuffle=True, random_state=r)
    ca_rmse_list = []
    for train_idx, val_idx in ca_kf.split(ca_x2_train):
        x2_train, x2_val = ca_x2_train[train_idx], ca_x2_train[val_idx]
        y2_train, y2_val = ca_y2_train[train_idx], ca_y2_train[val_idx]
        model = LGBMRegressor(**params, early_stopping_rounds=30, verbose=-1)
        model.fit(x2_train, y2_train,
                  eval_set=[(x2_val, y2_val)],
                  )
        preds = model.predict(ca_x2_test)
        ca_rmse = np.sqrt(mean_squared_error(ca_y2_test, preds))
        ca_rmse_list.append(ca_rmse)
    return np.mean(ca_rmse_list)

def ca_objective_cat(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-4, 10.0, log=True),
        "random_state": r,
        "verbose": 0
    }

    ca_kf = KFold(n_splits=5, shuffle=True, random_state=r)
    ca_rmse_list = []
    for train_idx, val_idx in ca_kf.split(ca_x2_train):
        x2_train, x2_val = ca_x2_train[train_idx], ca_x2_train[val_idx]
        y2_train, y2_val = ca_y2_train[train_idx], ca_y2_train[val_idx]
        model = CatBoostRegressor(**params)
        model.fit(x2_train, y2_train,
                  eval_set=(x2_val, y2_val),
                  early_stopping_rounds=30)
        preds = model.predict(ca_x2_test)
        ca_rmse = np.sqrt(mean_squared_error(ca_y2_test, preds))
        ca_rmse_list.append(ca_rmse)
    return np.mean(ca_rmse_list)

# 🎯 튜닝 실행
print("🔧 Category_Tuning XGBoost...")
ca_study_xgb = optuna.create_study(direction="minimize")
ca_study_xgb.optimize(ca_objective_xgb, n_trials=1)

print("🔧 Category_Tuning LightGBM...")
ca_study_lgb = optuna.create_study(direction="minimize")
ca_study_lgb.optimize(ca_objective_lgb, n_trials=1)

print("🔧 Category_Tuning CatBoost...")
ca_study_cat = optuna.create_study(direction="minimize")
ca_study_cat.optimize(ca_objective_cat, n_trials=1)

# ✅ 최적 모델 학습
ca_best_xgb = XGBRegressor(**ca_study_xgb.best_params)
ca_best_lgb = LGBMRegressor(**ca_study_lgb.best_params)
ca_best_cat = CatBoostRegressor(**ca_study_cat.best_params)

ca_kf = KFold(n_splits=5, shuffle=True, random_state=r)
ca_oof_preds = []
ca_oof_true = []

for train_idx, val_idx in ca_kf.split(ca_x2_train):
    x2_train, x2_val = ca_x2_train[train_idx], ca_x2_train[val_idx]
    y2_train, y2_val = ca_y2_train[train_idx], ca_y2_train[val_idx]

    ca_best_xgb.fit(x2_train, y2_train)
    ca_best_lgb.fit(x2_train, y2_train)
    ca_best_cat.fit(x2_train, y2_train)

    ca_pred_xgb = ca_best_xgb.predict(x2_val)
    ca_pred_lgb = ca_best_lgb.predict(x2_val)
    ca_pred_cat = ca_best_cat.predict(x2_val)

    ca_stacked_input = np.vstack([ca_pred_xgb, ca_pred_lgb, ca_pred_cat]).T
    ca_ridge = RidgeCV()
    ca_ridge.fit(ca_stacked_input, y2_val)

    ca_final_pred = ca_ridge.predict(ca_stacked_input)
    ca_oof_preds.append(ca_final_pred)
    ca_oof_true.append(y2_val)

# 📊 최종 결과 출력
ca_oof_preds = np.concatenate(ca_oof_preds)
ca_oof_true = np.concatenate(ca_oof_true)

print_scores(ca_oof_true, ca_oof_preds)

# 1️⃣ 최종 OOF 기반 앙상블 weight 계산 (RMSE 기준 또는 직접 weight 부여)
# 여기서는 동등 가중치 사용, 필요 시 성능 기반 조정 가능
final_preds_blend = 0.5 * nu_oof_preds + 0.5 * ca_oof_preds

# 2️⃣ 최종 성능 확인
print_scores(nu_oof_true, final_preds_blend, "Final Ensemble")

# 3️⃣ Test 데이터 예측
# Numeric 테스트 예측 (각 최적 모델 기반)
nu_pred_xgb_test = nu_best_xgb.predict(test1)
nu_pred_lgb_test = nu_best_lgb.predict(test1)
nu_pred_cat_test = nu_best_cat.predict(test1)

nu_test_stacked = np.vstack([nu_pred_xgb_test, nu_pred_lgb_test, nu_pred_cat_test]).T
nu_test_final = nu_ridge.predict(nu_test_stacked)

# Categorical 테스트 예측
ca_pred_xgb_test = ca_best_xgb.predict(test2)
ca_pred_lgb_test = ca_best_lgb.predict(test2)
ca_pred_cat_test = ca_best_cat.predict(test2)

ca_test_stacked = np.vstack([ca_pred_xgb_test, ca_pred_lgb_test, ca_pred_cat_test]).T
ca_test_final = ca_ridge.predict(ca_test_stacked)

# 4️⃣ 최종 앙상블
final_test_pred = 0.5 * nu_test_final + 0.5 * ca_test_final

# 5️⃣ 제출 파일 생성
submit = pd.DataFrame({
    "Id": np.arange(len(final_test_pred)),
    "Inhibition": final_test_pred
})

submit_path = "./Drug/_03/final_submission.csv"
submit.to_csv(submit_path, index=False)
print(f"📁 최종 제출 파일 저장 완료: {submit_path}")