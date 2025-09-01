# ===================================== #
#         📦 라이브러리 불러오기        #
# ===================================== #
import numpy as np
import pandas as pd
import random
import warnings
import optuna
import os

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

r = random.randint(1,1000)
random.seed(r)
np.random.seed(r)

# ===================================== #
#           🔄 1. 평가 함수             #
# ===================================== #
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
    print(f"Random   : {r}")

# ===================================== #
#         📂 2. 데이터 로드          #
# ===================================== #
data_path = './Drug/_03/1_data/'
numeric_train = pd.read_csv(data_path + 'train_numeric.csv')
numeric_test = pd.read_csv(data_path + 'test_numeric.csv')
categorical_train = pd.read_csv(data_path + 'train_category.csv')
categorical_test = pd.read_csv(data_path + 'test_category.csv')

x1_full = numeric_train.drop(['Canonical_Smiles', 'Inhibition'], axis=1).values
y1_full = numeric_train['Inhibition'].values
test1 = numeric_test.drop(['Canonical_Smiles'], axis=1).values

x2_full = categorical_train.drop(['Canonical_Smiles', 'Inhibition'], axis=1).values
y2_full = categorical_train['Inhibition'].values
test2 = categorical_test.drop(['Canonical_Smiles'], axis=1).values

# ===================================== #
#   🔄 3. 두 번째 split: train/valid/test   #
# ===================================== #
x1_train, x1_valid, y1_train, y1_valid = train_test_split(x1_full, y1_full, test_size=0.2, random_state=r)
x2_train, x2_valid, y2_train, y2_valid = train_test_split(x2_full, y2_full, test_size=0.2, random_state=r)

# ===================================== #
#    🏋️ 4. Optuna 모델 통합 함수     #
# ===================================== #
def optuna_train(model_class, suggest_func, x, y, label=""):
    def objective(trial):
        params = suggest_func(trial)
        kf = KFold(n_splits=5, shuffle=True, random_state=r)
        rmse_list = []
        for train_idx, val_idx in kf.split(x):
            x_t, x_v = x[train_idx], x[val_idx]
            y_t, y_v = y[train_idx], y[val_idx]
            if model_class == XGBRegressor:
                model = model_class(**params, early_stopping_rounds=30)
                model.fit(x_t, y_t, eval_set=[(x_v, y_v)], verbose=0)
            elif model_class == LGBMRegressor:
                model = model_class(**params, verbose=-1, early_stopping_rounds=30)
                model.fit(x_t, y_t, eval_set=[(x_v, y_v)])
            else:
                model = model_class(**params)
                model.fit(x_t, y_t, eval_set=[(x_v, y_v)], verbose=0, early_stopping_rounds=30)
            pred = model.predict(x_v)
            rmse_list.append(np.sqrt(mean_squared_error(y_v, pred)))
        return np.mean(rmse_list)

    print(f"\n🔧 {label} - {model_class.__name__} tuning...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1)
    return model_class(**study.best_params)

def suggest_xgb(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": r
    }

def suggest_lgb(trial):
    return {
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

def suggest_cat(trial):
    return {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-4, 10.0, log=True),
        "random_state": r,
        "verbose": 0
    }

# ===================================== #
#        🔧 튜닝 및 OOF 예측 생성        #
# ===================================== #
def train_and_predict(x_train, y_train, x_valid, y_valid, x_test, label):
    model_xgb = optuna_train(XGBRegressor, suggest_xgb, x_train, y_train, label)
    model_lgb = optuna_train(LGBMRegressor, suggest_lgb, x_train, y_train, label)
    model_cat = optuna_train(CatBoostRegressor, suggest_cat, x_train, y_train, label)

    model_xgb.fit(x_train, y_train)
    model_lgb.fit(x_train, y_train)
    model_cat.fit(x_train, y_train)

    valid_preds = np.vstack([
        model_xgb.predict(x_valid),
        model_lgb.predict(x_valid),
        model_cat.predict(x_valid)
    ]).T
    test_preds = np.vstack([
        model_xgb.predict(x_test),
        model_lgb.predict(x_test),
        model_cat.predict(x_test)
    ]).T

    ridge = RidgeCV()
    ridge.fit(valid_preds, y_valid)

    final_valid_pred = ridge.predict(valid_preds)
    final_test_pred = ridge.predict(test_preds)

    print_scores(y_valid, final_valid_pred, f"{label} Ridge Ensemble")
    return final_valid_pred, final_test_pred

# 실행
nu_oof, nu_test_final = train_and_predict(x1_train, y1_train, x1_valid, y1_valid, test1, label="Numeric")
ca_oof, ca_test_final = train_and_predict(x2_train, y2_train, x2_valid, y2_valid, test2, label="Categorical")

final_oof = 0.5 * nu_oof + 0.5 * ca_oof
final_test_pred = 0.5 * nu_test_final + 0.5 * ca_test_final

print_scores(y1_valid, final_oof, "Final OOF Ensemble")

# 제출
submit = pd.read_csv('./Drug/sample_submission.csv')
submit['Inhibition'] = final_test_pred
os.makedirs("./Drug/_03/", exist_ok=True)
submit.to_csv("./Drug/_03/final_submission.csv", index=False)
print("\n📁 최종 제출 저장 완료")