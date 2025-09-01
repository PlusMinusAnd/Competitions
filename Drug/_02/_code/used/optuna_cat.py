# optuna_cat.py

import optuna
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# 랜덤 시드
r = 394
np.random.seed(r)

# 📌 평가 함수 정의
def custom_score(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    pearson = pearsonr(y_true, y_pred)[0]
    return 0.5 * (1 - min(nrmse, 1)) + 0.5 * np.clip(pearson, 0, 1)

# 📂 데이터 로딩
train = pd.read_csv("./Drug/_02/full_pipeline/0_dataset/train_descriptor.csv")
X = train.drop(columns=["Inhibition"])
y = train["Inhibition"].values

kf = KFold(n_splits=5, shuffle=True, random_state=r)

# 🎯 Optuna 목적 함수
def objective(trial):
    params = {
        "iterations": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05),
        "depth": trial.suggest_int("depth", 3, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "random_seed": r,
        "verbose": 0
    }

    oof = np.zeros(len(X))
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=20, use_best_model=True)
        oof[val_idx] = model.predict(X_val)

    return custom_score(y, oof)

# 🔍 Optuna 실행
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=r))
study.optimize(objective, n_trials=50)

# ✅ 결과 출력
print("🎉 Best Score:", study.best_value)
print("✅ Best Params:")
for k, v in study.best_params.items():
    print(f"{k}: {v}")


# ✅ Best Params:
# learning_rate: 0.03905462846751998
# depth: 4
# l2_leaf_reg: 6.738080359145052
# bagging_temperature: 0.4929876007310619