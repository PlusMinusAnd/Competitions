# optuna_lgbm.py

import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
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
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": r,
        "verbosity": -1,
        "n_jobs": -1,
    }

    oof = np.zeros(len(X))
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(period=0)]
        )

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

# 🎉 Best Score: 0.6091349434843168
# ✅ Best Params:
# learning_rate: 0.024615467196152204
# max_depth: 3
# num_leaves: 31
# min_child_samples: 20
# colsample_bytree: 0.6411145386746391