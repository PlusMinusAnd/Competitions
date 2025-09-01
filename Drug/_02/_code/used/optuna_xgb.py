# optuna_xgb.py

import optuna
import xgboost as xgb
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
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "random_state": r,
        "verbosity": 0,
        "n_jobs": -1
    }

    oof = np.zeros(len(X))
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBRegressor(**params, early_stopping_rounds=20)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            
            verbose=False
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

# 🎉 Best Score: 0.6115461664254853
# ✅ Best Params:
# learning_rate: 0.04018614797174392
# max_depth: 3
# subsample: 0.7086680178296805
# colsample_bytree: 0.6201094943343927
# reg_alpha: 3.056666214937805
# reg_lambda: 2.370292354775882