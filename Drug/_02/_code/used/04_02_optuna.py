
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")

# 데이터 로드
train = pd.read_csv("./Drug/_02/0_dataset/train_descriptor.csv")
X = train.drop(columns=["Inhibition"])
y = train["Inhibition"].values

kf = KFold(n_splits=5, shuffle=True, random_state=73)

def objective(trial):
    model_name = trial.suggest_categorical("model", ["lgbm", "xgb", "cat"])

    if model_name == "lgbm":
        params = {
            "n_estimators": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0)
        }
        model = LGBMRegressor(**params)

    elif model_name == "xgb":
        params = {
            "n_estimators": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        model = XGBRegressor(**params, verbosity=0)

    elif model_name == "cat":
        params = {
            "iterations": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        }
        model = CatBoostRegressor(**params, verbose=0)

    oof = np.zeros(len(X))

    for tr_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_train, y_val = y[tr_idx], y[val_idx]

        model.fit(X_train, y_train)
        oof[val_idx] = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y, oof))
    nrmse = rmse / (np.max(y) - np.min(y))
    pearson = np.corrcoef(y, oof)[0, 1]
    score = 0.5 * (1 - min(nrmse, 1)) + 0.5 * np.clip(pearson, 0, 1)

    return -score  # Optuna는 최소화를 시도하므로 부호 반전

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

all_trials = study.trials
best_by_model = {}

for trial in all_trials:
    if trial.state.name != "COMPLETE":
        continue
    model_name = trial.params.get("model")
    if model_name not in best_by_model or trial.value < best_by_model[model_name].value:
        best_by_model[model_name] = trial

# 출력
print("\n✅ 모델별 Best Trial 요약:")
for model, trial in best_by_model.items():
    print(f"\n📌 [{model.upper()}] Score: {-trial.value:.5f}")
    for k, v in trial.params.items():
        print(f"{k:20}: {v}")

# 📌 [LGBM] Score: 0.59902
# model               : lgbm
# learning_rate       : 0.015875869112729403
# max_depth           : 5
# num_leaves          : 77
# min_child_samples   : 21
# colsample_bytree    : 0.8956019941712514

# 📌 [CAT] Score: 0.60728
# model               : cat
# learning_rate       : 0.020782688572110002
# depth               : 4
# l2_leaf_reg         : 9.322000582507451

# 📌 [XGB] Score: 0.60642
# model               : xgb
# learning_rate       : 0.010607554597937956
# max_depth           : 3
# subsample           : 0.7961002707424782
# colsample_bytree    : 0.8105132963922017